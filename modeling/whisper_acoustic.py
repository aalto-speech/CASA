"""Whisper(+LoRA) acoustic-prior encoder for the ASA scorer (RoPE).

Concatenates the per-chunk encodings, averages adjacent pairs of frames (`frame_pool=2`:
20 ms -> 40 ms, halving the sequence at negligible cost), then pools with a learnable
`[CLS]` Transformer to give one vector per scoring unit. Avoiding a global average pool
this way preserves acoustic detail that a single mean would discard. On top of that:

  * `task_emb`   — learned embedding per task/part/ROLE. SANDI: part P1/P3/P4/P5; Finnish:
                   task_id 01-06; dialogue (future): speaker role mono / dialogue-A / -B. 
                   By default, there are 16 rows, which is generous for SANDI (4) 
                   and Finnish (6) and dialogue (2-3). Those not used remain zero and inert.                    
  * `answer_emb` — learned per-frame "which segment" embedding (SANDI: answer Q1..Qk;
                   dialogue: turn index), computed EXACTLY at frame resolution from the
                   clip/segment-boundary times the dataset passes in.

Both are ADDED to the frames before the aggregator (Stage 2), like positional encoding —
not masks. For single-segment data (DTA monologue) the index is constant 0 -> `answer_emb`
inert -> the aggregator reduces to plain `[CLS]` pooling. Position encoding is RoPE
(relative, length-free — good for long/variable dialogue streams). Whisper encoder runs at
50 frames/sec (1500/30s). Returns (B, D).

Dialogue-ready (NOT implemented here): a per-speaker stream IS a monologue, so dialogue
reuses this encoder unchanged — just gather one speaker's diarized turns as the segment list
(role via `task_id`, turns via segment boundaries). Embedding tables are sized generously so
dialogue turn counts / roles fit without resizing.
"""

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

WHISPER_ENC_FPS = 50.0   # encoder frames per second (1500 frames / 30 s chunk)


# --------------------------------------------------------------------------- RoPE
"""
RoPE (Rotary Positional Embedding) implementation.
Put it here rather than relying on the HF `transformers` implementation.
"""
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim, self.dropout = n_heads, d_model // n_heads, dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.ffn(self.norm2(x))
        return x


class RoPEEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [RoPEEncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.norm(x)


# ------------------------------------------------------------------ acoustic encoder
# EXAMPLE — courtesy of Claude, do not trust it 100%
# one P1 response of 3 answers (2.5 s, 2.5 s, 2.0 s), whisper-medium.
# Everything below is the same data at successive stages; only the SHAPE changes.
#
# 1. AUDIO -> CHUNKS.  7 s of audio -> C=1 chunk of 30 s (padded).  A 95 s response would
#    give C=4 (capped by max_chunks; anything past 2 min is dropped).
#       input_features        (C=1, 80, 3000)      80 mel bins x 3000 mel frames
#
# 2. WHISPER ENCODER -> FRAMES.  Each 30 s chunk becomes 1500 vectors (50 fps = one per
#    20 ms); each vector has D=1024 numbers describing that instant of audio.
#       hidden                (C=1, 1500, 1024)  -> reshape -> (1500, 1024)
#
# 3. POOL x2.  Average adjacent pairs: 20 ms -> 40 ms frames, half as many. Cheaper, and
#    40 ms is still far shorter than any phoneme, so ~nothing is lost.
#       hidden                (750, 1024)          <- this sample's T = 750
#
# 4. SEGMENT IDS.  Which answer does each frame belong to?  clip_bounds holds cumulative
#    clip END times [2.5, 5.0, 7.0] s; frame i sits at t = i * 40 ms; searchsorted maps
#    t -> answer index:
#       frame   0 (t=0.00s) -> 0        frame 100 (t=4.00s) -> 1
#       frame  62 (t=2.48s) -> 0        frame 150 (t=6.00s) -> 2
#       frame  63 (t=2.52s) -> 1        (crossed the 2.5 s boundary)
#       ans                   (750,)
#
# 5. PAD THE BATCH.  T is 750 per chunk, so a longer response has more chunks and more
#    frames; a batch mixes them. A tensor must be rectangular, so allocate (B, T_max, D)
#    zeros and copy each sample in. With B=2, sample0 = 2 chunks and sample1 = 1 chunk:
#       padded[0, :1500] = sample0    padded[1, :750] = sample1
#       padded                (2, 1500, 1024)    rows 750..1499 of sample1 stay ZERO
#       pad_mask              (2, 1500)          True exactly where nothing was written
#    The mask matters: attention must ignore those invented frames.
#
# 6. ADD THE TAGS.  Both tables are (n_rows, 1024) — a row is as long as a frame, because
#    it is added into the frame ELEMENTWISE. This is the only place conditioning enters.
#       answer_emb(ans_pad)   (2, 1500, 1024)    a DIFFERENT row per frame -> varies in time
#       task_emb(task_ids)    (2,    1, 1024)    ONE row per sample, broadcast to all frames
#    Broadcasting copies that single row across all 1500 positions; the result stays
#    (2, 1500, 1024) — nothing is appended and no dimension grows.
#
#    WHAT THE NUMBERS ACTUALLY DO. Shrunk to 6 frames x D=4 so the matrix fits. Frames 0-2
#    are answer 0, frames 3-5 are answer 1 (ans ids [0,0,0,1,1,1]); the sample is part P1.
#
#      BEFORE                  answer_emb table            task_emb row (P1)
#      [1, 2, 3, 4]            id 0 = [0.5, 0,   0,  0]    [0, 0.2, 0, 0.2]
#      [1, 2, 3, 4]            id 1 = [0,   0, 0.5,  0]
#      [1, 2, 3, 4]
#      [5, 6, 7, 8]
#      [5, 6, 7, 8]
#      [5, 6, 7, 8]
#
#      + answer_emb            -> AFTER               (col 0 vs col 2 changed)
#      frames 0-2 use id 0        [1.5, 2, 3,   4]    <- +0.5 in COLUMN 0
#      frames 0-2 use id 0        [1.5, 2, 3,   4]
#      frames 0-2 use id 0        [1.5, 2, 3,   4]
#      frames 3-5 use id 1        [5,   6, 7.5, 8]    <- +0.5 in COLUMN 2 instead
#      frames 3-5 use id 1        [5,   6, 7.5, 8]
#      frames 3-5 use id 1        [5,   6, 7.5, 8]
#      The two answer groups moved in DIFFERENT directions. That is the whole point: the
#      aggregator's attention can now tell "this frame is from answer 1, that one isn't",
#      which raw Whisper frames do not encode.
#
#      + task_emb              -> AFTER               (cols 1 and 3, EVERY row)
#                                 [1.5, 2.2, 3,   4.2]
#                                 [1.5, 2.2, 3,   4.2]
#                                 [1.5, 2.2, 3,   4.2]
#                                 [5,   6.2, 7.5, 8.2]
#                                 [5,   6.2, 7.5, 8.2]
#                                 [5,   6.2, 7.5, 8.2]
#      Same shift on all 6 rows, because one row is broadcast. A P3 sample would get a
#      different row, so P1 and P3 land in different regions of the space.
#
#    NOTE ON REAL SCALE. The numbers above are large and round so the change is visible.
#    Reality it was a flop: in a trained CASA checkpoint the learned tags end up
#    negligible next to the frames they are added to, barely shifting them at all. Rows that
#    are never looked up (task 4-15, answer 7-31 on SANDI) stay exactly 0.0 — nn.Embedding
#    only sends gradient to rows that were used. Why the used rows stay so small is an open
#    question, not settled here.
#
# 7. PREPEND [CLS] AND AGGREGATE.  A learnable vector is put at position 0, the 2-layer RoPE
#    Transformer contextualizes everything, and position 0 is read out as the summary.
#       x                     (2, 1501, 1024) ->  aggregator  ->  x[:, 0]  =  (2, 1024)
#
# The returned (B, 1024) is the acoustic prior: it feeds the MLP projector (-> 4 soft tokens
# for Qwen) and the aux head (-> an acoustic-only CEFR estimate).
#
class WhisperAcousticEncoder(nn.Module):
    """
    Whisper(+LoRA) acoustic encoder for the CEFR scorer (RoPE).
    B = batch, C = 30 s chunks, T = frames after pooling, D = depth (Whisper d_enc).
    See the WORKED EXAMPLE above for the shapes at each stage.
    """
    def __init__(
        self,
        whisper_name: str = "openai/whisper-medium",
        max_chunks: int = 4,               # 4 * 30s = 120s (dialogue: raise to ~8 or 10)
        frame_pool: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_encoding: str = "rope",        # "rope" (default) | "sinusoidal"
        lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target: str = "attn_ffn",
        attn_impl: str = "sdpa",           # "sdpa" (flash kernels, no extra pkg) | "flash_attention_2" | "eager"
        grad_checkpoint: bool = True,
        n_tasks: int = 16,                 # task/part/role ids (generous for dialogue roles)
        n_answers: int = 32,               # segment/turn ids (generous for dialogue turns)
        use_task_emb: bool = True,
        use_answer_emb: bool = True,
        cache_dir: str | None = os.environ.get("HF_CACHE_DIR") or None,
    ):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(
            whisper_name, cache_dir=cache_dir, attn_implementation=attn_impl)
        self.hidden_size = self.whisper.config.d_model
        self.max_chunks = max_chunks
        self.frame_pool = frame_pool
        self.n_answers = n_answers
        self.use_task_emb = use_task_emb
        self.use_answer_emb = use_answer_emb

        if lora:
            from peft import LoraConfig, get_peft_model
            targets = {
                "attn": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "ffn": ["fc1", "fc2"],
                "attn_ffn": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            }[lora_target]
            self.whisper = get_peft_model(
                self.whisper,
                LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           target_modules=targets, bias="none"))
        if grad_checkpoint:
            self.whisper.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.whisper, "enable_input_require_grads"):
                self.whisper.enable_input_require_grads()

        # Conditioning tables: (n_rows, D) lookup tables, one row per id. A row is the SAME
        # length as a frame (D) because forward() ADDS it into the frame vectors, elementwise,
        # like a positional encoding — they are never passed to the aggregator as separate
        # inputs. By the time the aggregator runs, the tag is already part of each frame.
        #   task_emb   row = which PART (task_ids, one per sample) -> broadcast onto every
        #              frame of that sample, so it is constant along time.
        #   answer_emb row = which CLIP/ANSWER a frame falls in. _encode_chunks() derives that
        #              per frame by searchsorted(clip_bounds, frame_time): a frame at 3.2 s with
        #              clip end-times [2.5, 5.0, 7.0] lands in answer 1. So this one VARIES
        #              along time, which is why it can carry more than a constant offset.
        # Rows are zero-init (start inert) and learned by gradient descent like any weight; the
        # id -> row lookup is given by the data, only the row CONTENTS are learned. Rows that are
        # never looked up get no gradient and stay exactly zero, so an over-sized table is free:
        # SANDI uses task rows 0-3 and answer rows 0-6; the rest remain 0.
        self.task_emb = nn.Embedding(n_tasks, self.hidden_size) if use_task_emb else None      # (n_tasks, D)
        self.answer_emb = nn.Embedding(n_answers, self.hidden_size) if use_answer_emb else None  # (n_answers, D)

        # Initialize the embeddings to zero so they start inert and learn the conditioning.
        # This is important for both monologue and dialogue: 
        # for monologue, the SANDI corpus have multiple parts (P1/P3/P4/P5) 
        # and we don't want to bias the model to prefer one part over another.
        # For dialogue, the turn index is used to condition the acoustic prior. 
        # If it starts non-zero, it can bias the model to prefer one turn over another,
        # , which is not desired.

        # However, NOTE that the embeddings did not contributed much with their 
        # weight pretty close to zero after training. So we should consider 
        # different ways to condition the acoustic prior on the task/answer.
        # Also, due to the different task format, the task themselves in SANDI
        # can easily be inferred from the answer, so the task embedding 
        # may not be that useful. I do belive that being able to differentiate the task
        # is important for the model to learn the acoustic prior.
        # For the content branch, the task ID is already passed in as a token.

        for emb in (self.task_emb, self.answer_emb):
            if emb is not None:
                nn.init.zeros_(emb.weight)            # start inert; learn the conditioning

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.pos_enc = None
            self.aggregator = RoPEEncoder(n_layers, self.hidden_size, n_heads, dim_feedforward, dropout)
        elif pos_encoding == "sinusoidal":
            self.pos_enc = _Sinusoidal(self.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True)
            self.aggregator = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            raise ValueError(f"pos_encoding must be 'rope' or 'sinusoidal', got {pos_encoding!r}")

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.whisper.parameters()):
            self.whisper.eval()
        return self

    def _whisper_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Encode ONE sample's chunks. Shapes below use: C = 30 s chunks for this sample
        (<= max_chunks; there is no batch dim here — forward() loops over the batch),
        1500 = frames per chunk (50 fps x 30 s), D = Whisper d_model/d_enc (1024 for medium)."""
        frozen = not any(p.requires_grad for p in self.whisper.parameters())
        with torch.no_grad() if frozen else contextlib.nullcontext():
            return self.whisper.encoder(feats).last_hidden_state          # (C, 1500, D)

    def _encode_chunks(self, feats: torch.Tensor, bounds: torch.Tensor | None):
        feats = feats[: self.max_chunks]
        hidden = self._whisper_features(feats)                            # (C, 1500, D)
        C, T, D = hidden.shape
        hidden = hidden.reshape(C * T, D)
        if self.frame_pool > 1:
            hidden = hidden.transpose(0, 1).unsqueeze(0)
            hidden = F.avg_pool1d(hidden, self.frame_pool, self.frame_pool)
            hidden = hidden.squeeze(0).transpose(0, 1)                    # (Tp, D)

        Tp = hidden.shape[0]
        if self.use_answer_emb and bounds is not None and bounds.numel() > 0:
            t = torch.arange(Tp, device=hidden.device) * (self.frame_pool / WHISPER_ENC_FPS)
            ans = torch.searchsorted(bounds.to(hidden.device), t, right=True).clamp(max=self.n_answers - 1)
        else:
            ans = torch.zeros(Tp, dtype=torch.long, device=hidden.device)
        return hidden, ans

    def forward(self, input_features, clip_bounds=None, task_ids=None, return_frames=False):
        """SHAPE LEGEND (used throughout this file):
            B   batch — how many part-responses are encoded together (CASA: 16)
            C_i 30 s chunks of audio for sample i (<= max_chunks, so <= 4 = 2 min)
            T   time — frames after pooling, per sample (<= C*1500/frame_pool = 3000)
            D   depth — numbers describing ONE frame = Whisper d_enc (1024 for medium)
            k_i number of clips/answers in sample i (P1 ~6, P3/P4 = 1, P5 ~5)

        input_features: list[B] of (C_i,80,3000)   — 80-bin mel, 3000 mel frames per 30 s chunk
        clip_bounds:    list[B] of (k_i,)          — cumulative clip END times in seconds, or None
        task_ids:       (B,) long or None          — which part, via PART_TO_TASK_ID {1:0,3:1,4:2,5:3}

        Returns (B, D) — the CLS acoustic prior. If return_frames=True, additionally returns
        the contextualized per-frame sequence and its padding mask:
        (cls (B,D), frames (B,T,D), frame_pad_mask (B,T) True=pad) — for frame-level dim heads
        (e.g. the Finnish multidim scorer). English callers omit the flag -> unchanged."""
        device = next(self.parameters()).device
        if clip_bounds is None:
            clip_bounds = [None] * len(input_features)

        seqs, ans_ids = [], []
        for feats, b in zip(input_features, clip_bounds):
            s, a = self._encode_chunks(feats.to(device), None if b is None else b.to(device))
            seqs.append(s)
            ans_ids.append(a)

        lengths = [s.shape[0] for s in seqs]
        T_max, B, D = max(lengths), len(seqs), self.hidden_size
        padded = torch.zeros(B, T_max, D, device=device, dtype=seqs[0].dtype)
        ans_pad = torch.zeros(B, T_max, dtype=torch.long, device=device)
        pad_mask = torch.ones(B, T_max, dtype=torch.bool, device=device)
        for i, (s, a) in enumerate(zip(seqs, ans_ids)):
            padded[i, : lengths[i]] = s
            ans_pad[i, : lengths[i]] = a
            pad_mask[i, : lengths[i]] = False

        # ADD the answer + task/role embeddings to the frames (not masks). This is the ONLY place
        # the conditioning enters: it is summed into the frame vectors here, so the aggregator
        # below just sees frames that already carry it.
        if self.answer_emb is not None:
            # ans_pad is a per-frame segment id -> a DIFFERENT row per timestep.
            padded = padded + self.answer_emb(ans_pad).to(padded.dtype)           # (B,T,D) + (B,T,D)
        if self.task_emb is not None and task_ids is not None:
            # one id per sample -> unsqueeze(1) broadcasts the SAME row onto every frame.
            padded = padded + self.task_emb(task_ids.to(device)).unsqueeze(1).to(padded.dtype)  # (B,T,D) + (B,1,D)

        # [CLS] POOLING vs AVERAGE POOLING — why this is not just frames.mean(1).
        #   Average pooling gives EVERY frame weight 1/T, fixed, no matter what it contains.
        #   Which means unique acoustic features are blurred out. A single frame with a
        #   pronunciation error is diluted by the other frames and the error is lost. Which means
        #   higher proficiency speakers (occasionally errors) are not distinguished from 
        #   lower proficiency speakers (frequent errors). 
        #   We use average pooling in the One Whisper to Grade Them All paper, it was a trick
        #   to use the least amount of parameter as possible, but admittedly it is not the best
        #   way to summarize the frames. In this paper, we explose this idea of [CLS] from BERT.
        #   [CLS] does a LEARNED weighted average: the [CLS] vector is a query, each frame is a
        #   key, query·key -> softmax -> a weight per frame, output = weighted sum of frames.
        #   So it can put most of the weight on the frames that matter and little on the rest.
        #
        #   ILLUSTRATIVE TOY ONLY — invented numbers to show the mechanism, NOT anything measured
        #   from CASA. Here 4 frames and D=4; the REAL model has up to 3000 frames (750 per 30 s
        #   chunk, up to 4 chunks concatenated end-to-end) each of length D=1024.
        #
        #                    dim0  dim1  dim2  dim3
        #       frame 0        1     1     0     0      fluent speech
        #       frame 1        1     1     0     0      fluent speech
        #       frame 2        0     0     1     0      silence
        #       frame 3        0     0     0     9      the pronunciation error
        #
        #   AVERAGE POOL — every frame weight = 1/4, fixed:
        #       summary = [0.5, 0.5, 0.25, 2.25]   <- the 9 got diluted to 2.25, the error is lost
        #
        #   [CLS] POOL — weights are LEARNED per frame (query . key -> softmax), not fixed:
        #       query [0,0,0,3] (look in dim 3) -> weights [0, 0, 0, 1]   -> summary [0,0,0,9]
        #                                                  keeps the full 9 — error PRESERVED
        #       query [0,0,3,0] (look in dim 2) -> weights [.13,.13,.60,.13] -> frame 2 (silence)
        #                                                  dominates instead
        #   Same audio, two different readouts — average pooling, with its frozen 1/4, can never
        #   do that. (Real attention projects through q/k/v_proj first and runs 8 heads x 2 layers;
        #   the principle is identical.)
        #
        #   With [CLS] pooling, the aggregator can learn to look for the frames that matter instead
        #   of treating them all equally.

        cls = self.cls_token.expand(B, 1, D).to(padded.dtype)
        x = torch.cat([cls, padded], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pad_mask], dim=1)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        # x[:, 0] is [CLS]. After the aggregator its position-0 output is read out as the summary.
        x = self.aggregator(x, src_key_padding_mask=mask)
        if return_frames:
            # x[:, 0] = CLS prior; x[:, 1:] = contextualized per-frame states; pad_mask True=pad
            return x[:, 0], x[:, 1:], pad_mask
        return x[:, 0]                                                    # (B, D) acoustic prior/summary


class _Sinusoidal(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        _, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=x.device) * (-math.log(10000.0) / D))
        pe = torch.zeros(T, D, device=x.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return x + pe.unsqueeze(0)
