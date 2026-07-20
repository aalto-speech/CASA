"""Raw-waveform SSL (wav2vec2 / WavLM / HuBERT-family) acoustic-prior encoder.

Drop-in sibling of `WhisperAcousticEncoder` for the ASA scorer: same RoPE `[CLS]` aggregator,
same task/answer embeddings, same soft-token contract (returns (B, D)) — the ONLY difference is
the frozen backbone. Instead of a whisper log-mel encoder it wraps a self-supervised speech
encoder that consumes RAW WAVEFORM (`input_values`, 16 kHz) and emits frame states at ~50 fps
(conv stride 320 -> 16000/320 = 50, matching `WHISPER_ENC_FPS` so the answer-embedding
frame->segment time mapping is unchanged).

Used purely as the ACOUSTIC branch — the ASR transcript fed to Qwen still comes from
whisper-medium. Holding the transcript fixed and swapping only this encoder isolates the
contribution of the acoustic representation (whisper=ASR-oriented vs XLS-R=multilingual vs
WavLM=speaker/paralinguistic).

Backbone is loaded frozen; a LoRA adapter (r=whisper_lora_r) is added to the transformer
attention + feed-forward projections (`q/k/v/out_proj`, `intermediate_dense/output_dense`), and
the convolutional feature extractor is frozen outright (standard SSL fine-tuning practice).
hidden_size (1024 for the -large variants) matches the whisper-medium encoder, so the acoustic
projector / aggregator dims are identical.

Honestly I didn't pay attention much to this, it seems to work fine with the default hyperparameters. 
The only thing that matters is that the performance of the SSL models are not as good as Whisper,
so I didn't spend much time for it (for now).
"""

import contextlib

import torch
import torch.nn.functional as F
from transformers import AutoModel

from .whisper_acoustic import RoPEEncoder, _Sinusoidal, WHISPER_ENC_FPS

# SSL encoders also run at 50 fps (conv total stride 320 @ 16 kHz) -> reuse the constant.
SSL_ENC_FPS = WHISPER_ENC_FPS


class SSLAcousticEncoder(torch.nn.Module):
    """Same signature/behaviour as WhisperAcousticEncoder, backed by a raw-waveform SSL model."""

    def __init__(
        self,
        whisper_name: str = "facebook/wav2vec2-xls-r-300m",   # kept named `whisper_name` for drop-in
        max_chunks: int = 4,
        frame_pool: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target: str = "attn_ffn",
        attn_impl: str = "sdpa",
        grad_checkpoint: bool = True,
        n_tasks: int = 16,
        n_answers: int = 32,
        use_task_emb: bool = True,
        use_answer_emb: bool = True,
        layer_sum: bool = False,       # SUPERB-style learnable weighted sum over ALL layers
        cache_dir: str = "/scratch/elec/t412-slaam/hf_cache/hub",
    ):
        super().__init__()
        # WavLM's gated relative-position attention has no sdpa path in some transformers builds;
        # fall back to eager rather than crash.
        try:
            self.ssl = AutoModel.from_pretrained(
                whisper_name, cache_dir=cache_dir, attn_implementation=attn_impl)
        except (ValueError, TypeError, KeyError):
            self.ssl = AutoModel.from_pretrained(whisper_name, cache_dir=cache_dir)
        self.hidden_size = self.ssl.config.hidden_size
        self.max_chunks = max_chunks
        self.frame_pool = frame_pool
        self.n_answers = n_answers
        self.use_task_emb = use_task_emb
        self.use_answer_emb = use_answer_emb

        # Masked-prediction SSL models are LAYER-SPECIALIZED (final layers drift back toward the
        # pretraining objective and transfer poorly). Instead of the last layer only, learn a
        # softmax-weighted sum over all hidden states (embeddings + every transformer layer) --
        # the SUPERB-standard probe. The learned weights also reveal which layers matter.
        self.layer_sum = layer_sum
        if layer_sum:
            n_hidden = self.ssl.config.num_hidden_layers + 1
            self.layer_weights = torch.nn.Parameter(torch.zeros(n_hidden))

        # Freeze the convolutional feature extractor (standard practice; it carries no LoRA).
        if hasattr(self.ssl, "freeze_feature_encoder"):
            self.ssl.freeze_feature_encoder()

        if lora:
            from peft import LoraConfig, get_peft_model
            targets = {
                "attn": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "ffn": ["intermediate_dense", "output_dense"],
                "attn_ffn": ["q_proj", "k_proj", "v_proj", "out_proj",
                             "intermediate_dense", "output_dense"],
            }[lora_target]
            self.ssl = get_peft_model(
                self.ssl,
                LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           target_modules=targets, bias="none"))
        if grad_checkpoint:
            self.ssl.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.ssl, "enable_input_require_grads"):
                self.ssl.enable_input_require_grads()

        self.task_emb = torch.nn.Embedding(n_tasks, self.hidden_size) if use_task_emb else None
        self.answer_emb = torch.nn.Embedding(n_answers, self.hidden_size) if use_answer_emb else None
        for emb in (self.task_emb, self.answer_emb):
            if emb is not None:
                torch.nn.init.zeros_(emb.weight)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            self.pos_enc = None
            self.aggregator = RoPEEncoder(n_layers, self.hidden_size, n_heads, dim_feedforward, dropout)
        elif pos_encoding == "sinusoidal":
            self.pos_enc = _Sinusoidal(self.hidden_size)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True)
            self.aggregator = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            raise ValueError(f"pos_encoding must be 'rope' or 'sinusoidal', got {pos_encoding!r}")

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.ssl.parameters()):
            self.ssl.eval()
        return self

    def _ssl_features(self, input_values: torch.Tensor) -> torch.Tensor:
        frozen = not any(p.requires_grad for p in self.ssl.parameters())
        with torch.no_grad() if frozen else contextlib.nullcontext():
            if self.layer_sum:
                hs = self.ssl(input_values, output_hidden_states=True).hidden_states
            else:
                return self.ssl(input_values).last_hidden_state         # (C, T, D)
        # SUPERB-style softmax-weighted sum over all hidden states (embeddings + every layer).
        # Done outside the frozen no_grad block so layer_weights still receive gradient.
        stacked = torch.stack(hs, dim=0)                                # (L+1, C, T, D)
        w = torch.softmax(self.layer_weights, dim=0).to(stacked.dtype)
        return (w.view(-1, 1, 1, 1) * stacked).sum(0)                   # (C, T, D)

    def _encode_chunks(self, feats: torch.Tensor, bounds: torch.Tensor | None):
        feats = feats[: self.max_chunks]                                # (C, samples)
        hidden = self._ssl_features(feats)                              # (C, T, D)
        C, T, D = hidden.shape
        hidden = hidden.reshape(C * T, D)
        if self.frame_pool > 1:
            hidden = hidden.transpose(0, 1).unsqueeze(0)
            hidden = F.avg_pool1d(hidden, self.frame_pool, self.frame_pool)
            hidden = hidden.squeeze(0).transpose(0, 1)                  # (Tp, D)

        Tp = hidden.shape[0]
        if self.use_answer_emb and bounds is not None and bounds.numel() > 0:
            t = torch.arange(Tp, device=hidden.device) * (self.frame_pool / SSL_ENC_FPS)
            ans = torch.searchsorted(bounds.to(hidden.device), t, right=True).clamp(max=self.n_answers - 1)
        else:
            ans = torch.zeros(Tp, dtype=torch.long, device=hidden.device)
        return hidden, ans

    def forward(self, input_features, clip_bounds=None, task_ids=None, return_frames=False):
        """input_features: list[B] of (C_i, samples); clip_bounds: list[B] of (k_i,) cumulative
        segment end-times in seconds (or None); task_ids: (B,) long or None. Returns (B, D)."""
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

        if self.answer_emb is not None:
            padded = padded + self.answer_emb(ans_pad).to(padded.dtype)
        if self.task_emb is not None and task_ids is not None:
            padded = padded + self.task_emb(task_ids.to(device)).unsqueeze(1).to(padded.dtype)

        cls = self.cls_token.expand(B, 1, D).to(padded.dtype)
        x = torch.cat([cls, padded], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pad_mask], dim=1)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.aggregator(x, src_key_padding_mask=mask)
        if return_frames:
            return x[:, 0], x[:, 1:], pad_mask
        return x[:, 0]                                                   # (B, D) acoustic prior
