"""Fused Whisper-LoRA acoustic + Qwen-LoRA reasoning scorer (the ASA model).
The chart below was drawn by Claude. Instructed by me, but I won't be able to 
handdraw it like this. It is a good visual summary of the model's structure.

Per scoring unit (SANDI: one speaker x part):
  acoustic branch  ─► acoustic prior a (B, D_enc)     ──┐
                       │ MLP projector                  │ aux_head (Linear)
                       ▼                                ▼
  a ─► k soft tokens (B, k, Dq) ──┐          aux_logits (acoustic-only)
                                  ├─ prepend ─► Qwen3.5-LoRA(inputs_embeds) ─► hidden @ last token
  tokenized prompt (B, L, Dq) ────┘                                            │ head (Linear)
                                                                               ▼
                                                                     logits (full model)

`logits` returned as (B, 2): col 0 = full model, col 1 = acoustic-only aux head.
This lets metrics compare acoustic-branch RMSE vs full model RMSE in one training run.

Loss = MSE(logits[:,0], score) + aux_weight * auxiliary_loss.
By default auxiliary_loss is MSE. With `aux_tolerance > 0`, it is squared
distance beyond a zero-loss band around the target. NOTE that the distanced is from the 
predicted vs the distrance toward the tolerance band, not the distance from the target. 
This is to avoid the aux loss from overfit the CEFR score. 
`aux_weight=0` disables the aux head loss (aux logits still returned for monitoring).
`use_acoustic=False` drops both soft tokens and aux head (text-only ablation).
"""

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .whisper_acoustic import WhisperAcousticEncoder


def build_acoustic_encoder(acoustic_kwargs: dict | None):
    """Pick the acoustic backbone by the requested model's family. whisper -> log-mel
    WhisperAcousticEncoder; wav2vec2/wavlm/hubert-family -> raw-waveform SSLAcousticEncoder.
    Both expose the same (B, D) contract, so the rest of the scorer is agnostic."""
    kw = dict(acoustic_kwargs or {})
    name = kw.get("whisper_name", "openai/whisper-medium")
    try:
        model_type = AutoConfig.from_pretrained(name, cache_dir=HF_CACHE).model_type
    except Exception:
        model_type = "whisper"
    if model_type == "whisper":
        return WhisperAcousticEncoder(**kw)
    from .ssl_acoustic import SSLAcousticEncoder
    return SSLAcousticEncoder(**kw)



# HuggingFace model cache. None -> the default HF cache (respects HF_HOME);
# override with the HF_CACHE_DIR environment variable.
HF_CACHE = os.environ.get("HF_CACHE_DIR") or None


def tolerance_mse_loss(prediction: torch.Tensor, target: torch.Tensor,
                       tolerance: float) -> torch.Tensor:
    """Squared distance outside target +/- tolerance; MSE within tolerance is 0.
    I was thinking, maybe centralize the prediction at 0 may yield better results.
    But that would also require extra effort, because LLM can understand normal CEFR
    scores, but not the centralized ones. So I will keep it as is for now.
    """
    excess = (prediction - target).abs().sub(tolerance).clamp_min(0)
    return excess.square().mean()


class QwenAcousticScorer(nn.Module):
    def __init__(
        self,
        qwen_name: str = "Qwen/Qwen3.5-2B",
        acoustic_kwargs: dict | None = None,
        n_soft_tokens: int = 4,
        use_acoustic: bool = True,
        inject_aux_score: bool = True,  # add the Whisper-LoRA CEFR estimate as its own soft token
        aux_weight: float = 0.1,        # weight for acoustic-only aux loss; 0 = monitor only
        main_weight: float = 1.0,       # weight for the full-model main loss (schedulable at train)
        aux_tolerance: float = 1.0,     # zero aux loss inside target +/- this value
        acoustic_projector_dropout: float = 0.1,
        qwen_lora_r: int = 64,
        qwen_lora_alpha: int = 128,
        qwen_lora_dropout: float = 0.05,
        qwen_lora_target=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        attn_impl: str = "sdpa",        # "sdpa" (flash kernels, no extra pkg) | "flash_attention_2" | "eager"
        grad_checkpoint: bool = True,
        cache_dir: str = HF_CACHE,
    ):
        super().__init__()
        self.use_acoustic = use_acoustic
        self.n_soft = n_soft_tokens
        self.inject_aux_score = inject_aux_score and use_acoustic
        self.aux_weight = aux_weight
        self.main_weight = main_weight
        if aux_tolerance < 0:
            raise ValueError("aux_tolerance must be non-negative")
        self.aux_tolerance = aux_tolerance

        # --- acoustic branch -------------------------------------------------
        self.acoustic = build_acoustic_encoder(acoustic_kwargs)
        d_acoustic = self.acoustic.hidden_size

        # --- Qwen reasoning branch ------------------------------------------
        # bf16 base weights (fits an 80G GPU comfortably). peft keeps LoRA adapters in fp32
        # (autocast_adapter_dtype default) so AdamW stays stable. Needs Ampere+ (cc>=80).
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_name, cache_dir=cache_dir, dtype=torch.bfloat16, attn_implementation=attn_impl)
        d_qwen = self.qwen.config.hidden_size
        from peft import LoraConfig, get_peft_model
        self.qwen = get_peft_model(
            self.qwen,
            LoraConfig(r=qwen_lora_r, lora_alpha=qwen_lora_alpha, lora_dropout=qwen_lora_dropout,
                       target_modules=list(qwen_lora_target), bias="none", task_type="CAUSAL_LM"))
        if grad_checkpoint:
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.qwen, "enable_input_require_grads"):
                self.qwen.enable_input_require_grads()

        # --- fusion + heads --------------------------------------------------
        if use_acoustic:
            self.projector = nn.Sequential(
                nn.Linear(d_acoustic, d_qwen), nn.GELU(),
                nn.Linear(d_qwen, n_soft_tokens * d_qwen))
            # Dropout has no parameters, so keeping it out of the Sequential leaves the two
            # Linears at indices 0 and 2; inserting it between them would renumber the second
            # Linear to index 3 and rename its weights.
            self.projector_dropout = nn.Dropout(acoustic_projector_dropout)

            # Auxiliary head: reads directly from acoustic prior, bypassing Qwen entirely.
            # Trained jointly; its RMSE = what the acoustic branch alone can achieve.
            self.aux_head = nn.Linear(d_acoustic, 1)

            # The (detached) Whisper-LoRA CEFR estimate is rendered as a REAL NUMBER text line
            # ("acoustic_cefr_estimate: X.X") injected into the prompt at forward time, so Qwen
            # literally reads the acoustic branch's score. Needs a tokenizer to render it.
            if self.inject_aux_score:
                from transformers import AutoTokenizer
                self._htok = AutoTokenizer.from_pretrained(qwen_name, cache_dir=cache_dir)
                if self._htok.pad_token is None:
                    self._htok.pad_token = self._htok.eos_token
                    
        self.head = nn.Linear(d_qwen, 1)   # full model head (acoustic + Qwen)
        self.d_qwen = d_qwen
        self.loss_fn = nn.MSELoss()

    def _embed_tokens(self, input_ids):
        # Look token ids up in QWEN'S OWN embedding table -> (…, Dq). Used both for the prompt
        # and for the injected score line below, so everything lives in one embedding space.
        return self.qwen.get_input_embeddings()(input_ids)

    def _render_score_lines(self, aux_logits, device, dtype):
        """Turn the acoustic branch's CEFR estimate into a text line Qwen can read, e.g.
        "acoustic_cefr_estimate: 3.7\n\n", and return it already embedded so it can be spliced
        into the input sequence. This is how the acoustic score reaches the LLM as plain text.
        Returns (emb (B,H,Dq), mask (B,H)), H = tokens in the rendered line.

        Steps: detach (a fixed input to Qwen, not a path the main loss trains through) and clamp
        to the 0-6 CEFR range -> format one line per sample -> tokenize with Qwen's own tokenizer
        (`_htok`) so the ids are Qwen vocabulary -> embed with Qwen's table.

        `padding=True` is set for safety but in practice pads NOTHING: every line has the fixed
        form "acoustic_cefr_estimate: N.N\n\n" (clamp + :.1f -> always one digit each side of the
        dot), so all rows tokenize to the same length H. That constant length is also why the
        assembled Qwen sequence has no internal padding (see the forward-pass position_ids /
        readout comments). It still needs a pad_token defined, which __init__ provides."""
        vals = aux_logits.detach().clamp(0, 6)                            # CEFR-range for display
        lines = [f"acoustic_cefr_estimate: {v:.1f}\n\n" for v in vals.tolist()]
        enc = self._htok(lines, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        return self._embed_tokens(ids).to(dtype), mask

    def forward(self, input_ids, attention_mask, input_features=None, clip_bounds=None,
                task_ids=None, scores=None, **unused):
        # SHAPE LEGEND for the comments below:
        #   B      batch (part-responses)          k      acoustic soft tokens (n_soft_tokens, 4)
        #   L      prompt tokens                    H      tokens in the injected score line
        #   Dq     Qwen hidden size (2048)          D_enc  acoustic-encoder hidden size (1024)
        # The Qwen input sequence is [k soft tokens | H score line | L prompt] -> length k+H+L.
        tok_embeds = self._embed_tokens(input_ids)                        # (B, L, Dq)
        attn = attention_mask
        aux_logits = None

        if self.use_acoustic:
            prior = self.acoustic(input_features, clip_bounds, task_ids)  # (B, D_enc)
            # Auxiliary prediction: acoustic prior → score (no Qwen). The aux loss gradient
            # flows back through `prior`, so it trains the acoustic encoder too.
            aux_prior = prior
            aux_logits = self.aux_head(
                aux_prior.to(self.aux_head.weight.dtype)).squeeze(-1)  # (B,)
            # Full-model path: project prior to k acoustic soft tokens, prepend to Qwen input
            projected = self.projector[1](self.projector[0](prior))
            projected = self.projector_dropout(projected)
            soft = self.projector[2](projected).view(
                prior.size(0), self.n_soft, self.d_qwen)
            soft = soft.to(tok_embeds.dtype)
            parts = [soft]                                                # (B, k, Dq)
            masks = [torch.ones(attn.size(0), self.n_soft, dtype=attn.dtype, device=attn.device)]
            # inject the Whisper-LoRA CEFR estimate as a REAL-NUMBER text line right after the
            # acoustic soft tokens, before the prompt body
            if self.inject_aux_score:
                s_emb, s_mask = self._render_score_lines(aux_logits, attn.device, tok_embeds.dtype)
                parts.append(s_emb); masks.append(s_mask.to(attn.dtype))
            parts.append(tok_embeds); masks.append(attn)
            inputs_embeds = torch.cat(parts, dim=1)                       # (B, k+H+L, Dq)
            attn = torch.cat(masks, dim=1)
        else:
            inputs_embeds = tok_embeds

        # position_ids feed QWEN'S OWN RoPE (Qwen is a RoPE LLM — its every layer uses positions;
        # unrelated to the acoustic aggregator's RoPE). cumsum numbers the REAL tokens 0,1,2,...
        # so consecutive real tokens stay 1 apart even if a pad sits between them.
        # THIS IS JUST TO BE SAFE. it never actually triggers: the sequence is 
        # [soft | score | prompt]; the prompt is the LAST block and right-padded, so its padding 
        # is TRAILING (harmless), and the only mid-sequence block (the score line) is constant 
        # length so it never pads. With no internal padding, plain 0..N-1 would give the 
        # identical result here. So it's not sth important that we need to worry about, but 
        # it's a good habit to be safe.
        # Note that there is also acoustic evidence, but those are also plain text insert
        # into the prompt, so they are tokenized like any other text.
        position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0)
        out = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attn, position_ids=position_ids,
                        output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]                               # (B, k+H+L, Dq)

        # This readout issue is DIFFERENT from position_ids above: here we need to read the last
        # REAL token (the "<SCORE>:" cue), not the last token in the sequence. Samples in a batch
        # have different lengths, so short ones get right-padded to match the longest; that trailing
        # padding sits after the cue, so for shorter samples the last array index is padding.
        # flip+argmax finds the last position where attn == 1.
        last_idx = (attn.size(1) - 1) - attn.flip(1).float().argmax(dim=1)  # (B,)
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), last_idx]
        logits = self.head(pooled.to(self.head.weight.dtype)).squeeze(-1)  # (B,)

        # Stack: (B, 2) col 0 = full model, col 1 = acoustic-only aux.
        # Trainer passes this as eval_pred.predictions; metrics.py splits the columns.
        if aux_logits is not None:
            out_logits = torch.stack([logits, aux_logits], dim=1)
        else:
            # text-only mode: pad col 1 with zeros so metrics shape is always (B, 2)
            out_logits = torch.stack([logits, torch.zeros_like(logits)], dim=1)

        loss = None
        if scores is not None:
            tgt = scores.to(logits.device).float()
            main_loss = (logits.float() - tgt).square().mean()
            if aux_logits is not None and self.aux_weight > 0:
                aux_loss = tolerance_mse_loss(aux_logits.float(), tgt, self.aux_tolerance)
                loss = self.main_weight * main_loss + self.aux_weight * aux_loss
            else:
                loss = self.main_weight * main_loss

        return {"loss": loss, "logits": out_logits} if loss is not None else {"logits": out_logits}