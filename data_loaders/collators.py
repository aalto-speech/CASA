"""Collator for the SANDI Whisper-LoRA + Qwen scorer.

Keeps `input_features` / `segment_ids` as per-sample lists (variable chunk count, as the
acoustic encoder expects) and right-pads the Qwen token ids to the batch max. Carries
`speaker_id` / `part` / `overall_score` through so eval can group by speaker -> mean parts.
"""

import torch


class SandiCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        max_len = max(x.size(0) for x in input_ids)

        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        padded_attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, am) in enumerate(zip(input_ids, attn)):
            padded_ids[i, : ids.size(0)] = ids       # right-pad; <SCORE>: cue is last real token
            padded_attn[i, : am.size(0)] = am

        collated = {
            "input_features": [b["input_features"] for b in batch],   # list[(C,80,3000)]
            "clip_bounds": [b["clip_bounds"] for b in batch],         # list[(k,)] clip end-times (s)
            "task_ids": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
            "input_ids": padded_ids,
            "attention_mask": padded_attn,
            "scores": torch.tensor([b["score"] for b in batch], dtype=torch.float32),
            "overall_scores": torch.tensor([b["overall_score"] for b in batch], dtype=torch.float32),
            "speaker_ids": [b["speaker_id"] for b in batch],
            "parts": [b["part"] for b in batch],
        }
        # Finnish multi-dimension scorer: stack the 4 analytic-dimension targets when the
        # dataset provides them. English SANDI items have no "dim_targets" -> no effect.
        if "dim_targets" in batch[0]:
            collated["dim_targets"] = torch.stack([b["dim_targets"] for b in batch])
        # per-row, per-dim supervision mask (DigiTala acoustic-only regime). Additive:
        # absent on English items and when the mask policy is off -> no effect.
        if "dim_mask" in batch[0]:
            collated["dim_mask"] = torch.stack([b["dim_mask"] for b in batch])
        if "cefr_mask" in batch[0]:
            collated["cefr_mask"] = torch.stack([b["cefr_mask"] for b in batch])
        if "dim_tol" in batch[0]:
            collated["dim_tol"] = torch.stack([b["dim_tol"] for b in batch])   # (B,) per-row dead-zone
        return collated
