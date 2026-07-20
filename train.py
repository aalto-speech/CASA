"""Train the fused Whisper-LoRA + Qwen-LoRA scorer on SANDI.

Target = per-speaker, per-part, `part_score` (regression, MSE). Model-selection + headline metric
= speaker OVERALL RMSE (mean of predicted parts vs gold overall). The model also returns
an acoustic-only aux prediction (col 1); see utils.metrics / modeling.qwen_scorer. Run via
`scripts/train_casa*.sh` from the repository root.
"""

import argparse
import json
import os

import torch.nn as nn
import pandas as pd
try:
    import wandb
except ImportError:      # wandb is optional — run with --no_wandb if it is not installed
    wandb = None
from torch.optim import AdamW
from transformers import (EarlyStoppingCallback, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from data_loaders.collators import SandiCollator
from data_loaders.sandi_dataset import SandiPartDataset
from modeling.qwen_scorer import QwenAcousticScorer
from utils.metrics import coarse_cefr_label, coarse_cefr_numeric, make_compute_metrics


class ASATrainer(Trainer):
    """Trainer subclass that works around the safetensors shared-tensor error.
    Qwen3.5 ties lm_head.weight to embed_tokens.weight. Safetensors cannot
    directly serialize multiple tensor names sharing the same storage.
    Because Qwen is wrapped inside the custom CASA model, Trainer does not
    recognize this as an intentional tied weight and fails during checkpoint
    saving. We temporarily clone lm_head.weight before saving, then restore
    the tie so the live model continues sharing the parameters during training.
    
    Surely there is a better way to do this, but this is the simplest workaround I could find."""

    def _save(self, output_dir, state_dict=None):
        qwen_base = self.model.qwen.base_model.model
        lm_head = getattr(qwen_base, "lm_head", None)
        embed = getattr(getattr(qwen_base, "model", None), "embed_tokens", None)
        tied = (
            lm_head is not None and embed is not None
            and hasattr(lm_head, "weight") and hasattr(embed, "weight")
            and lm_head.weight.data_ptr() == embed.weight.data_ptr()
        )
        if tied:
            lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        try:
            super()._save(output_dir, state_dict)
        finally:
            if tied:
                lm_head.weight = embed.weight   # restore tie so training is unaffected


class LRLogger(TrainerCallback):
    """Log the 3 differential LRs (whisper / qwen / head) to wandb's train tab."""
    def on_log(self, args, state, control, logs=None, **kw):
        opt = kw.get("optimizer")
        if opt is None or logs is None or "loss" not in logs or wandb is None or wandb.run is None:
            return
        seen = {}
        for g in opt.param_groups:
            name = g.get("group_name")
            if name and name not in seen:
                seen[name] = g["lr"]
        wandb.log({f"train/lr_{k}": v for k, v in seen.items()}, step=state.global_step)


def create_optimizer(model, whisper_lr, qwen_lr, head_lr, weight_decay):
    """3 LR groups by name prefix: Whisper-LoRA / Qwen-LoRA / everything-else (projector, head,
    aggregator, embeddings)."""
    no_decay = ("bias", "norm.weight", "LayerNorm.weight", "layer_norm.weight")

    groups = {"whisper": [], "qwen": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (n.startswith("acoustic.whisper.") or n.startswith("acoustic.ssl.")
                or n == "acoustic.layer_weights"):
            key = "whisper"   # acoustic-encoder group: backbone LoRA (whisper OR ssl)
        elif n.startswith("qwen."):
            key = "qwen"
        else:
            key = "head"
        groups[key].append((n, p))
    lr = {"whisper": whisper_lr, "qwen": qwen_lr, "head": head_lr}

    param_groups = []
    for key, items in groups.items():
        for decay in (True, False):
            sel = [p for n, p in items if (not any(nd in n for nd in no_decay)) == decay]
            if sel:
                param_groups.append({"params": sel, "lr": lr[key], "group_name": key,
                                     "weight_decay": weight_decay if decay else 0.0})
    return AdamW(param_groups)


def _split_logits(raw):
    """Return full-model and optional acoustic-aux predictions from Trainer output."""
    if raw.ndim == 2 and raw.shape[1] == 2:
        return raw[:, 0], raw[:, 1]
    return raw.reshape(-1), None


def _jsonable_metrics(metrics: dict) -> dict:
    return {k: float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()}


def _save_predictions(output_dir: str, split_name: str, eval_df: pd.DataFrame, pred):
    """Save per-part predictions, speaker-level predictions, and metrics."""
    preds_main, preds_aux = _split_logits(pred.predictions)

    rows = eval_df[
        ["speaker_id", "part", "part_score", "overall_score"]
    ].copy()
    rows["pred_part"] = preds_main
    if preds_aux is not None:
        rows["pred_part_aux"] = preds_aux
    part_path = os.path.join(output_dir, f"{split_name}_part_predictions.csv")
    rows.to_csv(part_path, index=False)

    agg = {"pred_overall": ("pred_part", "mean"), "true_overall": ("overall_score", "first")}
    if preds_aux is not None:
        agg["pred_overall_aux"] = ("pred_part_aux", "mean")
    overall = rows.groupby("speaker_id").agg(**agg)
    overall["true_cefr_num"] = coarse_cefr_numeric(overall["true_overall"].to_numpy())
    overall["true_cefr"] = coarse_cefr_label(overall["true_overall"].to_numpy())
    overall["pred_cefr_num"] = coarse_cefr_numeric(overall["pred_overall"].to_numpy())
    overall["pred_cefr"] = coarse_cefr_label(overall["pred_overall"].to_numpy())
    if preds_aux is not None:
        overall["pred_cefr_num_aux"] = coarse_cefr_numeric(overall["pred_overall_aux"].to_numpy())
        overall["pred_cefr_aux"] = coarse_cefr_label(overall["pred_overall_aux"].to_numpy())
    overall_path = os.path.join(output_dir, f"{split_name}_overall_predictions.csv")
    overall.to_csv(overall_path)

    metrics = _jsonable_metrics(pred.metrics)
    metrics_path = os.path.join(output_dir, f"{split_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Saved {split_name} predictions:")
    print(" ", part_path)
    print(" ", overall_path)
    print(" ", metrics_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="csv/master_train_asr.csv")
    ap.add_argument("--eval_csv", default="csv/master_dev_asr.csv")
    ap.add_argument("--test_csv", default="",
                    help="optional held-out test master CSV. Evaluated once after dev-selected best model is loaded.")
    ap.add_argument("--whisper_name", default="openai/whisper-medium")
    ap.add_argument("--qwen_name", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--target_language", default="English",
                    help="language being assessed in the task prompts and ASR transcripts")
    ap.add_argument("--prompt_version", default="asr-content-v3",
                    help="scorer prompt identity used for CSV validation and feature-cache invalidation")
    ap.add_argument("--max_chunks", type=int, default=4)
    ap.add_argument("--frame_pool", type=int, default=2)
    ap.add_argument("--max_text_len", type=int, default=1024)
    ap.add_argument("--n_soft_tokens", type=int, default=4)
    ap.add_argument("--n_tasks", type=int, default=16,
                    help="size of the learned acoustic task-embedding table")
    ap.add_argument("--aux_weight", type=float, default=0.1,
                    help="weight for acoustic-only aux loss; 0 = monitor aux RMSE without training it")
    ap.add_argument("--aux_tolerance", type=float, default=1.0,
                    help="zero auxiliary loss within target +/- tolerance; 0 = ordinary MSE")
    ap.add_argument("--eval_only", default="",
                    help="skip training; load model.safetensors from this checkpoint dir and just "
                         "run predict on dev+test (for inference ablations)")
    ap.add_argument("--acoustic_projector_dropout", type=float, default=0.1,
                    help="dropout after the acoustic-to-Qwen projector GELU (CASA uses 0.1)")
    ap.add_argument("--whisper_lora_r", type=int, default=16)
    ap.add_argument("--qwen_lora_r", type=int, default=64)
    ap.add_argument("--qwen_lora_alpha", type=int, default=128)
    ap.add_argument("--agg_layers", type=int, default=2)
    ap.add_argument("--attn_impl", default="sdpa",
                    help="attention kernel: 'sdpa' (default, flash kernels under the hood, no extra "
                    "package) | 'flash_attention_2' (needs flash_attn installed) | 'eager'")
    ap.add_argument("--output_dir", default="checkpoints/casa")
    ap.add_argument("--run_name", default="casa")
    ap.add_argument("--project_name", default="asa_llm_sandi")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1011,
                    help="seed model initialization, data order, dropout, and Trainer RNGs")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--whisper_lr", type=float, default=2e-4)
    ap.add_argument("--qwen_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--lr_scheduler_type", default="cosine",
                    help="'cosine' (holds LR high then anneals hard — best final) | 'linear' "
                    "(steadier decay, more stable mid-training) | 'constant_with_warmup'")
    ap.add_argument("--early_stopping_patience", type=int, default=0,
                    help="stop if rmse_overall doesn't improve for N evals (0 = off). Lets you set a "
                    "high --epochs ceiling safely; load_best_model_at_end keeps the best checkpoint.")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--cache_dir", default="", help="dir for pre-extracted feature cache (.pt per item); "
                    "empty = no cache; first epoch populates it, subsequent epochs hit cache")
    return ap.parse_args()


def main():
    args = parse_args()
    # Trainer also seeds itself, but that happens after the model is constructed.
    # Seed here so LoRA adapters, projectors, and score heads are reproducible too.
    # Even with the same seed, the model will be different, I guess it is not that
    # useful. However, there are some seeds that return very bad results, so it is
    # worth keeping the seed fixed for reproducibility.
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    if wandb is None and not args.no_wandb:
        print("wandb is not installed -- continuing without experiment logging (--no_wandb).")
        args.no_wandb = True
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.project_name
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args))
        # attach the training code so the run records exact params.
        # NB: the local wandb files/ entry is a symlink, not a real backup — keep train.py in git.
        if os.path.exists("train.py"):
            wandb.save("train.py", policy="now")

    acoustic_kwargs = dict(
        whisper_name=args.whisper_name, max_chunks=args.max_chunks, frame_pool=args.frame_pool,
        n_layers=args.agg_layers, pos_encoding="rope", lora=True, lora_r=args.whisper_lora_r,
        attn_impl=args.attn_impl, grad_checkpoint=True, n_tasks=args.n_tasks)

    print("Loading datasets...")
    cache = args.cache_dir or None
    train_ds = SandiPartDataset(args.train_csv, args.whisper_name, args.qwen_name,
                                max_chunks=args.max_chunks, max_text_len=args.max_text_len,
                                target_language=args.target_language,
                                prompt_version=args.prompt_version, cache_dir=cache)
    eval_ds = SandiPartDataset(args.eval_csv, args.whisper_name, args.qwen_name,
                               max_chunks=args.max_chunks, max_text_len=args.max_text_len,
                               target_language=args.target_language,
                               prompt_version=args.prompt_version, cache_dir=cache)
    collator = SandiCollator(train_ds.tok.pad_token_id)

    print("Building scorer...")
    model = QwenAcousticScorer(
        qwen_name=args.qwen_name, acoustic_kwargs=acoustic_kwargs,
        n_soft_tokens=args.n_soft_tokens,
        aux_weight=args.aux_weight,
        main_weight=1.0,
        aux_tolerance=args.aux_tolerance,
        acoustic_projector_dropout=args.acoustic_projector_dropout,
        attn_impl=args.attn_impl,
        qwen_lora_r=args.qwen_lora_r, qwen_lora_alpha=args.qwen_lora_alpha,
        grad_checkpoint=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")

    eval_df = eval_ds.data   # keeps metrics aligned to eval order
    compute_metrics = make_compute_metrics(
        eval_df["speaker_id"].values, eval_df["overall_score"].values,
        eval_df["part"].values)
    optimizer = create_optimizer(
        model, args.whisper_lr, args.qwen_lr, args.head_lr, args.weight_decay)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch", save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=args.num_workers,
        num_train_epochs=args.epochs, warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed, data_seed=args.seed,
        logging_steps=10, bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rmse_overall",
        greater_is_better=False,
        remove_unused_columns=False, label_names=["scores"],
        report_to="none" if args.no_wandb else "wandb", run_name=args.run_name,
        save_total_limit=2,
    )
    callbacks = [LRLogger()]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = ASATrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, compute_metrics=compute_metrics, optimizers=(optimizer, None),
        callbacks=callbacks)

    if args.eval_only:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(args.eval_only, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"EVAL-ONLY: loaded {args.eval_only}/model.safetensors "
              f"(missing={len(missing)} unexpected={len(unexpected)}); skipping training.")
    else:
        print("Training...")
        trainer.train()
        print("Final eval:", trainer.evaluate())

    # At this point Trainer has reloaded the best checkpoint selected on dev
    # (`load_best_model_at_end=True`). Save dev predictions, then optionally touch test once.
    trainer.compute_metrics = make_compute_metrics(
        eval_df["speaker_id"].values, eval_df["overall_score"].values,
        eval_df["part"].values)
    pred = trainer.predict(eval_ds, metric_key_prefix="eval")
    _save_predictions(args.output_dir, "eval", eval_df, pred)

    if args.test_csv:
        print(f"Loading held-out test dataset: {args.test_csv}")
        test_ds = SandiPartDataset(args.test_csv, args.whisper_name, args.qwen_name,
                                   max_chunks=args.max_chunks, max_text_len=args.max_text_len,
                                   target_language=args.target_language,
                                   prompt_version=args.prompt_version, cache_dir=cache)
        test_df = test_ds.data
        trainer.compute_metrics = make_compute_metrics(
            test_df["speaker_id"].values, test_df["overall_score"].values,
            test_df["part"].values)
        pred = trainer.predict(test_ds, metric_key_prefix="test")
        _save_predictions(args.output_dir, "test", test_df, pred)


if __name__ == "__main__":
    main()
