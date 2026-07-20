#!/bin/bash
# Inference / evaluation only: load a trained checkpoint and score the dev + test sets
# (no training). Writes {eval,test}_metrics.json and per-part/overall prediction CSVs
# into OUT_DIR. Run from the repo root on one GPU:
#
#   bash scripts/evaluate.sh checkpoints/casa/checkpoint-885
#
# The argument is the HuggingFace checkpoint directory that contains model.safetensors
# (the best-on-dev checkpoint under checkpoints/<run>/checkpoint-*).
# Variants (must match how the checkpoint was trained):
#   QWEN_NAME=Qwen/Qwen3.5-4B                bash scripts/evaluate.sh <ckpt>   # 4B scorer
#   WHISPER_NAME=nyrahealth/CrisperWhisper CSV_SUFFIX=_crisper \
#                                            bash scripts/evaluate.sh <ckpt>   # crisper variant
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: bash scripts/evaluate.sh <checkpoint_dir containing model.safetensors>}"
QWEN_NAME="${QWEN_NAME:-Qwen/Qwen3.5-2B}"
WHISPER_NAME="${WHISPER_NAME:-openai/whisper-medium}"
CSV_SUFFIX="${CSV_SUFFIX:-_asr}"
OUT_DIR="${OUT_DIR:-${CKPT}/eval_only}"

python train.py \
    --eval_only "$CKPT" \
    --train_csv "csv/master_train${CSV_SUFFIX}.csv" \
    --eval_csv "csv/master_dev${CSV_SUFFIX}.csv" \
    --test_csv "csv/master_test${CSV_SUFFIX}.csv" \
    --whisper_name "$WHISPER_NAME" \
    --qwen_name "$QWEN_NAME" \
    --cache_dir feature_cache \
    --max_chunks 4 \
    --frame_pool 2 \
    --n_soft_tokens 4 \
    --whisper_lora_r 16 \
    --qwen_lora_r 64 \
    --qwen_lora_alpha 128 \
    --acoustic_projector_dropout 0.1 \
    --aux_weight 0.1 \
    --aux_tolerance 1.0 \
    --agg_layers 2 \
    --attn_impl sdpa \
    --output_dir "$OUT_DIR" \
    --project_name sandi_scorer \
    --run_name eval-only \
    --seed 1011 \
    --batch_size 16 \
    --num_workers 8 \
    --no_wandb
