#!/bin/bash
# CASA-Crisper — the exact CASA recipe with nyrahealth/CrisperWhisper (verbatim ASR) replacing
# whisper-medium as BOTH the transcript source and the acoustic encoder. Changes vs
# scripts/train_casa.sh: --whisper_name and the three master CSVs (crisper transcripts).
# Build the crisper masters first (see README):
#   python generate_asr.py --model nyrahealth/CrisperWhisper --out_subdir asr_crisper
#   python build_master.py --asr --asr_dir asr_crisper --suffix _crisper
# Then run from the repo root:
#   bash scripts/train_casa_crisper.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
[ "${USE_WANDB:-0}" = "1" ] || EXTRA_ARGS+=(--no_wandb)

python train.py \
    --train_csv csv/master_train_crisper.csv \
    --eval_csv csv/master_dev_crisper.csv \
    --test_csv csv/master_test_crisper.csv \
    --whisper_name nyrahealth/CrisperWhisper \
    --qwen_name Qwen/Qwen3.5-2B \
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
    --output_dir checkpoints/casa-crisper \
    --project_name sandi_scorer \
    --run_name casa-crisper \
    --epochs 5 \
    --seed 1011 \
    --early_stopping_patience 3 \
    --batch_size 16 \
    --grad_accum 2 \
    --whisper_lr 2e-4 \
    --qwen_lr 1e-4 \
    --head_lr 5e-5 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --num_workers 8 \
    "${EXTRA_ARGS[@]}"
