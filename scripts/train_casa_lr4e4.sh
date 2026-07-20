#!/bin/bash
# CASA with the acoustic-branch LoRA learning rate DOUBLED to 4e-4 (variance study, "4e-4" in the
# paper). Everything else identical to train_casa.sh. This configuration is both worse on average
# and markedly less stable than 2e-4 - it is released for reproducibility, not as a recommended
# setting. Vary --seed to reproduce the 10-run protocol (see README).
#   bash scripts/train_casa_lr4e4.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
[ "${USE_WANDB:-0}" = "1" ] || EXTRA_ARGS+=(--no_wandb)

python train.py \
    --train_csv csv/master_train_asr.csv \
    --eval_csv csv/master_dev_asr.csv \
    --test_csv csv/master_test_asr.csv \
    --whisper_name openai/whisper-medium \
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
    --output_dir checkpoints/casa-lr4e4 \
    --project_name sandi_scorer \
    --run_name casa-lr4e4 \
    --epochs 5 \
    --seed 1011 \
    --early_stopping_patience 3 \
    --batch_size 16 \
    --grad_accum 2 \
    --whisper_lr 4e-4 \
    --qwen_lr 1e-4 \
    --head_lr 5e-5 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --num_workers 8 \
    "${EXTRA_ARGS[@]}"
