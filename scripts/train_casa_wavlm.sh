#!/bin/bash
# CASA-WavLM — acoustic-encoder ablation. Replaces the whisper-medium ACOUSTIC ENCODER with
# microsoft/wavlm-large (self-supervised, English-focused), while the ASR transcript stays whisper-medium
# (csv/master_*_asr.csv) and the scorer stays Qwen3.5-2B. The acoustic representation is therefore
# the only difference from CASA, isolating the effect of the encoder.
# Raw-waveform SSL encoders are ~300M/1024-d, so the microbatch is halved (8x4 = effective 32,
# matching CASA's 16x2). Run from the repo root on one 80GB GPU:
#   bash scripts/train_casa_wavlm.sh
# Prerequisites: csv/master_{train,dev,test}_asr.csv and SANDI_AUDIO_ROOT (same as train_casa.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
[ "${USE_WANDB:-0}" = "1" ] || EXTRA_ARGS+=(--no_wandb)

python train.py \
    --train_csv csv/master_train_asr.csv \
    --eval_csv csv/master_dev_asr.csv \
    --test_csv csv/master_test_asr.csv \
    --whisper_name microsoft/wavlm-large \
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
    --output_dir checkpoints/casa-wavlm \
    --project_name sandi_scorer \
    --run_name casa-wavlm \
    --epochs 5 \
    --seed 1011 \
    --early_stopping_patience 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --whisper_lr 2e-4 \
    --qwen_lr 1e-4 \
    --head_lr 5e-5 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --num_workers 8 \
    "${EXTRA_ARGS[@]}"
