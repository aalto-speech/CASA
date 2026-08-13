# CASA: Content–Acoustic Speaking Assessment with a Speech Encoder and an LLM

Code release for CASA, an automatic speaking assessment (ASA) model for the SANDI
(Speak & Improve 2025) corpus. CASA scores each exam part from two branches — an **acoustic**
branch (Whisper encoder + LoRA) and a **content** branch (ASR transcript read by an LLM) — in a
single forward pass.

Overall performance on the S&I test set (300 speakers). CASA-Crisper uses CrisperWhisper:

| Model | RMSE | PCC | %≤0.5 | %≤1.0 |
|---|---|---|---|---|
| NTNU (Lin et al.) | 0.360 | 0.827 | **85.7** | 99.0 |
| Perezoso (Cai et al.) | 0.364 | 0.826 | 83.0 | **99.7** |
| **CASA** | **0.358** | 0.829 | 84.7 | 98.7 |
| CASA-Crisper | 0.363 | **0.836** | 84.0 | **99.7** |

RMSE is speaker-level on the SANDI overall score (0–6 scale, 2=A2 … 5=C1). The reported CASA
checkpoint is selected by dev RMSE. Released configurations: **CASA** (main, Qwen3.5-2B),
**CASA-4B** (Qwen3.5-4B) and **CASA-Crisper** (CrisperWhisper).

> **Full run-level results are in [`RESULTS.md`](RESULTS.md)** — per-CEFR-band and per-part RMSE,
> the auxiliary-head scores, and a 5-configuration × 10-run variability study (50 runs).

All runs: seed 1011 unless stated, one 80 GB GPU (A100/H100), ~2 h for 2B.

## Architecture

![CASA architecture](figure/CASA_architecture.png)

*The editable source of this figure is [`figure/CASA-ICASSP.drawio`](figure/CASA-ICASSP.drawio) —
open it with [draw.io / diagrams.net](https://app.diagrams.net) (or the VS Code "Draw.io
Integration" extension) to edit or re-export it.*

Each test part (audio + task prompt) is scored independently; speaker overall = mean of part scores.

1. **Acoustic branch** — frozen `openai/whisper-medium` encoder with a LoRA adapter (r16),
   per-frame task/segment embeddings, frame pooling (×2), and a 2-layer RoPE Transformer
   aggregator with a `[CLS]` token. Produces (a) **4 acoustic soft tokens** projected into the
   LLM embedding space and (b) an **auxiliary CEFR regression head** (the "acoustic prior") trained
   with a tolerance-MSE loss (dead-zone ±1.0, weight 0.1).
2. **Text branch** — the exam task questions + the candidate's **ASR-transcribed** answers
   (never the human reference transcripts: those leak content unavailable at deployment),
   rendered into a rubric prompt (see `DEFAULT_RUBRIC` in `data_loaders/sandi_dataset.py`).
3. **Scorer** — frozen `Qwen/Qwen3.5-2B` with a LoRA adapter (r64/α128) reads
   `[soft tokens] + [rubric + acoustic estimate + task/answer text]`, single forward pass
   (no generation); a linear head on the final-token hidden state regresses the part score.
   Loss = MSE(main) + 0.1 · tolerance-MSE(aux).

## Setup

```bash
conda env create -f environment.yml && conda activate slaam   # exact export, or:
pip install -r requirements.txt                                # minimal pins (Python 3.12)
```

The `pip` route also needs the **ffmpeg** system binary on `PATH` — `generate_asr.py`'s
HuggingFace ASR pipeline uses it to decode the FLAC clips. It is not a pip package (so it is not
in `requirements.txt`), but it *is* bundled in `environment.yml`; if you took the pip route,
install it separately, e.g. `conda install -c conda-forge ffmpeg` (or your OS package manager).

Optional environment variables:

| var | meaning | default |
|---|---|---|
| `SANDI_CORPUS_ROOT` | installed official S&I corpus (for `build_csvs.py` only) | `data/sandi` |
| `SANDI_AUDIO_ROOT` | root of the SANDI `flac/` audio tree | `data/sandi/flac` |
| `HF_CACHE_DIR` | HuggingFace model cache dir | HF default (`HF_HOME`) |
| `USE_WANDB=1` | enable Weights & Biases logging in the train scripts | off |

## Data

The SANDI (Speak & Improve 2025) corpus is licensed and **not redistributed here** — obtain it
from the official Speak & Improve challenge release and install it per the corpus instructions.
The corpus's native layout (`example_data.tsv`, `reference-materials/sla-marks/*.tsv`,
`reference-materials/annotations/*.json`, `data/flac/{train,dev,eval}/`) is **not** what this
repo reads directly — `build_csvs.py` (step 0 below) converts it into the wide per-(speaker,
part) CSVs used everywhere else, mapping the official **eval** split to our **test** split.

### `csv/{train,dev,test}.csv` format (produced by `build_csvs.py`)

One row per speaker×part (parts 1, 3, 4, 5; train 1742 / dev 438 / test 300 speakers):

| column | meaning |
|---|---|
| `speaker_id` | e.g. `SI114J-00011` |
| `part` | 1, 3, 4 or 5 (part 2 has no scored audio) |
| `part_score` | gold score for this part, 0–6 scale (2=A2 … 5=C1, halves allowed) |
| `overall_score` | gold speaker score = mean of the 4 part scores |
| `audio_path_1..6` | FLAC clip paths, filled left-to-right in question order; empty when unused |
| `question_text_1..6` | exam question shown for each clip (P5 topic list reduced to the selected topic) |
| `transcript_1..6` | human reference transcript per clip (analysis only — **never** used in training) |

Parts 1/5 have up to 6/5 short clips; parts 3/4 always use only slot `_1`. Synthetic example row
(abridged): `SI999X-00001, 1, 3.5, 3.625, .../flac/dev/SI999X-00001-P10003.flac, …, "Where are
you from?", …, "i am from …", …`. Audio paths are re-rooted at load time onto
`$SANDI_AUDIO_ROOT` via their stable `/flac/` suffix, so the absolute prefix does not matter.

`build_csvs.py` also writes `{split}_overall.csv` (one row per speaker:
`speaker_id, part1_score, part3_score, part4_score, part5_score, overall_score`) and verifies
`overall = mean(parts)` against the corpus's stored values.

**Synthetic samples** of this format are in [`examples/csv/`](examples/) — use them to
check your conversion before training. All other CSVs (`asr/{split}.csv`, `master_*`) are
generated from these by the released scripts, so no samples are needed for them.

## Pipeline

```bash
# 0. Convert the official corpus into this repo's CSV format (CPU, seconds)
SANDI_CORPUS_ROOT=/path/to/sandi-corpus-2025 python build_csvs.py
                                    # -> csv/{train,dev,test}.csv (+ _overall.csv)

# 1. ASR-transcribe every clip with frozen whisper-medium (GPU; cached + resumable)
python generate_asr.py

# 2. Build the per-(speaker, part) master tables the trainer reads
python build_master.py --asr        # -> csv/master_{train,dev,test}_asr.csv

# 3. Train the scorer (~2 h on one 80 GB GPU)
bash scripts/train_casa.sh           # -> checkpoints/casa

# 4. Evaluate a trained checkpoint (dev + test metrics + prediction CSVs, no training)
bash scripts/evaluate.sh checkpoints/casa/checkpoint-<best>
```

`train.py` already evaluates dev and test at the end of training and writes
`{eval,test}_metrics.json` plus per-part/overall prediction CSVs to the output dir;
`scripts/evaluate.sh` reruns that inference path standalone from any checkpoint.

### Variants

- **Qwen 4B**: `bash scripts/train_casa_4b.sh` — identical recipe, only `--qwen_name` differs.
- **CrisperWhisper** (verbatim ASR as both transcript source and acoustic encoder):

```bash
python generate_asr.py --model nyrahealth/CrisperWhisper --out_subdir asr_crisper
python build_master.py --asr --asr_dir asr_crisper --suffix _crisper
bash scripts/train_casa_crisper.sh
```

- **Acoustic-encoder ablation** (SSL encoders): swaps only the acoustic encoder while keeping the
  whisper-medium ASR transcript, so the acoustic representation is the sole difference from CASA.
  No extra ASR step is needed — these reuse `csv/master_*_asr.csv`.

```bash
bash scripts/train_casa_wavlm.sh     # microsoft/wavlm-large
bash scripts/train_casa_xlsr.sh      # facebook/wav2vec2-xls-r-300m
bash scripts/train_casa_xlsr_en.sh   # XLSR-53 fine-tuned for English ASR
```

## Reproducing the paper's experiments

**Run-to-run variability (Table: 3 configurations x 10 runs).** Each configuration is trained 10
times: 5 runs at seed 1011 (a base run plus 4 repetitions, which differ only through nondeterministic
GPU kernels) and 5 runs at seeds 2022/3033/4044/5055/6066. Only `--seed` changes between runs:

```bash
for SEED in 1011 1011 1011 1011 1011 2022 3033 4044 5055 6066; do
  bash scripts/train_casa.sh        # edit --seed / --output_dir per run, or pass them through
done
# the two comparison configurations differ from CASA by a single flag:
bash scripts/train_casa_lr4e4.sh    # --whisper_lr 4e-4  (worse and less stable; not recommended)
bash scripts/train_casa_aux0.sh     # --aux_weight 0.0   (auxiliary loss disabled)
```

**Few-shot content validation.** Uses CASA's LLM backbone (base Qwen3.5-2B, LoRA off) to judge
whether an answer addresses its question. `build_content_attacks.py` pairs every test answer with an
unrelated question (about nuclear reactors), leaving the audio and answers untouched:

```bash
python build_content_attacks.py                                          # -> csv/attacks/*.csv
python content_validation.py --master_csv csv/master_test_asr.csv        # clean control
python content_validation.py --master_csv csv/attacks/master_test_asr_unrelated_question.csv
```

Each run prints the good / average / bad distribution (`bad` = off-topic). The judge prompt itself
(system prompt, label definitions and the four few-shot examples) is recorded verbatim in
[`prompts/content_validation.json`](prompts/content_validation.json).

## Repository layout

```
train.py                     # training + evaluation entry point (--eval_only for inference)
build_csvs.py                # official S&I corpus -> csv/{train,dev,test}.csv (step 0)
generate_asr.py              # frozen-Whisper ASR transcription of the SANDI clips
build_master.py              # raw SANDI CSVs -> master_{split}[_asr|_crisper].csv
build_content_attacks.py     # builds the unrelated-question / cross-part attacked test masters
content_validation.py        # few-shot LLM judge: does the answer address the question?
modeling/qwen_scorer.py      # QwenAcousticScorer: LLM+LoRA, soft tokens, aux head, tolerance-MSE
modeling/whisper_acoustic.py # WhisperAcousticEncoder: Whisper+LoRA, RoPE aggregator, [CLS] pool
modeling/ssl_acoustic.py     # SSLAcousticEncoder: raw-waveform wav2vec2/WavLM+LoRA, same (B,D) contract
data_loaders/sandi_dataset.py# dataset + the full scoring prompt (DEFAULT_RUBRIC)
data_loaders/collators.py    # batch collation
utils/metrics.py             # speaker-level RMSE / macro-RMSE / PCC / CEFR-band metrics
scripts/                     # train_casa*.sh (main, 4b, crisper, wavlm, xlsr, xlsr_en,
                             #   lr4e4, aux0), evaluate.sh
prompts/content_validation.json # few-shot content-validation judge prompt (system + demonstrations)
examples/                    # synthetic samples of every CSV format in the pipeline
figure/                      # architecture figure (PNG) + its editable draw.io source
```

## Notes for reproduction

- Training uses **ASR transcripts only**; `build_master.py` without `--asr` builds gold-transcript
  masters for analysis, which must never be used for training (content leakage).
- The dev-selected checkpoint (`load_best_model_at_end`, selection metric includes a macro-RMSE
  weight) is the one evaluated on test; test is touched once per run.
- Feature extraction is cached under `feature_cache/` (keyed by model names + prompt version);
  the first epoch is slower while the cache fills.
- Results were obtained with the pinned versions in `requirements.txt` on CUDA 12.8.

## License

Apache License 2.0 — see `LICENSE`. This applies to the code in this repository only.
The Qwen and Whisper model weights are downloaded from their upstream sources under their own
licenses (both Apache 2.0), and the SANDI corpus is separately licensed and not redistributed here.
