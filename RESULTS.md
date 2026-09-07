# CASA — detailed results

All numbers are on the held-out S&I **test** set (300 speakers, 1200 part-responses).
`dev` is the dev-set overall RMSE used for checkpoint selection; `aux` is the acoustic-only
auxiliary head (no text channel). PCC / SRC are the speaker-level Pearson / Spearman correlations
between predicted and gold overall score. %≤0.5 / %≤1.0 are the fractions of speakers whose predicted
overall score falls within 0.5 / 1.0 of the gold score.

This file reports the configurations discussed in the paper.

> **Note on the `aux` column:** in the `aux-0` configuration the auxiliary head receives no training
> signal (its loss weight is 0), so it stays at its random initialisation and its `aux` values are
> not meaningful — they are reported only for completeness.

## Released model variants

| Model | Acoustic encoder | LLM scorer | dev (overall / macro) ↓ | test (overall / macro) ↓ | PCC ↑ | SRC ↑ | %≤0.5 ↑ | %≤1.0 ↑ |
|---|---|---|---|---|---|---|---|---|
| **CASA** | whisper-medium | Qwen3.5-2B | **0.3607** / 0.4403 | **0.3579** / 0.4371 | 0.829 | 0.828 | **84.7** | 98.7 |
| CASA-4B | whisper-medium | Qwen3.5-4B | 0.3645 / 0.4472 | 0.3641 / 0.4447 | 0.823 | 0.821 | 83.3 | **99.7** |
| CASA-Crisper | CrisperWhisper (verbatim) | Qwen3.5-2B | 0.3654 / 0.4430 | 0.3628 / 0.4399 | **0.836** | **0.839** | 84.0 | **99.7** |

## Acoustic-encoder ablation

Only the acoustic encoder is swapped; the ASR transcript stays whisper-medium and the scorer stays
Qwen3.5-2B, so the acoustic representation is the sole difference from CASA. All three are
self-supervised (SSL) encoders; XLSR-53-en is additionally fine-tuned for English ASR.

| Acoustic encoder | dev (overall / macro) ↓ | test (overall / macro) ↓ | PCC ↑ | SRC ↑ | %≤0.5 ↑ | %≤1.0 ↑ | aux-only ↓ |
|---|---|---|---|---|---|---|---|
| **whisper-medium (CASA)** | **0.3607** / 0.4403 | **0.3579** / 0.4371 | **0.829** | **0.828** | **84.7** | 98.7 | **0.392** |
| WavLM-large | 0.3851 / 0.4484 | 0.3905 / 0.4561 | 0.798 | 0.799 | 79.0 | **99.3** | 0.515 |
| XLS-R-300M (multilingual) | 0.3838 / 0.4711 | 0.4046 / 0.5018 | 0.776 | 0.777 | 78.7 | 98.7 | 0.635 |
| XLSR-53 English-ASR-tuned | 0.3861 / 0.4882 | 0.4013 / 0.5199 | 0.808 | 0.811 | 81.3 | 99.0 | 0.531 |

## Run-to-run variability (3 configurations × 10 runs)

For investigating the variability, we train each configuration 9 additional runs beyond the base configuration (10 total): 5 runs at seed 1011 (a base run plus 4 repetitions,
capturing nondeterministic GPU variation) and 5 runs at distinct seeds (2022/3033/4044/5055/6066). 

| Config | n | Mean ↓ | Median ↓ | Min–Max  | 95% CI  | sd |
|---|---|---|---|---|---|---|
| **CASA** | 10 | **0.363** | **0.362** | 0.357–0.377 | 0.359–0.367 | 0.006 |
| 4e-4 (doubled encoder LR) | 10 | 0.378 | 0.376 | 0.350–0.402 | 0.364–0.392 | 0.019 |
| aux-0 (auxiliary weight 0) | 10 | 0.367 | 0.366 | 0.362–0.376 | 0.364–0.370 | 0.005 |

The reported result follows our standard protocol: a single run per configuration, with the model configuration and checkpoint selected on the development set, never the test set. The headline CASA result (test RMSE 0.358) is that dev-selected run — not the best variability run overall (otherwise we would report 0.350 from the unstable 4e-4 configuration instead). The variability study above is a separate robustness check and does not feed the reported number.

> **Note — ASR batch size.** The ASR transcripts also depend on batch size: regenerating them with `generate_asr.py --batch_size 32` instead of the original 16 alters a small fraction of clips (~4%), which slightly shifts the content-branch input and gives a test RMSE of 0.357 (vs 0.358).

## Per-run detail

### CASA

| Run | Seed | dev ↓ | test ↓ | aux ↓ | P1 ↓ | P3 ↓ | P4 ↓ | P5 ↓ | A2 ↓ | B1 ↓ | B2 ↓ | C1 ↓ | macro ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CASA | 1011 | **0.361** | 0.358 | 0.392 | 0.476 | 0.454 | 0.490 | 0.444 | 0.553 | 0.351 | 0.290 | 0.554 | 0.437 |
| CASA-r2 | 1011 | 0.365 | 0.363 | 0.405 | 0.472 | 0.458 | 0.490 | 0.469 | 0.560 | 0.341 | 0.313 | 0.557 | 0.443 |
| CASA-r3 | 1011 | 0.367 | 0.365 | 0.404 | 0.473 | 0.468 | 0.492 | 0.446 | 0.547 | 0.353 | 0.302 | 0.581 | 0.446 |
| CASA-r4 | 1011 | 0.379 | 0.363 | 0.398 | 0.469 | 0.459 | 0.494 | 0.459 | **0.526** | 0.355 | 0.301 | 0.580 | 0.441 |
| CASA-r5 | 1011 | 0.362 | **0.357** | 0.399 | 0.465 | **0.453** | **0.476** | 0.450 | 0.566 | 0.343 | 0.290 | 0.570 | 0.442 |
| CASA-s2 | 2022 | 0.375 | 0.359 | 0.423 | 0.471 | 0.461 | 0.503 | **0.437** | 0.565 | 0.359 | 0.287 | 0.532 | 0.436 |
| CASA-s3 | 3033 | 0.362 | 0.361 | **0.389** | 0.472 | 0.463 | 0.491 | 0.443 | 0.535 | **0.321** | 0.317 | 0.633 | 0.451 |
| CASA-s4 | 4044 | 0.371 | 0.362 | 0.403 | 0.467 | 0.460 | 0.489 | 0.447 | 0.558 | 0.366 | 0.289 | **0.528** | **0.435** |
| CASA-s5 | 5055 | 0.367 | 0.361 | 0.414 | **0.459** | 0.472 | 0.482 | 0.453 | 0.578 | 0.358 | **0.281** | 0.565 | 0.445 |
| CASA-s6 | 6066 | 0.368 | 0.377 | 0.406 | 0.485 | 0.464 | 0.517 | 0.451 | 0.560 | 0.327 | 0.325 | 0.710 | 0.481 |

### 4e-4 (doubled encoder LR)

| Run | Seed | dev ↓ | test ↓ | aux ↓ | P1 ↓ | P3 ↓ | P4 ↓ | P5 ↓ | A2 ↓ | B1 ↓ | B2 ↓ | C1 ↓ | macro ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4e-4 | 1011 | 0.368 | **0.350** | **0.389** | **0.460** | **0.450** | 0.488 | **0.441** | 0.554 | 0.344 | 0.281 | **0.535** | **0.428** |
| 4e-4-r2 | 1011 | 0.370 | 0.363 | **0.389** | 0.475 | 0.459 | 0.480 | 0.454 | 0.561 | 0.344 | 0.303 | 0.589 | 0.449 |
| 4e-4-r3 | 1011 | **0.364** | 0.367 | 0.431 | 0.475 | 0.484 | **0.479** | 0.450 | 0.554 | 0.362 | 0.299 | 0.564 | 0.445 |
| 4e-4-r4 | 1011 | 0.368 | 0.361 | 0.413 | 0.476 | 0.473 | 0.486 | 0.456 | **0.550** | **0.340** | 0.303 | 0.596 | 0.447 |
| 4e-4-r5 | 1011 | 0.373 | 0.385 | 0.496 | 0.498 | 0.499 | 0.520 | 0.479 | 0.554 | 0.362 | 0.323 | 0.669 | 0.477 |
| 4e-4-s2 | 2022 | 0.384 | 0.400 | 0.485 | 0.511 | 0.504 | 0.543 | 0.495 | 0.617 | 0.379 | 0.325 | 0.680 | 0.500 |
| 4e-4-s3 | 3033 | 0.386 | 0.402 | 0.638 | 0.526 | 0.517 | 0.544 | 0.495 | 0.616 | 0.388 | 0.321 | 0.674 | 0.500 |
| 4e-4-s4 | 4044 | 0.391 | 0.398 | 0.633 | 0.522 | 0.525 | 0.533 | 0.500 | 0.605 | 0.387 | 0.316 | 0.663 | 0.493 |
| 4e-4-s5 | 5055 | 0.372 | 0.363 | 0.431 | 0.472 | 0.456 | 0.492 | 0.454 | 0.603 | 0.363 | **0.274** | 0.557 | 0.449 |
| 4e-4-s6 | 6066 | 0.379 | 0.394 | 0.535 | 0.517 | 0.493 | 0.529 | 0.487 | 0.609 | 0.361 | 0.332 | 0.683 | 0.496 |

### aux-0 (auxiliary weight 0)

| Run | Seed | dev ↓ | test ↓ | aux ↓ | P1 ↓ | P3 ↓ | P4 ↓ | P5 ↓ | A2 ↓ | B1 ↓ | B2 ↓ | C1 ↓ | macro ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aux-0 | 1011 | 0.367 | **0.362** | 3.121 | **0.457** | 0.461 | 0.490 | 0.457 | 0.553 | 0.347 | 0.300 | 0.582 | 0.445 |
| aux-0-r2 | 1011 | 0.370 | 0.366 | 4.231 | 0.473 | 0.457 | 0.493 | 0.452 | 0.561 | 0.350 | 0.303 | 0.587 | 0.450 |
| aux-0-r3 | 1011 | **0.363** | **0.362** | 3.485 | 0.475 | 0.468 | **0.481** | 0.450 | 0.532 | 0.356 | 0.296 | 0.571 | **0.439** |
| aux-0-r4 | 1011 | 0.365 | 0.369 | 3.561 | 0.477 | 0.466 | 0.499 | 0.455 | 0.563 | 0.351 | 0.302 | 0.616 | 0.458 |
| aux-0-r5 | 1011 | 0.369 | 0.373 | 4.343 | 0.480 | 0.464 | 0.488 | 0.463 | **0.521** | **0.334** | 0.340 | 0.624 | 0.455 |
| aux-0-s2 | 2022 | 0.374 | 0.368 | 4.032 | 0.478 | 0.462 | 0.491 | 0.451 | 0.601 | 0.362 | 0.291 | 0.563 | 0.454 |
| aux-0-s3 | 3033 | 0.369 | 0.365 | 4.100 | 0.476 | **0.451** | 0.498 | 0.447 | 0.553 | 0.345 | 0.314 | 0.568 | 0.445 |
| aux-0-s4 | 4044 | 0.371 | 0.367 | 3.472 | 0.482 | 0.466 | 0.497 | 0.459 | 0.567 | 0.356 | 0.296 | 0.597 | 0.454 |
| aux-0-s5 | 5055 | 0.372 | 0.376 | 3.730 | 0.481 | 0.469 | 0.515 | 0.471 | 0.593 | 0.371 | 0.298 | 0.580 | 0.461 |
| aux-0-s6 | 6066 | 0.372 | 0.363 | 4.166 | 0.475 | 0.453 | 0.505 | **0.443** | 0.603 | 0.353 | **0.288** | **0.556** | 0.450 |
