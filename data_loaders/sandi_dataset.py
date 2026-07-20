"""SANDI (Speak & Improve) per-(speaker, part) dataset for the Whisper-LoRA + Qwen scorer.

One item = one speaking *part* (P1/P3/P4/P5) of one speaker. It produces BOTH branches'
inputs from `csv/master_{split}.csv`:

  * Acoustic branch -> Whisper mel chunks `(C, 80, 3000)` with a per-chunk `segment_id`
    (which clip/answer the chunk came from) and a `task_id` (the part). Multi-clip parts
    (P1 ~6 answers, P5 ~5) are concatenated; single-clip parts (P3/P4) give one clip.
  * Text branch -> a tokenized LLM prompt = rubric + interleaved Q/A (the boundary +
    question marker, already in `llm_input`) + an `<ACOUSTIC_EVIDENCE>` block of cheap
    handcrafted features + a trailing `<SCORE>:` cue. The scorer reads the hidden state at
    the final (non-pad) token, so no special-token embedding resize is needed.

Target = `part_score` (regression). `speaker_id` / `part` are returned so eval can group
predictions per speaker and average parts -> overall RMSE (overall = mean of part scores).
"""

import hashlib
import json
import math
import os
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, WhisperFeatureExtractor

# HuggingFace model cache. None -> the default HF cache (respects HF_HOME);
# override with the HF_CACHE_DIR environment variable.
HF_CACHE = os.environ.get("HF_CACHE_DIR") or None

# Canonical SANDI audio root (override with the SANDI_AUDIO_ROOT environment variable).
# The raw SANDI CSVs contain inconsistent absolute audio paths, so every path is rebuilt
# from the stable `/flac/` suffix onto this root.
SANDI_AUDIO_ROOT = os.environ.get("SANDI_AUDIO_ROOT", "data/sandi/flac")
_FLAC_MARKER = "/flac/"

# Parts kept (P2 has no scored audio in SANDI); contiguous task ids for the task embedding.
PART_TO_TASK_ID = {1: 0, 3: 1, 4: 2, 5: 3}

# Bump when the prompt text / structure changes -> invalidates the feature cache (which stores
# tokenized prompts). asr-content-v3 = explicit language for English training / Finnish transfer.
PROMPT_VERSION = "asr-content-v3"

DEFAULT_RUBRIC = (
    "<RUBRIC>\n"
    "target_language: {target_language}\n"
    "You are scoring ONE part of an L2 speaking test on the CEFR scale, giving a single "
    "holistic CEFR score for the whole part.\n"
    "A Whisper-medium speech encoder with a LoRA adapter has already assessed the ACOUSTIC and "
    "DELIVERY of this response — fluency, pronunciation, intonation, hesitation and speech rate. "
    "Its judgement (supplied above as the acoustic soft tokens and the `acoustic_cefr_estimate` "
    "value) is reliable ONLY for that acoustic/delivery aspect — it can hear how the speech sounds "
    "but it cannot read what was actually said, so its CEFR estimate is just an acoustic-based "
    "guess.\n"
    "Your job is to determine the TRUE overall CEFR score. Take the acoustic estimate as the "
    "delivery signal, then compensate using the CONTENT of the response, which only you can judge "
    "in the target language from the transcript: vocabulary range, grammatical range and "
    "accuracy, coherence, and how fully each question/task is addressed. Raise the score when "
    "the content is richer or more accurate than the delivery alone implies, lower it when the "
    "content is weak, and output one holistic CEFR score for the part.\n"
    "The transcript below is produced by automatic speech recognition (ASR) and may contain "
    "recognition errors; judge the content through them."
)

LEGACY_HUMAN_RUBRIC = (
    "<RUBRIC>\n"
    "Assess the L2 English speaking proficiency shown in the answers below using "
    "CEFR-style criteria (fluency, pronunciation, grammatical range and accuracy, and how "
    "well each question is addressed). Give one holistic score for this part."
)


def _load_wave(path: str, target_sr: int = 16000,
               start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    """Mono float32 waveform at `target_sr`, via soundfile (librosa/torchaudio are broken in
    this env — numba/NumPy and torchcodec ABI; see root AGENTS.md).

    `start_s`/`end_s` read only a sub-slice of the file — unused by SANDI monologue (whole
    files), the hook a future dialogue loader uses to gather one speaker's diarized turns
    from the shared recording without decoding the whole thing."""
    if start_s or end_s is not None:
        nsr = sf.info(path).samplerate
        stop = int(end_s * nsr) if end_s is not None else None
        wave, sr = sf.read(path, start=int(start_s * nsr), stop=stop, dtype="float32")
    else:
        wave, sr = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        wave = resample_poly(wave, target_sr // g, sr // g).astype(np.float32)
    return wave


def _silence_ratio(wave: np.ndarray, sr: int, frame_ms: int = 25, thresh_db: float = -35.0) -> float:
    """Fraction of short frames whose RMS energy is below `thresh_db` relative to the peak.

    Cheap pure-numpy proxy for pausing/hesitation; no VAD model needed."""
    if wave.size == 0:
        return 0.0
    hop = max(1, int(sr * frame_ms / 1000))
    n = wave.size // hop
    if n == 0:
        return 0.0
    rms = np.sqrt((wave[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    if rms.max() == 0:
        return 0.0
    db = 20.0 * np.log10(rms / rms.max() + 1e-9)
    return float((db < thresh_db).mean())


class SandiPartDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        whisper_name: str = "openai/whisper-medium",
        qwen_name: str = "Qwen/Qwen3.5-2B",
        max_chunks: int = 4,          # 4 * 30s = 120s; covers SANDI/DTA/DigiTala, ~p99 of YKI
        chunk_duration: int = 30,
        max_text_len: int = 1024,
        rubric: str = DEFAULT_RUBRIC,
        target_language: str = "English",
        prompt_version: str = PROMPT_VERSION,
        score_cue: str = "\n\n<SCORE>:",
        audio_root: str = SANDI_AUDIO_ROOT,
        cache_dir: str | None = None,
    ):
        self.data = pd.read_csv(csv_path)
        self.audio_root = audio_root.rstrip("/")
        self.sr = 16000
        self.chunk_duration = chunk_duration
        self.chunk_samples = chunk_duration * self.sr
        self.max_chunks = max_chunks
        self.max_text_len = max_text_len
        self.target_language = target_language.strip()
        if not self.target_language:
            raise ValueError("target_language must not be empty")
        self.prompt_version = prompt_version.strip()
        if not self.prompt_version:
            raise ValueError("prompt_version must not be empty")
        if "prompt_version" in self.data.columns:
            versions = set(self.data["prompt_version"].dropna().astype(str))
            if versions != {self.prompt_version}:
                raise ValueError(
                    f"{csv_path} has scorer prompt versions {sorted(versions)}; "
                    f"expected only {self.prompt_version}"
                )
        self.rubric = rubric.format(target_language=self.target_language)
        self.score_cue = score_cue

        # whisper -> log-mel input_features (C,80,3000); wav2vec2/wavlm -> raw input_values
        # (C, chunk_samples). The acoustic encoder family is auto-detected from the model config.
        try:
            self._acoustic_type = AutoConfig.from_pretrained(whisper_name, cache_dir=HF_CACHE).model_type
        except Exception:
            self._acoustic_type = "whisper"
        self._acoustic_is_mel = (self._acoustic_type == "whisper")
        if self._acoustic_is_mel:
            self.feat = WhisperFeatureExtractor.from_pretrained(whisper_name, cache_dir=HF_CACHE)
        else:
            self.feat = AutoFeatureExtractor.from_pretrained(whisper_name, cache_dir=HF_CACHE)
        self.tok = AutoTokenizer.from_pretrained(qwen_name, cache_dir=HF_CACHE)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Feature cache: store the full __getitem__ output as .pt files so every epoch
        # after the first is a cheap torch.load (avoids ~3.9s/item FLAC+mel+tokenize cost).
        # Config hash in the dir name means any change to audio/text processing auto-invalidates.
        self.cache_root = None
        if cache_dir:
            cfg = "|".join([
                os.path.abspath(csv_path), whisper_name, qwen_name,
                str(max_chunks), str(chunk_duration), str(max_text_len), self.prompt_version,
                "no-content-analysis",
                self.target_language, self.rubric, self.score_cue,
            ])
            prefix = hashlib.md5(cfg.encode()).hexdigest()[:12]
            self.cache_root = os.path.join(cache_dir, prefix)
            os.makedirs(self.cache_root, exist_ok=True)
            print(f"Feature cache: {self.cache_root}  "
                  f"({sum(1 for f in os.scandir(self.cache_root) if f.name.endswith('.pt'))} "
                  f"/ {len(self.data)} items warm)")

    def __len__(self) -> int:
        return len(self.data)

    def _norm_path(self, p: str) -> str:
        """Rebuild an audio path onto the canonical root from its stable `/flac/` suffix."""
        if self.audio_root and _FLAC_MARKER in p:
            return f"{self.audio_root}/{p.split(_FLAC_MARKER, 1)[1]}"
        return p

    # ------------------------------------------------------------------ audio
    def _load_chunks(self, segments: list):
        """Concatenate a part's clips, THEN cut into 30s mel chunks + per-chunk segment ids.

        Concatenating before chunking (vs per-clip) avoids padding every short P1/P5 answer up
        to 30s and the resulting chunk-count blow-up that `max_chunks` would truncate. The
        per-chunk `segment_id` = the clip the chunk *starts* in (approximate; exact answer
        boundaries are carried losslessly by the interleaved Q/A text instead).
        Returns (input_features (C,80,3000), segment_ids (C,), feats_dict).

        Each element of `segments` is a path str (whole file, SANDI) OR a
        (path, start_s, end_s) tuple (a sub-slice — the dialogue hook)."""
        waves, clip_lens = [], []
        for seg in segments:
            if isinstance(seg, (tuple, list)):
                path, start_s, end_s = seg
                wave = _load_wave(self._norm_path(path), self.sr, start_s, end_s)
            else:
                wave = _load_wave(self._norm_path(seg), self.sr)
            if wave.size:
                waves.append(wave)
                clip_lens.append(wave.size)

        full = np.concatenate(waves) if waves else np.zeros(self.chunk_samples, dtype=np.float32)
        bounds = np.cumsum(clip_lens) if clip_lens else np.array([full.size])

        n = max(1, int(math.ceil(full.size / self.chunk_samples)))
        n = min(n, self.max_chunks)
        chunks = []
        for j in range(n):
            chunk = full[j * self.chunk_samples : (j + 1) * self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunks.append(chunk)

        if self._acoustic_is_mel:
            input_features = self.feat(
                chunks, sampling_rate=self.sr, return_tensors="pt"
            ).input_features                                    # (C, 80, 3000)
        else:
            input_features = self.feat(
                chunks, sampling_rate=self.sr, return_tensors="pt"
            ).input_values                                      # (C, chunk_samples) raw waveform

        # cumulative clip end-times (seconds) -> encoder maps each frame to its answer index
        clip_bounds = torch.tensor(bounds / self.sr, dtype=torch.float32)
        feats = {
            "duration": round(full.size / self.sr, 1),
            "silence_ratio": round(_silence_ratio(full, self.sr), 3),
        }
        return input_features, clip_bounds, feats

    # ------------------------------------------------------------------ text
    def _build_prompt(self, llm_input: str, feats: dict, transcript: str) -> str:
        n_words = len(str(transcript).split())
        speech_rate = round(n_words / feats["duration"], 2) if feats["duration"] > 0 else 0.0
        evidence = (
            "\n\n<ACOUSTIC_EVIDENCE>\n"
            f"duration_sec: {feats['duration']}\n"
            f"silence_ratio: {feats['silence_ratio']}\n"
            f"speech_rate_wps: {speech_rate}"
        )
        return (
            f"{self.rubric}\n\n{llm_input}"
            f"{evidence}{self.score_cue}"
        )

    def __getitem__(self, idx: int) -> dict:
        # Cache hit: skip all I/O and feature extraction
        if self.cache_root is not None:
            cache_file = os.path.join(self.cache_root, f"{idx:06d}.pt")
            if os.path.exists(cache_file):
                try:
                    item = torch.load(cache_file, weights_only=True)
                    return item
                except (EOFError, FileNotFoundError, RuntimeError):
                    # Cache files can disappear or be half-written if a cache cleanup/job
                    # interruption races a DataLoader worker. Treat as a miss.
                    pass

        row = self.data.iloc[idx]
        paths = [p for p in str(row["audio_paths"]).split("|") if p]

        input_features, clip_bounds, feats = self._load_chunks(paths)
        llm_input = row["llm_input"]
        prompt = self._build_prompt(
            llm_input,
            feats,
            row.get("transcript_concat", ""),
        )
        enc = self.tok(
            prompt, truncation=True, max_length=self.max_text_len, return_tensors="pt"
        )

        item = {
            "input_features": input_features,                  # (C, 80, 3000)
            "clip_bounds": clip_bounds,                        # (k,) cumulative clip end-times (s)
            "task_id": (
                int(row["model_task_id"])
                if "model_task_id" in row and not pd.isna(row["model_task_id"])
                else PART_TO_TASK_ID.get(int(row["part"]), 0)
            ),
            "input_ids": enc["input_ids"].squeeze(0),          # (L,)
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": float(row["part_score"]),
            "overall_score": float(row["overall_score"]),
            "speaker_id": str(row["speaker_id"]),
            "part": int(row["part"]),
        }

        # Cache miss: save for all future epochs
        if self.cache_root is not None:
            cache_file = os.path.join(self.cache_root, f"{idx:06d}.pt")
            tmp_file = f"{cache_file}.{os.getpid()}.tmp"
            try:
                os.makedirs(self.cache_root, exist_ok=True)
                torch.save(item, tmp_file)
                os.replace(tmp_file, cache_file)
            except (OSError, RuntimeError):
                # Cache is an optimization, not training state. If the directory was removed
                # while this job is running, keep training and just skip this write.
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except OSError:
                    pass

        return item
