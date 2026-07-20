#!/usr/bin/env python3
"""Generate ASR transcripts for SANDI audio with FROZEN openai/whisper-medium.

WHY: training must NOT use SANDI's human reference transcripts (they leak gold content the
deployed pipeline never has). This transcribes every clip with the same frozen Whisper-medium
used by the acoustic encoder, and writes `csv/asr/{split}.csv` with
`asr_transcript_1 ... asr_transcript_6` aligned to `audio_path_1 ... audio_path_6`.
`build_master.py --asr` then builds the master tables from these instead of the manual text.

Frozen base Whisper (LoRA OFF) — the scoring LoRA is for acoustics, not transcription.
Checkpointed: results cached per clip in csv/asr/.asr_cache.json; safe to cancel/resubmit.

Run on a GPU node from the repo root:  python generate_asr.py
CrisperWhisper variant:  python generate_asr.py --model nyrahealth/CrisperWhisper --out_subdir asr_crisper
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "csv"
OUT = SRC / "asr"
CACHE = OUT / ".asr_cache.json"

# Canonical SANDI audio root (override with the SANDI_AUDIO_ROOT environment variable).
# The raw CSVs contain inconsistent absolute paths; rebuild from the /flac/ marker.
AUDIO_ROOT = os.environ.get("SANDI_AUDIO_ROOT", "data/sandi/flac")
_FLAC = "/flac/"
# HuggingFace model cache. None -> the default HF cache (respects HF_HOME);
# override with the HF_CACHE_DIR environment variable.
HF_CACHE = os.environ.get("HF_CACHE_DIR") or None


def norm_path(p: str) -> str:
    p = str(p).strip()
    return f"{AUDIO_ROOT}/{p.split(_FLAC, 1)[1]}" if _FLAC in p else p


def load_cache() -> dict:
    if CACHE.exists():
        raw = json.loads(CACHE.read_text())
        return {
            path: transcript
            for path, transcript in raw.items()
            if isinstance(transcript, str) and transcript.strip()
        }
    return {}


def save_cache(cache: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))


def build_pipeline(batch_size: int, model_id: str = None):
    model_id = model_id or "openai/whisper-medium"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=HF_CACHE)
    model.to("cuda")
    # Clear any stale forced_decoder_ids so the explicit language/task kwargs are authoritative —
    # otherwise a multilingual model (e.g. CrisperWhisper, 100 langs) can auto-detect/drift to a
    # non-English language on heavily-accented L2 speech instead of transcribing English.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_decoder_ids = None
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE)
    return pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, torch_dtype=torch.float16,
        device="cuda", chunk_length_s=30, stride_length_s=5, batch_size=batch_size,
        generate_kwargs={"language": "english", "task": "transcribe"})


def _as_list(results):
    if isinstance(results, dict):
        return [results]
    return list(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--model", default=None, help="HF id or path (default openai/whisper-medium)")
    ap.add_argument("--out_subdir", default="asr", help="csv/<subdir>/ for output + cache")
    args = ap.parse_args()

    global OUT, CACHE
    OUT = SRC / args.out_subdir
    CACHE = OUT / ".asr_cache.json"
    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    # gather every unique clip across all requested splits
    apc = [f"audio_path_{i}" for i in range(1, 7)]
    clips = set()
    frames = {}
    for split in args.splits:
        df = pd.read_csv(SRC / f"{split}.csv")
        frames[split] = df
        for c in apc:
            if c in df.columns:
                clips.update(norm_path(v) for v in df[c].dropna() if str(v).strip())
    todo = sorted(p for p in clips if p not in cache)
    print(f"{len(clips)} unique clips, {len(todo)} to transcribe ({len(cache)} cached)")

    if todo:
        asr = build_pipeline(args.batch_size, args.model)
        for start in tqdm(range(0, len(todo), args.batch_size), desc="ASR", unit="batch"):
            batch = todo[start: start + args.batch_size]
            existing = [p for p in batch if os.path.exists(p)]
            for path in batch:
                if path not in existing:
                    cache[path] = ""
            if existing:
                try:
                    results = _as_list(asr(existing))
                    if len(results) != len(existing):
                        raise RuntimeError(f"ASR returned {len(results)} results for {len(existing)} clips")
                    for path, res in zip(existing, results):
                        cache[path] = res["text"].strip()
                except Exception as e:       # noqa: BLE001 — retry the batch safely
                    print(f"  WARN batch {start}-{start + len(batch)}: {e}", file=sys.stderr)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for path in existing:
                        try:
                            cache[path] = asr(path)["text"].strip()
                        except Exception as single_e:  # noqa: BLE001 — leave uncached for the next run
                            print(f"  WARN {path}: {single_e}", file=sys.stderr)
                            cache.pop(path, None)
            if (start // args.batch_size + 1) % 10 == 0:
                save_cache(cache)
        save_cache(cache)

    # write per-split ASR CSVs aligned to audio_path_i
    for split, df in frames.items():
        out_rows = []
        for _, r in df.iterrows():
            row = {"speaker_id": r["speaker_id"], "part": int(r["part"])}
            for i in range(1, 7):
                v = r.get(f"audio_path_{i}")
                row[f"asr_transcript_{i}"] = cache.get(norm_path(v), "") if isinstance(v, str) and v.strip() else ""
            out_rows.append(row)
        outp = OUT / f"{split}.csv"
        pd.DataFrame(out_rows).to_csv(outp, index=False)
        print(f"wrote {outp} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
