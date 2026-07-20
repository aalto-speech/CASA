"""Build per-(speaker, part) SANDI master tables for the Whisper-LoRA + Qwen scorer.

Reads the raw wide SANDI CSVs (`csv/{train,dev,test}.csv`, one row per speaker×part
with up to 6 audio/question/transcript clips) and emits `csv/master_{split}.csv` with:
  speaker_id, part, part_score, overall_score, split, n_clips,
  audio_paths   (|-joined, normalized onto the canonical /flac/ root),
  llm_input     (rubric-free body: "<TASK part=Pk>" + interleaved "Q{i}:/A{i}:" blocks
                 = the answer-boundary + question marker for the LLM text path),
  transcript_concat (plain space-joined answers).

TRANSCRIPT SOURCE:
  default = SANDI human reference transcripts (`transcript_i`) — these LEAK gold content the
            deployed pipeline never sees, so they are for analysis only, NOT for training.
  --asr   = ASR transcripts from `csv/asr/{split}.csv` (`asr_transcript_i`), produced by
            generate_asr.py with frozen Whisper-medium. THIS is the training source.
            Emits `csv/master_{split}_asr.csv`.

overall_score is SANDI's gold (= mean of the 4 part scores). Run from the repo root:
  python build_master.py --asr
"""

import argparse
import os

import pandas as pd

SRC = "csv"
# Canonical SANDI audio root (override with the SANDI_AUDIO_ROOT environment variable);
# the raw CSVs contain inconsistent absolute paths.
AUDIO_ROOT = os.environ.get("SANDI_AUDIO_ROOT", "data/sandi/flac")
_FLAC = "/flac/"


def _norm(p: str) -> str:
    return f"{AUDIO_ROOT}/{p.split(_FLAC, 1)[1]}" if _FLAC in p else p


def build(split: str, asr: bool, asr_dir: str = "asr", suffix: str = "_asr") -> pd.DataFrame:
    df = pd.read_csv(f"{SRC}/{split}.csv")
    ap = [f"audio_path_{i}" for i in range(1, 7)]
    qc = [f"question_text_{i}" for i in range(1, 7)]
    tc = [f"transcript_{i}" for i in range(1, 7)]

    if asr:
        # merge ASR transcripts (asr_transcript_i) keyed by (speaker_id, part)
        adf = pd.read_csv(f"{SRC}/{asr_dir}/{split}.csv").set_index(["speaker_id", "part"])

    rows = []
    for _, r in df.iterrows():
        if asr:
            key = (r["speaker_id"], int(r["part"]))
            arow = adf.loc[key] if key in adf.index else None
        paths, qa, tr = [], [], []
        for i in range(6):
            v = r.get(ap[i])
            if isinstance(v, str) and v.strip():
                paths.append(_norm(v.strip()))
                q = r.get(qc[i])
                if asr:
                    t = (arow.get(f"asr_transcript_{i + 1}") if arow is not None else "")
                else:
                    t = r.get(tc[i])
                q = "" if not isinstance(q, str) else q.strip().replace("\n", " ")
                t = "" if not isinstance(t, str) else str(t).strip().replace("\n", " ")
                qa.append(f"Q{i + 1}: {q}\nA{i + 1}: {t}")
                tr.append(t)
        rows.append(dict(
            speaker_id=r["speaker_id"], part=int(r["part"]),
            part_score=float(r["part_score"]), overall_score=float(r["overall_score"]),
            split=split, n_clips=len(paths), audio_paths="|".join(paths),
            llm_input=f"<TASK part=P{int(r['part'])}>\n" + "\n".join(qa),
            transcript_concat=" ".join(tr),
        ))
    out = pd.DataFrame(rows)
    out_suffix = (suffix if asr else "")
    out.to_csv(f"{SRC}/master_{split}{out_suffix}.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asr", action="store_true",
                   help="use ASR transcripts from csv/asr/ -> master_{split}_asr.csv (training source)")
    p.add_argument("--asr_dir", default="asr", help="subdir under csv/ holding {split}.csv ASR")
    p.add_argument("--suffix", default="_asr", help="output suffix -> master_{split}<suffix>.csv")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                   help="splits to build; default: train dev test")
    args = p.parse_args()
    for s in args.splits:
        o = build(s, args.asr, asr_dir=args.asr_dir, suffix=args.suffix)
        print(f"{s:5s}{'(asr)' if args.asr else ''}: {len(o):4d} rows | "
              f"speakers={o.speaker_id.nunique()} | parts={sorted(o.part.unique())} | "
              f"clips {o.n_clips.min()}-{o.n_clips.max()}")
