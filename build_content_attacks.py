"""Build SANDI test masters with deliberately mismatched task questions.

Audio, OpenAI Whisper transcripts, and original score columns stay fixed. Only
the questions embedded in ``llm_input`` change, so attacked rows have no valid
gold score; evaluate them by score change relative to the clean prediction.
"""

import argparse
from pathlib import Path

import pandas as pd


AUDIO_ROOT = "/scratch/elec/t405-puhe/c/sandi2025/data/flac"
FLAC_MARKER = "/flac/"
SWAP_PART = {1: 5, 5: 1, 3: 4, 4: 3}
DEFAULT_UNRELATED_QUESTION = (
    "How does a nuclear reactor control a chain reaction, and what is the "
    "purpose of its control rods?"
)


def _text(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _audio_path(value) -> str:
    path = _text(value)
    if FLAC_MARKER in path:
        return f"{AUDIO_ROOT}/{path.split(FLAC_MARKER, 1)[1]}"
    return path


def _clips(row: pd.Series, asr_row: pd.Series) -> list[dict]:
    clips = []
    for index in range(1, 7):
        path = _audio_path(row.get(f"audio_path_{index}"))
        if not path:
            continue
        clips.append({
            "path": path,
            "question": _text(row.get(f"question_text_{index}")),
            "answer": _text(asr_row.get(f"asr_transcript_{index}")),
        })
    return clips


def _master_row(row: pd.Series, clips: list[dict], questions: list[str],
                attack: str, question_source_part: str) -> dict:
    qa = [
        f"Q{index}: {question}\nA{index}: {clip['answer']}"
        for index, (clip, question) in enumerate(zip(clips, questions), start=1)
    ]
    part = int(row["part"])
    return {
        "speaker_id": str(row["speaker_id"]),
        "part": part,
        "part_score": float(row["part_score"]),
        "overall_score": float(row["overall_score"]),
        "split": "test",
        "n_clips": len(clips),
        "audio_paths": "|".join(clip["path"] for clip in clips),
        "llm_input": f"<TASK part=P{part}>\n" + "\n".join(qa),
        "transcript_concat": " ".join(clip["answer"] for clip in clips),
        "attack": attack,
        "question_source_part": question_source_part,
    }


def build(raw_csv: str, asr_csv: str, output_dir: str,
          unrelated_question: str = DEFAULT_UNRELATED_QUESTION) -> tuple[Path, Path]:
    raw = pd.read_csv(raw_csv)
    asr = pd.read_csv(asr_csv)
    if asr.duplicated(["speaker_id", "part"]).any():
        raise ValueError("ASR input has duplicate (speaker_id, part) rows")

    raw["_speaker_key"] = raw["speaker_id"].astype(str)
    asr["_speaker_key"] = asr["speaker_id"].astype(str)
    raw_index = raw.set_index(["_speaker_key", "part"], drop=False)
    asr_index = asr.set_index(["_speaker_key", "part"], drop=False)

    swapped_rows = []
    unrelated_rows = []
    unchanged_questions = 0
    for key, row in raw_index.iterrows():
        speaker_id, part = key[0], int(key[1])
        if part not in SWAP_PART:
            continue
        if key not in asr_index.index:
            raise KeyError(f"Missing ASR row for {key}")

        target_clips = _clips(row, asr_index.loc[key])
        source_key = (speaker_id, SWAP_PART[part])
        if source_key not in raw_index.index or source_key not in asr_index.index:
            raise KeyError(f"Missing swap source row for {source_key}")
        source_clips = _clips(raw_index.loc[source_key], asr_index.loc[source_key])
        source_questions = [clip["question"] for clip in source_clips]
        if not source_questions:
            raise ValueError(f"No source questions for {source_key}")

        swapped_questions = [
            source_questions[min(index, len(source_questions) - 1)]
            for index in range(len(target_clips))
        ]
        unchanged_questions += sum(
            question == clip["question"]
            for question, clip in zip(swapped_questions, target_clips)
        )
        swapped_rows.append(_master_row(
            row, target_clips, swapped_questions, "cross_part_swap",
            f"P{SWAP_PART[part]}"))

        unrelated_rows.append(_master_row(
            row, target_clips, [unrelated_question] * len(target_clips),
            "unrelated_question", "fixed_unrelated"))

    swapped = pd.DataFrame(swapped_rows)
    unrelated = pd.DataFrame(unrelated_rows)
    expected = len(raw)
    if len(swapped) != expected or len(unrelated) != expected:
        raise ValueError(
            f"Expected {expected} rows per attack, got {len(swapped)} and {len(unrelated)}")
    if swapped[["speaker_id", "part"]].duplicated().any():
        raise ValueError("Attack output has duplicate (speaker_id, part) rows")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    swapped_path = destination / "master_test_asr_cross_part_swap.csv"
    unrelated_path = destination / "master_test_asr_unrelated_question.csv"
    swapped.to_csv(swapped_path, index=False)
    unrelated.to_csv(unrelated_path, index=False)

    print(f"cross-part swap: {swapped_path} ({len(swapped)} rows)")
    print(f"unrelated question: {unrelated_path} ({len(unrelated)} rows)")
    print(f"unchanged question slots after swap: {unchanged_questions}")
    return swapped_path, unrelated_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="csv/test.csv")
    parser.add_argument("--asr_csv", default="csv/asr/test.csv")
    parser.add_argument("--output_dir", default="csv/attacks")
    parser.add_argument("--unrelated_question", default=DEFAULT_UNRELATED_QUESTION)
    args = parser.parse_args()
    build(args.raw_csv, args.asr_csv, args.output_dir, args.unrelated_question)


if __name__ == "__main__":
    main()
