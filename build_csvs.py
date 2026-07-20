#!/usr/bin/env python3
"""
Build csv/{train,dev,test}.csv and *_overall.csv from the official
Speak & Improve Corpus 2025 release. This is STEP 0 of the pipeline — it converts
the corpus's native layout into the wide per-(speaker, part) CSVs every other
script in this repo reads. The official *eval* split becomes our *test* split.

Expected corpus layout under $SANDI_CORPUS_ROOT (default: data/sandi), as installed
per the corpus's own instructions:
  example_data.tsv                                   (train recordings + scores + refs)
  reference-materials/sla-marks/{split}-sla-P{1,3,4,5}.tsv  and  {split}-sla-overall.tsv
  reference-materials/annotations/{split}-trans-ref.json    (question + reference transcript)
  reference-materials/annotations/{split}-notrans-quesonly.json  (question-only fallback)
  data/flac/{train,dev,eval}/**/*.flac               (audio)

Main CSV columns (one row per speaker-part):
  speaker_id, part, part_score, overall_score,
  audio_path_1 … audio_path_6,
  question_text_1 … question_text_6,
  transcript_1 … transcript_6

  Parts 1 and 5 have up to 6 / 5 short recordings respectively; columns are
  filled left-to-right in question_id order and left empty when fewer exist.
  Parts 3 and 4 always use only column _1.

Overall CSV columns (one row per speaker):
  speaker_id, part1_score, part3_score, part4_score, part5_score, overall_score

  overall_score = mean(part1, part3, part4, part5) — verified against the
  stored value in the reference files.

Run from the repo root:  SANDI_CORPUS_ROOT=/path/to/corpus python build_csvs.py
"""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get('SANDI_CORPUS_ROOT', 'data/sandi'))
REF  = BASE / 'reference-materials'
FLAC = BASE / 'data' / 'flac'
OUT  = Path(__file__).resolve().parent / 'csv'
OUT.mkdir(parents=True, exist_ok=True)

MAX_SLOTS = 6  # Part 1 has up to 6 short answers; Part 5 up to 5

MAIN_FIELDS = (
    ['speaker_id', 'part', 'part_score', 'overall_score']
    + [f'audio_path_{i}'    for i in range(1, MAX_SLOTS + 1)]
    + [f'question_text_{i}' for i in range(1, MAX_SLOTS + 1)]
    + [f'transcript_{i}'    for i in range(1, MAX_SLOTS + 1)]
)
OVERALL_FIELDS = [
    'speaker_id', 'part1_score', 'part3_score', 'part4_score',
    'part5_score', 'overall_score',
]


def write_csv(rows, fieldnames, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Wrote {len(rows):>6} rows → {path.name}')


def load_sla_marks(split):
    """Return speaker_id → {part1_score, part3_score, part4_score, part5_score, overall_score}."""
    marks = {}
    for part in [1, 3, 4, 5]:
        fpath = REF / 'sla-marks' / f'{split}-sla-P{part}.tsv'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                spk, score = line.split('\t')
                marks.setdefault(spk, {})[f'part{part}_score'] = float(score)

    overall_path = REF / 'sla-marks' / f'{split}-sla-overall.tsv'
    if overall_path.exists():
        with open(overall_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                spk, score = line.split('\t')
                marks.setdefault(spk, {})['overall_score'] = float(score)
    return marks


def verify_overall(marks, label):
    """Check overall_score == mean(part1, part3, part4, part5) within 0.01."""
    errors = 0
    for spk, info in marks.items():
        if 'overall_score' not in info:
            continue
        part_scores = [info[f'part{p}_score'] for p in [1, 3, 4, 5]
                       if f'part{p}_score' in info]
        if len(part_scores) != 4:
            continue
        computed = sum(part_scores) / 4
        if abs(computed - info['overall_score']) > 0.01:
            print(f'  MISMATCH {label} {spk}: computed={computed:.4f} stored={info["overall_score"]:.4f}')
            errors += 1
    status = 'PASSED' if errors == 0 else f'{errors} ERRORS'
    print(f'  Overall score verification [{label}]: {status}')


def _parse_annotation_entry(entry):
    """Parse a single annotation JSON entry into a dict."""
    q_raw = entry.get('Question', '{}')
    try:
        q = json.loads(q_raw)
        question_text = q.get('text', '')
        speaking_time = q.get('speaking-time', '')
    except (json.JSONDecodeError, TypeError):
        question_text = ''
        speaking_time = ''
    words = [item['word'].lower() for item in entry.get('Transcript', []) if 'word' in item]
    return {
        'question_text': question_text,
        'transcript': ' '.join(words),
        'speaking_time': speaking_time,
    }


def load_annotations(split):
    """Return rec_id → {question_text, transcript, speaking_time}.

    Primary source: {split}-trans-ref.json (question + transcript).
    Fallback source: {split}-notrans-quesonly.json (question only, no transcript).
    Entries in trans-ref take precedence.
    """
    rec_info = {}

    # Load fallback first so trans-ref can override
    for fname in [f'{split}-notrans-quesonly.json', f'{split}-trans-ref.json']:
        ann_path = REF / 'annotations' / fname
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            data = json.load(f)
        for entry in data.get('files', []):
            rec_id = entry['File-id']
            parsed = _parse_annotation_entry(entry)
            # Always overwrite: trans-ref (loaded second) wins over notrans
            if rec_id not in rec_info or fname.endswith('-trans-ref.json'):
                rec_info[rec_id] = parsed

    return rec_info


def find_audio_paths(split):
    """Return rec_id → absolute path to .flac file. Downstream tools re-root these paths
    onto $SANDI_AUDIO_ROOT via the stable '/flac/' suffix, so any absolute prefix works."""
    return {f.stem: str(f.resolve()) for f in (FLAC / split).rglob('*.flac')}


def part_from_qid(question_id):
    """P10003 → 1, P30017 → 3, P40018 → 4, P50019 → 5."""
    return int(question_id[1])


def clean_p5_question(question_text):
    """
    P5 questions list all 5 topics with the active one in **bold**.
    Keep the preamble, discard the bullet list, append the selected topic.

    Before: "...He will ask you questions about:\\n\\n* **eating habits** \\n* drinking water \\n..."
    After:  "...He will ask you questions about: eating habits"
    """
    if not question_text:
        return question_text
    bold = re.search(r'\*\*(.+?)\*\*', question_text)
    if not bold:
        return question_text
    topic = bold.group(1).strip()
    # Split at the literal \n\n* that begins the bullet list
    # (stored as two-char sequences in the CSV string)
    preamble = re.split(r'\\n\\n\*', question_text, maxsplit=1)[0].rstrip()
    return f"{preamble} {topic}"


def group_into_main_rows(per_rec_list, marks):
    """
    Group individual recordings into one row per (speaker_id, part).
    per_rec_list: list of dicts with keys speaker_id, part, question_id,
                  audio_path, question_text, transcript, part_score, overall_score.
    Returns list of grouped rows matching MAIN_FIELDS.
    """
    # bucket by (speaker_id, part), preserving question_id order
    buckets = defaultdict(list)
    for rec in per_rec_list:
        buckets[(rec['speaker_id'], rec['part'])].append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: r['question_id'])

    rows = []
    for (speaker_id, part) in sorted(buckets):
        recs = buckets[(speaker_id, part)]
        row = {
            'speaker_id':   speaker_id,
            'part':         part,
            'part_score':   recs[0]['part_score'],
            'overall_score': recs[0]['overall_score'],
        }
        for i in range(1, MAX_SLOTS + 1):
            rec = recs[i - 1] if i <= len(recs) else None
            if rec:
                qt = rec['question_text']
                if part == 5:
                    qt = clean_p5_question(qt)
                row[f'audio_path_{i}']    = rec['audio_path']
                row[f'question_text_{i}'] = qt
                row[f'transcript_{i}']    = rec['transcript']
            else:
                row[f'audio_path_{i}']    = ''
                row[f'question_text_{i}'] = ''
                row[f'transcript_{i}']    = ''
        rows.append(row)
    return rows


# ── TRAIN ──────────────────────────────────────────────────────────────────
print('\n[train]')
train_marks = load_sla_marks('train')
verify_overall(train_marks, 'train')

train_per_rec = []
train_overall_dict = {}  # speaker_id → overall row

# Question-text fallback for train recordings not covered by example_data.tsv
_train_notrans = load_annotations('train')

with open(BASE / 'example_data.tsv') as f:
    next(f)  # skip header
    for line in f:
        cols = line.rstrip('\n').split('\t')
        rec_id        = cols[0]
        audio_path    = cols[1]
        speaker_id    = cols[2]
        part          = int(cols[3])
        question_id   = cols[4]
        p1            = cols[5]
        p3            = cols[6]
        p4            = cols[7]
        p5            = cols[8]
        part_score    = cols[9]
        overall_score = cols[10]
        question_text = cols[11] if len(cols) > 11 else ''
        transcript    = cols[13] if len(cols) > 13 else ''
        if not question_text:
            question_text = _train_notrans.get(rec_id, {}).get('question_text', '')

        train_per_rec.append({
            'speaker_id':    speaker_id,
            'part':          part,
            'question_id':   question_id,
            'audio_path':    audio_path,
            'question_text': question_text,
            'transcript':    transcript,
            'part_score':    float(part_score) if part_score else '',
            'overall_score': float(overall_score) if overall_score else '',
        })

        if speaker_id not in train_overall_dict:
            train_overall_dict[speaker_id] = {
                'speaker_id':   speaker_id,
                'part1_score':  '',
                'part3_score':  '',
                'part4_score':  '',
                'part5_score':  '',
                'overall_score': float(overall_score) if overall_score else '',
            }
        row = train_overall_dict[speaker_id]
        if p1: row['part1_score'] = float(p1)
        if p3: row['part3_score'] = float(p3)
        if p4: row['part4_score'] = float(p4)
        if p5: row['part5_score'] = float(p5)

train_main_rows   = group_into_main_rows(train_per_rec, train_marks)
train_overall_rows = sorted(train_overall_dict.values(), key=lambda r: r['speaker_id'])
write_csv(train_main_rows,    MAIN_FIELDS,    OUT / 'train.csv')
write_csv(train_overall_rows, OVERALL_FIELDS, OUT / 'train_overall.csv')


# ── DEV / EVAL ─────────────────────────────────────────────────────────────
for split, out_name in [('dev', 'dev'), ('eval', 'test')]:
    print(f'\n[{split} → {out_name}]')
    marks       = load_sla_marks(split)
    verify_overall(marks, split)
    annotations = load_annotations(split)
    audio_map   = find_audio_paths(split)

    # Warn about recordings with no score
    no_score = [r for r in audio_map if r.rsplit('-', 1)[0] not in marks]
    if no_score:
        print(f'  WARNING: {len(no_score)} recordings have no SLA marks')

    per_rec = []
    for rec_id in sorted(audio_map):
        parts_split = rec_id.rsplit('-', 1)
        if len(parts_split) != 2:
            print(f'  Skipping unexpected rec_id format: {rec_id}')
            continue
        speaker_id, question_id = parts_split
        part = part_from_qid(question_id)

        spk_marks     = marks.get(speaker_id, {})
        part_score    = spk_marks.get(f'part{part}_score', '')
        overall_score = spk_marks.get('overall_score', '')

        ann = annotations.get(rec_id, {})
        per_rec.append({
            'speaker_id':    speaker_id,
            'part':          part,
            'question_id':   question_id,
            'audio_path':    audio_map[rec_id],
            'question_text': ann.get('question_text', ''),
            'transcript':    ann.get('transcript', ''),
            'part_score':    part_score,
            'overall_score': overall_score,
        })

    main_rows = group_into_main_rows(per_rec, marks)

    overall_rows = []
    for spk in sorted(marks):
        info = marks[spk]
        if 'overall_score' not in info:
            continue
        overall_rows.append({
            'speaker_id':   spk,
            'part1_score':  info.get('part1_score', ''),
            'part3_score':  info.get('part3_score', ''),
            'part4_score':  info.get('part4_score', ''),
            'part5_score':  info.get('part5_score', ''),
            'overall_score': info['overall_score'],
        })

    write_csv(main_rows,    MAIN_FIELDS,    OUT / f'{out_name}.csv')
    write_csv(overall_rows, OVERALL_FIELDS, OUT / f'{out_name}_overall.csv')

print('\nDone.')
