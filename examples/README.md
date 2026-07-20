# Example data files (synthetic)

Fully **synthetic** samples of the raw input format — no Speak & Improve corpus content.

```
csv/{train,dev,test}.csv    # the wide per-(speaker, part) format the whole pipeline reads
```

This is the only format you need to produce yourself (normally via `build_csvs.py` from the
official corpus). Everything else — `csv/asr/{split}.csv`, `master_{split}[_asr].csv` —
is generated from these by the released scripts (`generate_asr.py`, `build_master.py`).

Notes on the format visible in the samples:
- one row per speaker×part; parts 1/5 have multiple clips (slots filled left-to-right,
  unused `_i` columns empty), parts 3/4 use only slot `_1`;
- `part_score` / `overall_score` on the 0–6 scale; `overall_score` = mean of the four parts;
- audio paths only need to contain the `/flac/` marker — they are re-rooted onto
  `$SANDI_AUDIO_ROOT` at load time (the `/path/to/corpus/...` prefixes here are placeholders);
- real data goes in `csv/` at the repo root, not in this folder.
