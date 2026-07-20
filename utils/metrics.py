"""Eval metrics for the SANDI scorer.

The model predicts ONE part score per (speaker, part) row and returns logits as (N, 2):
  col 0 = full model (Whisper-LoRA acoustic prior + Qwen reasoning)
  col 1 = acoustic-only aux head (Whisper-LoRA prior → Linear, no Qwen)

Headline metric = speaker-grouped OVERALL RMSE on col 0 (full model).
Score-level secondary metrics include MAE, Pearson correlation, and Spearman
rank correlation. Label-level precision/recall/F1 are reported with both macro
and support-weighted averaging after coarse CEFR conversion.
Macro RMSE is also reported by coarse gold CEFR band so rare levels count equally:
  rmse_overall_macro      = mean per-band continuous RMSE
  rmse_overall_macro_cefr = mean per-band RMSE after floor-style CEFR binning
                            (A2+ -> A2, B1+ -> B1, etc.)
Aux metrics (col 1) show what the acoustic branch alone achieves — the diff vs col 0
quantifies how much Qwen reasoning adds on top of the acoustic prior.
Per-part metrics (rmse_P1/P3/P4/P5 + aux_*) break the part RMSE down by SANDI part, so we
can see which parts the model struggles on (P1/P5 are multi-clip → harder than P3/P4) and
compare against prior work that scores only a subset of the parts.

Trainer eval preserves dataset order, so predictions align to the eval CSV by position.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

CEFR_LABELS = {
    0: "<A1",
    1: "A1",
    2: "A2",
    3: "B1",
    4: "B2",
    5: "C1",
    6: "C2",
}

def coarse_cefr_numeric(scores) -> np.ndarray:
    """Map continuous CEFR scores to coarse numeric bands.

    This intentionally floors half-step/plus levels into the lower coarse band:
    2.5 (A2+) -> 2 (A2), 3.5 (B1+) -> 3 (B1), etc. Finnish
    fair scores may also use 0 for <A1.
    """
    arr = np.asarray(scores, dtype=float)
    return np.clip(np.floor(arr + 1e-8), 0, 6).astype(int)


def coarse_cefr_label(scores) -> np.ndarray:
    nums = coarse_cefr_numeric(scores)
    return np.asarray([CEFR_LABELS.get(int(x), str(int(x))) for x in nums], dtype=object)


def _macro_rmse_by_cefr(preds: np.ndarray, true_scores: np.ndarray, tag: str,
                        band_fn=None, label_map=None) -> dict:
    """Equal-weight RMSE across true CEFR bands.

    `band_fn(scores) -> int array` selects the banding; default is the coarse integer
    floor (`coarse_cefr_numeric`), which is what the English SANDI runs use.
    `label_map(band) -> str` names each band; default is `CEFR_LABELS`.
    """
    out = {}
    _band = band_fn if band_fn is not None else coarse_cefr_numeric
    _label = label_map if label_map is not None else (
        lambda b: CEFR_LABELS.get(int(b), str(int(b))))
    true_band = _band(true_scores)
    pred_band = _band(preds)

    macro_cont = []
    macro_cefr = []
    for band in sorted(np.unique(true_band)):
        mask = true_band == band
        if mask.sum() == 0:
            continue
        label = _label(band)
        cont_rmse = float(np.sqrt(np.mean((preds[mask] - true_scores[mask]) ** 2)))
        cefr_rmse = float(np.sqrt(np.mean((pred_band[mask] - true_band[mask]) ** 2)))
        macro_cont.append(cont_rmse)
        macro_cefr.append(cefr_rmse)
        out[f"{tag}rmse_overall_{label}"] = cont_rmse
        out[f"{tag}rmse_overall_cefr_{label}"] = cefr_rmse
        out[f"{tag}n_overall_{label}"] = int(mask.sum())

    if macro_cont:
        out[f"{tag}rmse_overall_macro"] = float(np.mean(macro_cont))
        out[f"{tag}rmse_overall_macro_cefr"] = float(np.mean(macro_cefr))
    return out


def _classification_metrics(preds: np.ndarray, true_scores: np.ndarray, tag: str) -> dict:
    """Precision/recall/F1 after converting continuous scores to CEFR labels."""
    pred_band = coarse_cefr_numeric(preds)
    true_band = coarse_cefr_numeric(true_scores)

    out = {}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_band,
            pred_band,
            average=average,
            zero_division=0,
        )
        out[f"{tag}precision_overall_{average}"] = float(precision)
        out[f"{tag}recall_overall_{average}"] = float(recall)
        out[f"{tag}f1_overall_{average}"] = float(f1)
    return out


def _overall_metrics(preds: np.ndarray, speaker_ids: np.ndarray,
                     overall_scores: np.ndarray, labels: np.ndarray, tag: str,
                     band_fn=None, label_map=None) -> dict:
    """Regression and label metrics at part and speaker-grouped overall levels."""
    rmse_part = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae_part  = float(np.mean(np.abs(preds - labels)))

    df = pd.DataFrame({"spk": speaker_ids, "pred": preds, "true_overall": overall_scores})
    g  = df.groupby("spk").agg(pred_overall=("pred", "mean"),
                                true_overall=("true_overall", "first"))
    diff = g["pred_overall"].to_numpy() - g["true_overall"].to_numpy()
    rmse_overall = float(np.sqrt(np.mean(diff ** 2)))
    mae_overall  = float(np.mean(np.abs(diff)))
    po, to = g["pred_overall"].to_numpy(), g["true_overall"].to_numpy()
    both_vary = po.std() > 1e-8 and to.std() > 1e-8
    corr = float(np.corrcoef(po, to)[0, 1]) if both_vary else 0.0
    spearman = float(spearmanr(po, to).statistic) if both_vary else 0.0

    out = {
        f"{tag}rmse_overall": rmse_overall,
        f"{tag}mae_overall":  mae_overall,
        f"{tag}corr_overall": corr,
        f"{tag}spearman_overall": spearman,
        f"{tag}rmse_part":    rmse_part,
        f"{tag}mae_part":     mae_part,
    }
    out.update(_macro_rmse_by_cefr(po, to, tag, band_fn=band_fn, label_map=label_map))
    out.update(_classification_metrics(po, to, tag))
    return out


def _per_part_rmse(preds: np.ndarray, labels: np.ndarray, parts: np.ndarray, tag: str) -> dict:
    """RMSE for each SANDI part separately -> {tag}rmse_P{1,3,4,5}."""
    out = {}
    for p in np.unique(parts):
        m = parts == p
        if m.sum() == 0:
            continue
        out[f"{tag}rmse_P{int(p)}"] = float(np.sqrt(np.mean((preds[m] - labels[m]) ** 2)))
    return out


def make_compute_metrics(speaker_ids: np.ndarray, overall_scores: np.ndarray,
                         parts: np.ndarray | None = None,
                         band_fn=None, label_map=None):
    speaker_ids    = np.asarray(speaker_ids)
    overall_scores = np.asarray(overall_scores, dtype=float)
    parts          = np.asarray(parts) if parts is not None else None

    def compute(eval_pred):
        raw = np.asarray(eval_pred.predictions, dtype=float)   # (N,) or (N, 2)
        labels = np.asarray(eval_pred.label_ids, dtype=float).reshape(-1)

        if raw.ndim == 2 and raw.shape[1] == 2:
            preds_main = raw[:, 0]
            preds_aux  = raw[:, 1]
        else:
            preds_main = raw.reshape(-1)
            preds_aux  = None

        out = _overall_metrics(preds_main, speaker_ids, overall_scores, labels, tag="",
                               band_fn=band_fn, label_map=label_map)
        out["n_speakers"] = int(pd.Series(speaker_ids).nunique())
        if parts is not None:
            out.update(_per_part_rmse(preds_main, labels, parts, tag=""))

        # Auxiliary (acoustic-only) metrics — non-zero only when use_acoustic=True
        if preds_aux is not None and preds_aux.std() > 1e-8:
            out.update(_overall_metrics(preds_aux, speaker_ids, overall_scores, labels, tag="aux_",
                                        band_fn=band_fn, label_map=label_map))
            if parts is not None:
                out.update(_per_part_rmse(preds_aux, labels, parts, tag="aux_"))

        return out

    return compute
