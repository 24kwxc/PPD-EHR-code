import math
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / total) * abs(conf - acc)
    return float(ece)


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    label_names: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
    rare_prevalence: float = 0.10,
) -> Dict:
    y_true = (np.asarray(y_true) > 0.5).astype(np.int32)
    y_prob = np.nan_to_num(np.asarray(y_prob), nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(np.int32)
    n_labels = y_true.shape[1]
    names = list(label_names or [f"label_{i}" for i in range(n_labels)])

    per_label = []
    aurocs, auprcs, briers, eces = [], [], [], []
    rare_recalls = []

    for idx in range(n_labels):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        prevalence = float(np.mean(yt))
        row = {
            "label": names[idx],
            "index": idx,
            "prevalence": prevalence,
            "support": int(np.sum(yt)),
        }
        if 0 < np.sum(yt) < len(yt):
            row["auroc"] = float(roc_auc_score(yt, yp))
            row["auprc"] = float(average_precision_score(yt, yp))
            row["brier"] = float(brier_score_loss(yt, yp))
            row["ece"] = _ece_binary(yt, yp)
            aurocs.append(row["auroc"])
            auprcs.append(row["auprc"])
            briers.append(row["brier"])
            eces.append(row["ece"])
        else:
            row["auroc"] = None
            row["auprc"] = None
            row["brier"] = None
            row["ece"] = None
        row["precision"] = float(precision_score(yt, y_pred[:, idx], zero_division=0))
        row["recall"] = float(recall_score(yt, y_pred[:, idx], zero_division=0))
        row["f1"] = float(f1_score(yt, y_pred[:, idx], zero_division=0))
        if prevalence <= rare_prevalence and row["support"] > 0:
            rare_recalls.append(row["recall"])
        per_label.append(row)

    return {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(n_labels),
        "threshold": float(threshold),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
        "macro_brier": float(np.mean(briers)) if briers else 0.0,
        "macro_ece": float(np.mean(eces)) if eces else 0.0,
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "rare_label_recall": float(np.mean(rare_recalls)) if rare_recalls else 0.0,
        "per_label": per_label,
    }


def mean_std_summary(rows: List[Dict], metric_keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def aggregate_seed_metrics(seed_runs: List[Dict], metric_keys: Optional[Iterable[str]] = None) -> Dict:
    metric_keys = list(metric_keys or [
        "macro_auroc",
        "macro_auprc",
        "macro_brier",
        "macro_ece",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "rare_label_recall",
    ])
    summary = mean_std_summary(seed_runs, metric_keys)
    summary["n_seeds"] = int(len(seed_runs))
    return summary


def flatten_per_label_rows(
    *,
    experiment: str,
    split_name: str,
    model_name: str,
    seed: int,
    metrics: Dict,
) -> List[Dict]:
    rows: List[Dict] = []
    for row in metrics.get("per_label", []):
        rows.append(
            {
                "experiment": experiment,
                "split": split_name,
                "model": model_name,
                "seed": seed,
                **row,
            }
        )
    return rows
