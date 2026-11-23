"""Probability-quality metrics for the ML models (Phase 15, Sections 27/82/83).

Accuracy alone is not enough: with imbalanced classes it can look good while
the model never finds a single DOWN bar, and a model can rank bars well
(high ROC-AUC) while its stated probabilities are badly miscalibrated. So
every model is reported with balanced accuracy, precision/recall/F1 for UP,
ROC-AUC, PR-AUC, Brier score, log loss, and a calibration table (when the
model says 70%, does UP actually happen ~70% of the time?).

UP (1.0) is the positive class. Undefined metrics are NaN, never 0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.models import evaluation as ev

_EPS = 1e-6


@dataclass(frozen=True)
class ProbabilityMetrics:
    n: int
    up_rate: float
    classification: ev.ClassificationMetrics  # at the decision threshold
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ece: float  # expected calibration error


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def calibration_table(y_true, p_up, bins: int = 10) -> list[dict]:
    """Equal-width probability bins: how many bars, the mean predicted
    probability, and the observed UP rate. Empty bins are omitted."""
    y = np.asarray(y_true, dtype=float)
    p = _clip(p_up)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    table = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        table.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()), "observed_up_rate": float(y[mask].mean()),
        })
    return table


def expected_calibration_error(table: list[dict]) -> float:
    total = sum(row["n"] for row in table)
    if total == 0:
        return float("nan")
    return float(sum(row["n"] / total * abs(row["observed_up_rate"] - row["mean_predicted"]) for row in table))


def probability_metrics(y_true, p_up, threshold: float = 0.5, bins: int = 10) -> ProbabilityMetrics:
    """All metrics for one set of predicted UP probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = _clip(p_up)
    pred = pd.Series((p >= threshold).astype(float))
    cls = ev.classification_metrics(pd.Series(y), pred)

    has_both = len(np.unique(y)) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roc_auc = float(roc_auc_score(y, p)) if has_both else float("nan")
        pr_auc = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    table = calibration_table(y, p, bins)
    return ProbabilityMetrics(
        n=len(y),
        up_rate=float(y.mean()) if len(y) else float("nan"),
        classification=cls,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=float(brier_score_loss(y, p)) if len(y) else float("nan"),
        log_loss=float(log_loss(y, p, labels=[0.0, 1.0])) if len(y) else float("nan"),
        ece=expected_calibration_error(table),
    )
