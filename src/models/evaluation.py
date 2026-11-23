"""Evaluation helpers for direction predictors (Phase 14).

Prediction quality only. Section 81 is explicit that accuracy is not
profitability — a 60%-accurate model can still lose money after fees,
slippage and spread — and costs are not modeled until the Phase 16
backtester. Nothing here should be read as a profit claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_INDEPENDENT_OUTCOMES = 30


@dataclass(frozen=True)
class ClassificationMetrics:
    n_labeled: int  # bars with a known label in the evaluated segment
    n_predicted: int  # of those, bars the predictor actually called
    coverage: float  # n_predicted / n_labeled
    accuracy: float
    balanced_accuracy: float
    precision_up: float
    recall_up: float
    f1_up: float
    null_accuracy: float  # best a constant guess would do on the same rows
    tp: int
    fp: int
    tn: int
    fn: int


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    """Confusion-matrix metrics with UP (1.0) as the positive class.

    Rows where the label is unknown are dropped. Rows where the predictor
    abstained are excluded from accuracy/precision/recall but counted in
    `coverage`. Undefined ratios (e.g. recall when no UP bars were scored)
    are NaN, never silently zero.
    """
    y_true, y_pred = y_true.align(y_pred, join="left")
    labeled = y_true.notna()
    n_labeled = int(labeled.sum())
    scored = labeled & y_pred.notna()
    n_predicted = int(scored.sum())

    truth = y_true[scored].to_numpy()
    guess = y_pred[scored].to_numpy()
    tp = int(((guess == 1) & (truth == 1)).sum())
    fp = int(((guess == 1) & (truth == 0)).sum())
    tn = int(((guess == 0) & (truth == 0)).sum())
    fn = int(((guess == 0) & (truth == 1)).sum())

    recall_up = _ratio(tp, tp + fn)
    recall_down = _ratio(tn, tn + fp)
    recalls = [r for r in (recall_up, recall_down) if not math.isnan(r)]
    balanced = float(np.mean(recalls)) if recalls else float("nan")

    precision_up = _ratio(tp, tp + fp)
    if math.isnan(precision_up) or math.isnan(recall_up):
        f1 = float("nan")
    elif precision_up + recall_up == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision_up * recall_up / (precision_up + recall_up)

    up_rate = float(truth.mean()) if n_predicted else float("nan")
    null_accuracy = max(up_rate, 1 - up_rate) if n_predicted else float("nan")

    return ClassificationMetrics(
        n_labeled=n_labeled,
        n_predicted=n_predicted,
        coverage=_ratio(n_predicted, n_labeled),
        accuracy=_ratio(tp + tn, n_predicted),
        balanced_accuracy=balanced,
        precision_up=precision_up,
        recall_up=recall_up,
        f1_up=f1,
        null_accuracy=null_accuracy,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def effective_sample_size(n_predicted: int, horizon: int) -> int:
    """Approximate number of INDEPENDENT outcomes among `n_predicted` scored bars.

    A label looks `horizon` bars ahead, so consecutive labels share
    horizon-1 of their horizon returns; roughly one outcome in every
    `horizon` is independent. Counting every bar as independent made the
    naive test call pure random walks 'significant' up to 35% of the time
    (measured, 300 trials, nominal 5%); thinning by the horizon brings that
    to at or below the nominal rate. It remains an approximation — a
    persistent signal (like a moving-average state) is still correlated
    across bars — so it is deliberately on the conservative side.
    """
    return max(0, n_predicted // max(1, horizon))


def approx_p_value_vs_chance(accuracy: float, n: int) -> float:
    """Two-sided normal-approximation p-value for 'accuracy differs from 0.5'.

    Pass the EFFECTIVE sample size (see `effective_sample_size`), not the
    raw bar count — with overlapping label horizons the raw count wildly
    overstates the evidence. Even so, treat a small p-value as 'worth a
    closer look', never as proof.
    """
    if n <= 0 or math.isnan(accuracy):
        return float("nan")
    z = (accuracy - 0.5) / math.sqrt(0.25 / n)
    return math.erfc(abs(z) / math.sqrt(2))


def verdict(metrics: ClassificationMetrics, p_value: float, n_effective: int, alpha: float = 0.05) -> str:
    """A plain-language, deliberately conservative read of one result.

    `alpha` is the significance bar; tighten it (e.g. 0.05 / number of models
    compared) when several candidates are tested against the same data, since
    each extra comparison is another chance for noise to look like skill."""
    if n_effective < MIN_INDEPENDENT_OUTCOMES or math.isnan(p_value):
        return f"INSUFFICIENT DATA (fewer than {MIN_INDEPENDENT_OUTCOMES} independent outcomes)"
    if p_value >= alpha:
        return "NOT DISTINGUISHABLE FROM CHANCE"
    if metrics.accuracy < 0.5:
        return "SIGNIFICANTLY WORSE THAN CHANCE (anti-predictive on this sample; not a usable signal)"
    if metrics.accuracy <= metrics.null_accuracy:
        return "ABOVE 50% BUT NO BETTER THAN ALWAYS GUESSING THE MAJORITY CLASS"
    return "BEATS A CONSTANT GUESS (approximate test, adjusted for overlapping labels; still not proof)"


def chronological_split(n_bars: int, train_fraction: float, horizon: int) -> tuple[int, int]:
    """Return (train_end, test_start) as positional indices.

    Train is rows [0, train_end); test is rows [test_start, n_bars). The
    series is never shuffled. The last `horizon` bars before the split are
    PURGED from train: their labels are computed from prices at or after
    the split, so using them to fit anything would leak the test period
    into training.
    """
    test_start = int(n_bars * train_fraction)
    train_end = max(0, test_start - horizon)
    return train_end, test_start
