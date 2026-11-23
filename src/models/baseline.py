"""Baseline direction predictors (Phase 14, Section 23).

Every predictor returns a Series aligned to its input, with
    1.0 = predicts UP,  0.0 = predicts DOWN,  NaN = no opinion (abstains).
Abstaining is deliberate: forcing a guess where a rule has no signal would
inflate or deflate its apparent skill. Evaluation reports coverage so an
abstaining rule is never compared as if it had called every bar.

All predictors except `majority_class_predictions` are causal: the value at
bar t uses only bars at or before t (tests/test_baselines.py enforces this
with the mutate-the-future check). The majority-class predictor instead
takes *training labels explicitly*, so it is structurally unable to peek at
the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import momentum, trend


def buy_and_hold_predictions(index: pd.Index) -> pd.Series:
    """Always predicts UP — the 'just stay long' stance."""
    return pd.Series(1.0, index=index)


def previous_direction_predictions(close: pd.Series) -> pd.Series:
    """Predicts that the last bar's move continues. Flat bars and the first
    bar have no direction, so they abstain."""
    change = close.diff()
    pred = pd.Series(np.nan, index=close.index)
    pred[change > 0] = 1.0
    pred[change < 0] = 0.0
    return pred


def moving_average_predictions(close: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
    """UP when the fast SMA is above the slow SMA, DOWN when below.
    Abstains during warm-up and when the two are exactly equal."""
    fast = trend.sma(close, fast_period)
    slow = trend.sma(close, slow_period)
    pred = pd.Series(np.nan, index=close.index)
    pred[fast > slow] = 1.0
    pred[fast < slow] = 0.0
    return pred


def rsi_predictions(close: pd.Series, period: int, oversold: float, overbought: float) -> pd.Series:
    """Mean-reversion: oversold (RSI < oversold) predicts UP, overbought
    (RSI > overbought) predicts DOWN, anything in between abstains."""
    rsi_series = momentum.rsi(close, period)
    pred = pd.Series(np.nan, index=close.index)
    pred[rsi_series < oversold] = 1.0
    pred[rsi_series > overbought] = 0.0
    return pred


def random_predictions(index: pd.Index, seed: int) -> pd.Series:
    """Fair-coin UP/DOWN guesses, reproducible from `seed` (Section 59)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, len(index)).astype(float), index=index)


def majority_class_predictions(train_labels: pd.Series, index: pd.Index) -> tuple[pd.Series, float | None]:
    """Predicts whichever direction was more common in the TRAINING labels,
    for every bar. Ties go to UP. Returns (predictions, majority_class);
    if the training set has no labels at all, predictions are all NaN and
    the class is None.
    """
    known = train_labels.dropna()
    if known.empty:
        return pd.Series(np.nan, index=index), None
    majority = 1.0 if (known == 1.0).sum() >= (known == 0.0).sum() else 0.0
    return pd.Series(majority, index=index), majority
