"""Tests for src.models.baseline (Phase 14)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import baseline


def test_buy_and_hold_always_predicts_up():
    pred = baseline.buy_and_hold_predictions(pd.RangeIndex(5))
    assert (pred == 1.0).all() and len(pred) == 5


def test_previous_direction_follows_last_move_and_abstains_when_flat():
    close = pd.Series([10.0, 11.0, 11.0, 10.5, 12.0])
    pred = baseline.previous_direction_predictions(close)
    assert pd.isna(pred.iloc[0])  # no prior bar
    assert pred.iloc[1] == 1.0  # rose
    assert pd.isna(pred.iloc[2])  # flat: no direction
    assert pred.iloc[3] == 0.0  # fell
    assert pred.iloc[4] == 1.0  # rose


def test_moving_average_up_down_and_warmup():
    close = pd.Series(list(range(1, 21)) + list(range(20, 0, -1)), dtype=float)
    pred = baseline.moving_average_predictions(close, 3, 6)
    assert pred.iloc[:5].isna().all()  # slow SMA not ready until 6 bars
    assert pred.iloc[10] == 1.0  # rising: fast above slow
    assert pred.iloc[-1] == 0.0  # long decline: fast below slow


def test_rsi_oversold_up_overbought_down_else_abstain():
    rising = pd.Series(np.arange(1, 40, dtype=float))  # RSI pinned at 100
    falling = pd.Series(np.arange(40, 1, -1, dtype=float))  # RSI pinned at 0
    assert (baseline.rsi_predictions(rising, 14, 30, 70).dropna() == 0.0).all()  # overbought -> DOWN
    assert (baseline.rsi_predictions(falling, 14, 30, 70).dropna() == 1.0).all()  # oversold -> UP
    # a series that wiggles near 50 never crosses either threshold
    wiggle = pd.Series([100.0, 101.0] * 30)
    assert baseline.rsi_predictions(wiggle, 14, 30, 70).dropna().empty


def test_random_is_reproducible_and_seed_dependent():
    idx = pd.RangeIndex(200)
    a = baseline.random_predictions(idx, 7)
    b = baseline.random_predictions(idx, 7)
    c = baseline.random_predictions(idx, 8)
    assert a.equals(b)
    assert not a.equals(c)
    assert set(a.unique()) <= {0.0, 1.0}
    assert 0.35 < a.mean() < 0.65  # roughly a fair coin


def test_majority_class_uses_only_the_labels_it_is_given():
    train = pd.Series([1.0, 1.0, 0.0, np.nan])
    pred, majority = baseline.majority_class_predictions(train, pd.RangeIndex(4))
    assert majority == 1.0 and (pred == 1.0).all()

    pred2, majority2 = baseline.majority_class_predictions(pd.Series([0.0, 0.0, 1.0]), pd.RangeIndex(3))
    assert majority2 == 0.0 and (pred2 == 0.0).all()


def test_majority_class_tie_goes_to_up_and_empty_train_abstains():
    _, tie = baseline.majority_class_predictions(pd.Series([1.0, 0.0]), pd.RangeIndex(2))
    assert tie == 1.0
    pred, none = baseline.majority_class_predictions(pd.Series([np.nan, np.nan]), pd.RangeIndex(2))
    assert none is None and pred.isna().all()


# ---- causality: Section 58's mutate-the-future check on every predictor ----

np.random.seed(5)
_N, _CUT = 90, 60
_close = pd.Series(100 + np.cumsum(np.random.randn(_N) * 0.5))


def _mutate(series):
    out = series.copy()
    out.iloc[_CUT + 1 :] += 10_000.0
    return out


def _prefix_equal(before, after):
    b, a = before.iloc[: _CUT + 1], after.iloc[: _CUT + 1]
    return ((b == a) | (b.isna() & a.isna())).all()


def test_previous_direction_has_no_lookahead():
    assert _prefix_equal(baseline.previous_direction_predictions(_close),
                         baseline.previous_direction_predictions(_mutate(_close)))


def test_moving_average_has_no_lookahead():
    assert _prefix_equal(baseline.moving_average_predictions(_close, 5, 20),
                         baseline.moving_average_predictions(_mutate(_close), 5, 20))


def test_rsi_has_no_lookahead():
    assert _prefix_equal(baseline.rsi_predictions(_close, 14, 30, 70),
                         baseline.rsi_predictions(_mutate(_close), 14, 30, 70))
