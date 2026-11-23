"""Tests for src.features.labels (Phase 13), including the label-shifting
and leakage tests Sections 21/101 require."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import labels, market_regime, stock_features


def test_future_return_matches_hand_calculation():
    close = pd.Series([100.0, 110.0, 99.0, 108.9])
    result = labels.future_return(close, 1)
    assert np.isclose(result.iloc[0], 10.0)  # 100 -> 110
    assert np.isclose(result.iloc[1], -10.0)  # 110 -> 99
    assert np.isclose(result.iloc[2], 10.0)  # 99 -> 108.9
    assert pd.isna(result.iloc[3])  # no future bar exists


def test_label_at_t_comes_from_bar_t_plus_horizon_not_t():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    result = labels.future_return(close, 3)
    assert np.isclose(result.iloc[0], (103.0 - 100.0) / 100.0 * 100)
    assert np.isclose(result.iloc[2], (105.0 - 102.0) / 102.0 * 100)


def test_last_horizon_rows_are_nan_never_filled():
    close = pd.Series(np.linspace(100, 110, 20))
    for h in (1, 5, 7):
        result = labels.future_return(close, h)
        assert result.iloc[-h:].isna().all()
        assert result.iloc[:-h].notna().all()


def test_direction_label_up_down_and_flat_is_nan():
    ret = pd.Series([1.5, -0.2, 0.0, np.nan])
    result = labels.direction_label(ret)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.0
    assert pd.isna(result.iloc[2])  # exactly flat: no direction, not forced
    assert pd.isna(result.iloc[3])


def test_three_class_label_bands():
    ret = pd.Series([0.5, 0.1, 0.0, -0.1, -0.5, np.nan])
    result = labels.three_class_label(ret, 0.1)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.0  # exactly on the band edge is NEUTRAL
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 0.0
    assert result.iloc[4] == -1.0
    assert pd.isna(result.iloc[5])


def test_non_positive_start_price_gives_an_undefined_label_not_an_inverted_one():
    # dividing by a negative price flips the sign: a rise from -10 to -5 would
    # read as a -50% "fall". It must be NaN instead.
    close = pd.Series([-10.0, -5.0, 100.0, 110.0, 0.0, 50.0])
    result = labels.future_return(close, 1)
    assert pd.isna(result.iloc[0])  # start price -10
    assert pd.isna(result.iloc[4])  # start price 0
    assert np.isclose(result.iloc[2], 10.0)  # ordinary positive case unaffected


def _bars(highs, lows, closes):
    return pd.Series(highs, dtype=float), pd.Series(lows, dtype=float), pd.Series(closes, dtype=float)


def test_tp_before_sl_take_profit_first():
    # entry 100, tp 101 (1%), sl 99.5 (0.5%). Bar 1 reaches 101.5 without touching 99.5.
    high, low, close = _bars([100, 101.5, 100, 100], [100, 100, 100, 100], [100, 101, 100, 100])
    result = labels.tp_before_sl_label(high, low, close, 2, 1.0, 0.5)
    assert result.iloc[0] == 1.0


def test_tp_before_sl_stop_loss_first():
    high, low, close = _bars([100, 100.2, 102, 100], [100, 99.0, 100, 100], [100, 99.5, 101, 100])
    result = labels.tp_before_sl_label(high, low, close, 2, 1.0, 0.5)
    assert result.iloc[0] == 0.0  # SL touched on bar 1, before TP on bar 2


def test_tp_before_sl_same_bar_both_barriers_is_pessimistic():
    high, low, close = _bars([100, 102, 100, 100], [100, 99.0, 100, 100], [100, 100, 100, 100])
    result = labels.tp_before_sl_label(high, low, close, 2, 1.0, 0.5)
    assert result.iloc[0] == 0.0  # bar touches both: assume the worse one


def test_tp_before_sl_timeout_is_zero():
    high, low, close = _bars([100, 100.3, 100.2, 100.1], [100, 99.8, 99.9, 99.9], [100] * 4)
    result = labels.tp_before_sl_label(high, low, close, 2, 1.0, 0.5)
    assert result.iloc[0] == 0.0  # neither barrier reached within 2 bars


def test_tp_before_sl_last_horizon_rows_are_nan():
    n = 12
    close = pd.Series(np.linspace(100, 101, n))
    result = labels.tp_before_sl_label(close + 0.1, close - 0.1, close, 4, 1.0, 0.5)
    assert result.iloc[-4:].isna().all()
    assert result.iloc[:-4].notna().all()


# ---- leakage: labels and features must depend on opposite sides of T ----

def _frame(n=80, seed=3):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)))
    return close, close + rng.random(n) * 0.5, close - rng.random(n) * 0.5


def test_mutating_the_past_does_not_change_labels_from_t_onward():
    close, high, low = _frame()
    k = 40
    base = labels.future_return(close, 5)
    mutated = close.copy()
    mutated.iloc[:k] = mutated.iloc[:k] + 5000.0  # only bars BEFORE k
    after = labels.future_return(mutated, 5)
    # labels at bars >= k look only forward from there, so they are unchanged
    assert np.allclose(base.iloc[k:].dropna(), after.iloc[k:].dropna())


def test_mutating_the_future_changes_labels_but_not_features_at_t():
    close, high, low = _frame()
    cutoff = 50
    mutated = close.copy()
    mutated.iloc[cutoff + 1 :] = mutated.iloc[cutoff + 1 :] + 10_000.0

    label_before = labels.future_return(close, 5)
    label_after = labels.future_return(mutated, 5)
    # the label at T = cutoff depends on bars after T, so it MUST change...
    assert not np.isclose(label_before.iloc[cutoff], label_after.iloc[cutoff])

    # ...while every feature at or before T must NOT.
    for before, after in (
        (market_regime.position_vs_sma(close, 20), market_regime.position_vs_sma(mutated, 20)),
        (stock_features.log_return(close, 5), stock_features.log_return(mutated, 5)),
        (market_regime.index_trend(close, 5, 20), market_regime.index_trend(mutated, 5, 20)),
    ):
        b, a = before.iloc[: cutoff + 1], after.iloc[: cutoff + 1]
        assert ((b == a) | (b.isna() & a.isna())).all()


def test_tp_before_sl_label_at_t_ignores_everything_at_or_before_t_minus_one():
    close, high, low = _frame()
    k = 40
    base = labels.tp_before_sl_label(high, low, close, 5, 1.0, 0.5)
    h2, l2, c2 = high.copy(), low.copy(), close.copy()
    h2.iloc[:k] += 5000.0
    l2.iloc[:k] += 5000.0
    c2.iloc[:k] += 5000.0
    after = labels.tp_before_sl_label(h2, l2, c2, 5, 1.0, 0.5)
    assert ((base.iloc[k:] == after.iloc[k:]) | (base.iloc[k:].isna() & after.iloc[k:].isna())).all()
