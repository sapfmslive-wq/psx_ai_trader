"""Section 58's mandatory leakage test, applied to every feature function
(same methodology as tests/test_indicators_no_lookahead.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import market_regime, stock_features

np.random.seed(11)
N = 80
CUTOFF = 50

_close = pd.Series(100 + np.cumsum(np.random.randn(N) * 0.5))
_high = _close + np.random.rand(N) * 0.5
_low = _close - np.random.rand(N) * 0.5
_open = _close.shift(1).fillna(_close.iloc[0])
_volume = pd.Series(np.random.randint(100, 1000, N).astype(float))


def _mutate_future(series: pd.Series, cutoff: int) -> pd.Series:
    mutated = series.copy()
    mutated.iloc[cutoff + 1 :] = mutated.iloc[cutoff + 1 :] + 10_000.0
    return mutated


def _assert_prefix_unchanged(before: pd.Series, after: pd.Series, cutoff: int, label: str) -> None:
    prefix_before = before.iloc[: cutoff + 1]
    prefix_after = after.iloc[: cutoff + 1]
    mismatches = ~((prefix_before == prefix_after) | (prefix_before.isna() & prefix_after.isna()))
    assert not mismatches.any(), (
        f"{label}: {mismatches.sum()} value(s) up to index {cutoff} changed after "
        f"mutating only FUTURE data."
    )


def test_index_returns_has_no_lookahead():
    before = market_regime.index_returns(_close, (1, 5, 15, 60))
    after = market_regime.index_returns(_mutate_future(_close, CUTOFF), (1, 5, 15, 60))
    for key in before:
        _assert_prefix_unchanged(before[key], after[key], CUTOFF, f"index_returns.{key}")


def test_index_trend_has_no_lookahead():
    before = market_regime.index_trend(_close, 20, 50)
    after = market_regime.index_trend(_mutate_future(_close, CUTOFF), 20, 50)
    _assert_prefix_unchanged(before, after, CUTOFF, "index_trend")


def test_position_vs_sma_has_no_lookahead():
    before = market_regime.position_vs_sma(_close, 20)
    after = market_regime.position_vs_sma(_mutate_future(_close, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "position_vs_sma")


def test_index_volatility_has_no_lookahead():
    before = market_regime.index_volatility(_close, 20)
    after = market_regime.index_volatility(_mutate_future(_close, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "index_volatility")


def test_log_return_has_no_lookahead():
    before = stock_features.log_return(_close, 5)
    after = stock_features.log_return(_mutate_future(_close, CUTOFF), 5)
    _assert_prefix_unchanged(before, after, CUTOFF, "log_return")


def test_volume_change_has_no_lookahead():
    before = stock_features.volume_change(_volume)
    after = stock_features.volume_change(_mutate_future(_volume, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "volume_change")


def test_volume_acceleration_has_no_lookahead():
    before = stock_features.volume_acceleration(_volume)
    after = stock_features.volume_acceleration(_mutate_future(_volume, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "volume_acceleration")


def test_ma_distance_has_no_lookahead():
    before = stock_features.ma_distance(_close, 20)
    after = stock_features.ma_distance(_mutate_future(_close, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "ma_distance")


def test_high_low_breakout_has_no_lookahead():
    before = stock_features.high_low_breakout(_high, _low, _close, 20)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    after = stock_features.high_low_breakout(mutated_high, mutated_low, mutated_close, 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "high_low_breakout")


def test_intraday_range_has_no_lookahead():
    before = stock_features.intraday_range(_high, _low, _close)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    after = stock_features.intraday_range(mutated_high, mutated_low, mutated_close)
    _assert_prefix_unchanged(before, after, CUTOFF, "intraday_range")


def test_gap_has_no_lookahead():
    previous_close = _close.shift(1)
    before = stock_features.gap(_open, previous_close)
    mutated_open = _mutate_future(_open, CUTOFF)
    mutated_previous_close = _mutate_future(previous_close, CUTOFF)
    after = stock_features.gap(mutated_open, mutated_previous_close)
    _assert_prefix_unchanged(before, after, CUTOFF, "gap")


def test_relative_volume_has_no_lookahead():
    before = stock_features.relative_volume(_volume, 20)
    after = stock_features.relative_volume(_mutate_future(_volume, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "relative_volume")


def test_relative_performance_has_no_lookahead():
    asset_return = market_regime.index_returns(_close, (5,))["return_5bar"]
    benchmark_return = asset_return * 0.5
    before = stock_features.relative_performance(asset_return, benchmark_return)

    mutated_asset_return = market_regime.index_returns(_mutate_future(_close, CUTOFF), (5,))["return_5bar"]
    mutated_benchmark_return = mutated_asset_return * 0.5
    after = stock_features.relative_performance(mutated_asset_return, mutated_benchmark_return)
    _assert_prefix_unchanged(before, after, CUTOFF, "relative_performance")
