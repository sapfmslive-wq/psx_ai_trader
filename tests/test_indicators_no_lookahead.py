"""Section 58's mandatory leakage test, applied to every indicator:

    "Modify future prices. Recalculate features at time T. Features
    available at T MUST NOT change. If they change: FAIL TEST."

For each indicator, this computes it once on a base series, then again on
a version of that series where only values strictly AFTER a cutoff index
have been mutated, and asserts every value up to and including the cutoff
is identical (or identically NaN) in both runs. If any indicator implicitly
used future data — a centered rolling window, a backward fill, a shift in
the wrong direction — this test catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import momentum, trend, volatility
from src.indicators import volume as vol_ind

np.random.seed(7)
N = 80
CUTOFF = 50  # mutate everything after this index; [0, CUTOFF] must be stable

_dates = pd.date_range("2026-09-01", periods=N, freq="1min")
_close = pd.Series(100 + np.cumsum(np.random.randn(N) * 0.5), index=_dates)
_high = _close + np.random.rand(N) * 0.5
_low = _close - np.random.rand(N) * 0.5
_volume = pd.Series(np.random.randint(100, 1000, N).astype(float), index=_dates)


def _mutate_future(series: pd.Series, cutoff: int) -> pd.Series:
    """Return a copy with every value after `cutoff` wildly changed."""
    mutated = series.copy()
    mutated.iloc[cutoff + 1 :] = mutated.iloc[cutoff + 1 :] + 10_000.0
    return mutated


def _assert_prefix_unchanged(before: pd.Series, after: pd.Series, cutoff: int, label: str) -> None:
    prefix_before = before.iloc[: cutoff + 1]
    prefix_after = after.iloc[: cutoff + 1]
    mismatches = ~(
        (prefix_before == prefix_after) | (prefix_before.isna() & prefix_after.isna())
    )
    assert not mismatches.any(), (
        f"{label}: {mismatches.sum()} value(s) up to index {cutoff} changed after "
        f"mutating only FUTURE data — this indicator is using look-ahead information."
    )


def test_sma_has_no_lookahead():
    before = trend.sma(_close, 20)
    after = trend.sma(_mutate_future(_close, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "sma")


def test_ema_has_no_lookahead():
    before = trend.ema(_close, 20)
    after = trend.ema(_mutate_future(_close, CUTOFF), 20)
    _assert_prefix_unchanged(before, after, CUTOFF, "ema")


def test_macd_has_no_lookahead():
    before = trend.macd(_close)
    after = trend.macd(_mutate_future(_close, CUTOFF))
    for col in before.columns:
        _assert_prefix_unchanged(before[col], after[col], CUTOFF, f"macd.{col}")


def test_adx_has_no_lookahead():
    before = trend.adx(_high, _low, _close)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    after = trend.adx(mutated_high, mutated_low, mutated_close)
    for col in before.columns:
        _assert_prefix_unchanged(before[col], after[col], CUTOFF, f"adx.{col}")


def test_rsi_has_no_lookahead():
    before = momentum.rsi(_close)
    after = momentum.rsi(_mutate_future(_close, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "rsi")


def test_roc_has_no_lookahead():
    before = momentum.roc(_close)
    after = momentum.roc(_mutate_future(_close, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "roc")


def test_momentum_has_no_lookahead():
    before = momentum.momentum(_close)
    after = momentum.momentum(_mutate_future(_close, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "momentum")


def test_stochastic_has_no_lookahead():
    before = momentum.stochastic(_high, _low, _close)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    after = momentum.stochastic(mutated_high, mutated_low, mutated_close)
    for col in before.columns:
        _assert_prefix_unchanged(before[col], after[col], CUTOFF, f"stochastic.{col}")


def test_atr_has_no_lookahead():
    before = volatility.atr(_high, _low, _close)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    after = volatility.atr(mutated_high, mutated_low, mutated_close)
    _assert_prefix_unchanged(before, after, CUTOFF, "atr")


def test_bollinger_bands_has_no_lookahead():
    before = volatility.bollinger_bands(_close)
    after = volatility.bollinger_bands(_mutate_future(_close, CUTOFF))
    for col in before.columns:
        _assert_prefix_unchanged(before[col], after[col], CUTOFF, f"bollinger.{col}")


def test_rolling_volatility_has_no_lookahead():
    before = volatility.rolling_volatility(_close)
    after = volatility.rolling_volatility(_mutate_future(_close, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "rolling_volatility")


def test_obv_has_no_lookahead():
    before = vol_ind.obv(_close, _volume)
    after = vol_ind.obv(_mutate_future(_close, CUTOFF), _mutate_future(_volume, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "obv")


def test_volume_moving_average_has_no_lookahead():
    before = vol_ind.volume_moving_average(_volume)
    after = vol_ind.volume_moving_average(_mutate_future(_volume, CUTOFF))
    _assert_prefix_unchanged(before, after, CUTOFF, "volume_moving_average")


def test_vwap_has_no_lookahead():
    before = vol_ind.vwap(_high, _low, _close, _volume)
    mutated_high = _mutate_future(_high, CUTOFF)
    mutated_low = _mutate_future(_low, CUTOFF)
    mutated_close = _mutate_future(_close, CUTOFF)
    mutated_volume = _mutate_future(_volume, CUTOFF)
    after = vol_ind.vwap(mutated_high, mutated_low, mutated_close, mutated_volume)
    _assert_prefix_unchanged(before, after, CUTOFF, "vwap")


def test_price_distance_from_vwap_has_no_lookahead():
    vwap_series = vol_ind.vwap(_high, _low, _close, _volume)
    before = vol_ind.price_distance_from_vwap(_close, vwap_series)

    mutated_close = _mutate_future(_close, CUTOFF)
    mutated_vwap = vol_ind.vwap(_high, _low, mutated_close, _volume)
    after = vol_ind.price_distance_from_vwap(mutated_close, mutated_vwap)
    _assert_prefix_unchanged(before, after, CUTOFF, "price_distance_from_vwap")
