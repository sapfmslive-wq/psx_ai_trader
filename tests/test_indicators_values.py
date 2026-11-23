"""Correctness tests for indicator values — real pandas computation checked
against hand-calculable or known-reference values, not just "it ran"."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import momentum, trend, volatility, volume as vol_ind


def test_sma_matches_hand_calculation():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = trend.sma(close, 3)
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == 2.0  # mean(1,2,3)
    assert result.iloc[3] == 3.0  # mean(2,3,4)
    assert result.iloc[4] == 4.0  # mean(3,4,5)


def test_ema_first_valid_value_equals_sma_seed():
    # With adjust=False and min_periods=period, pandas' first non-NaN EMA
    # value for a constant series should equal that constant.
    close = pd.Series([10.0] * 30)
    result = trend.ema(close, 10)
    assert result.dropna().iloc[0] == 10.0


def test_rsi_is_100_for_strictly_increasing_series():
    close = pd.Series(range(1, 30), dtype=float)  # always up, no losses at all
    result = momentum.rsi(close, 14)
    # avg_loss is 0 throughout -> RS is inf -> RSI should be 100
    valid = result.dropna()
    assert (valid == 100).all()


def test_rsi_is_0_for_strictly_decreasing_series():
    close = pd.Series(range(30, 1, -1), dtype=float)
    result = momentum.rsi(close, 14)
    valid = result.dropna()
    assert (valid == 0).all()


def test_roc_matches_hand_calculation():
    close = pd.Series([100.0, 105.0, 110.0, 90.0])
    result = momentum.roc(close, 1)
    assert result.iloc[1] == 5.0  # (105-100)/100*100
    assert np.isclose(result.iloc[2], (110 - 105) / 105 * 100)
    assert np.isclose(result.iloc[3], (90 - 110) / 110 * 100)


def test_momentum_matches_hand_calculation():
    close = pd.Series([10.0, 12.0, 15.0, 11.0])
    result = momentum.momentum(close, 1)
    assert result.iloc[1] == 2.0
    assert result.iloc[2] == 3.0
    assert result.iloc[3] == -4.0


def test_bollinger_bands_middle_equals_sma():
    close = pd.Series(np.random.default_rng(1).normal(100, 5, 50))
    bb = volatility.bollinger_bands(close, period=20, num_std=2.0)
    expected_middle = trend.sma(close, 20)
    pd.testing.assert_series_equal(bb["middle"], expected_middle, check_names=False)


def test_bollinger_upper_lower_are_symmetric_around_middle():
    close = pd.Series(np.random.default_rng(2).normal(100, 5, 50))
    bb = volatility.bollinger_bands(close, period=20, num_std=2.0)
    valid = bb.dropna()
    upper_dist = valid["upper"] - valid["middle"]
    lower_dist = valid["middle"] - valid["lower"]
    assert np.allclose(upper_dist, lower_dist)


def test_atr_is_never_negative():
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
    high = close + rng.random(60) * 2
    low = close - rng.random(60) * 2
    result = volatility.atr(high, low, close, 14).dropna()
    assert (result >= 0).all()


def test_stochastic_is_bounded_0_to_100():
    rng = np.random.default_rng(4)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
    high = close + rng.random(60) * 2
    low = close - rng.random(60) * 2
    result = momentum.stochastic(high, low, close, 14, 3).dropna()
    assert (result["percent_k"] >= -1e-9).all() and (result["percent_k"] <= 100 + 1e-9).all()
    assert (result["percent_d"] >= -1e-9).all() and (result["percent_d"] <= 100 + 1e-9).all()


def test_obv_direction_matches_price_moves():
    close = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = pd.Series([100.0, 200.0, 150.0, 50.0, 300.0])
    result = vol_ind.obv(close, volume)
    # bar0: base 0 (no prior bar)
    # bar1: up -> +200 => 200
    # bar2: down -> -150 => 50
    # bar3: flat -> +0 => 50
    # bar4: up -> +300 => 350
    assert list(result) == [0.0, 200.0, 50.0, 50.0, 350.0]


def test_vwap_equals_single_bar_typical_price_on_first_bar():
    high = pd.Series([101.0, 102.0])
    low = pd.Series([99.0, 100.0])
    close = pd.Series([100.0, 101.0])
    volume = pd.Series([1000.0, 2000.0])
    result = vol_ind.vwap(high, low, close, volume)
    expected_first = (101.0 + 99.0 + 100.0) / 3
    assert np.isclose(result.iloc[0], expected_first)


def test_price_distance_from_vwap_zero_when_close_equals_vwap():
    close = pd.Series([100.0, 100.0])
    vwap_series = pd.Series([100.0, 100.0])
    result = vol_ind.price_distance_from_vwap(close, vwap_series)
    assert (result == 0.0).all()


def test_volume_moving_average_matches_hand_calculation():
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    result = vol_ind.volume_moving_average(volume, 2)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 150.0
    assert result.iloc[2] == 250.0
    assert result.iloc[3] == 350.0


def test_macd_histogram_equals_macd_minus_signal():
    close = pd.Series(np.random.default_rng(5).normal(100, 3, 60))
    result = trend.macd(close, 12, 26, 9)
    valid = result.dropna()
    assert np.allclose(valid["histogram"], valid["macd"] - valid["signal"])
