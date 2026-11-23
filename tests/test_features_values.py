"""Correctness tests for feature values against hand-calculated references."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import market_regime, stock_features


def test_index_trend_up_down_sideways():
    # Construct a series that's clearly trending up at the end.
    close = pd.Series(list(range(1, 60)), dtype=float)  # strictly increasing
    result = market_regime.index_trend(close, 5, 10)
    valid = result.dropna()
    assert (valid == 1.0).all()  # strictly increasing -> always "up" once warmed up


def test_index_trend_down_for_strictly_decreasing():
    close = pd.Series(list(range(100, 40, -1)), dtype=float)
    result = market_regime.index_trend(close, 5, 10)
    valid = result.dropna()
    assert (valid == -1.0).all()


def test_position_vs_sma_matches_hand_calculation():
    close = pd.Series([10.0, 10.0, 10.0, 20.0])
    result = market_regime.position_vs_sma(close, 3)
    # sma at index 2 = mean(10,10,10) = 10 -> (10-10)/10*100 = 0
    assert result.iloc[2] == 0.0
    # sma at index 3 = mean(10,10,20) = 13.333 -> (20-13.333)/13.333*100
    expected = (20.0 - (40 / 3)) / (40 / 3) * 100
    assert np.isclose(result.iloc[3], expected)


def test_log_return_matches_hand_calculation():
    close = pd.Series([100.0, 110.0, 121.0])
    result = stock_features.log_return(close, 1)
    assert np.isclose(result.iloc[1], np.log(110 / 100))
    assert np.isclose(result.iloc[2], np.log(121 / 110))


def test_volume_change_matches_hand_calculation():
    volume = pd.Series([100.0, 150.0, 75.0])
    result = stock_features.volume_change(volume, 1)
    assert result.iloc[1] == 50.0  # (150-100)/100*100
    assert np.isclose(result.iloc[2], (75 - 150) / 150 * 100)


def test_high_low_breakout_detects_new_high():
    # Flat prices for the window, then a clear breakout above.
    high = pd.Series([10.0] * 5 + [20.0])
    low = pd.Series([9.0] * 5 + [19.0])
    close = pd.Series([9.5] * 5 + [20.0])
    result = stock_features.high_low_breakout(high, low, close, 5)
    assert result.iloc[5] == 1.0  # close 20 > prior 5-bar high of 10


def test_high_low_breakout_detects_new_low():
    high = pd.Series([10.0] * 5 + [1.0])
    low = pd.Series([9.0] * 5 + [0.5])
    close = pd.Series([9.5] * 5 + [0.5])
    result = stock_features.high_low_breakout(high, low, close, 5)
    assert result.iloc[5] == -1.0


def test_high_low_breakout_excludes_current_bar_from_threshold():
    # If the current bar's own high were included in the threshold, a
    # breakout would be nearly impossible to detect since close <= high
    # always. Confirm using only PRIOR bars lets a genuine breakout fire.
    high = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 15.0])
    low = pd.Series([9.0] * 6)
    close = pd.Series([9.5, 9.5, 9.5, 9.5, 9.5, 15.0])
    result = stock_features.high_low_breakout(high, low, close, 5)
    assert result.iloc[5] == 1.0


def test_intraday_range_matches_hand_calculation():
    high = pd.Series([105.0])
    low = pd.Series([95.0])
    close = pd.Series([100.0])
    result = stock_features.intraday_range(high, low, close)
    assert result.iloc[0] == 10.0  # (105-95)/100*100


def test_gap_matches_hand_calculation():
    open_ = pd.Series([110.0])
    previous_close = pd.Series([100.0])
    result = stock_features.gap(open_, previous_close)
    assert result.iloc[0] == 10.0


def test_relative_volume_above_and_below_one():
    volume = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0])
    result = stock_features.relative_volume(volume, 4)
    # average of first 4 = 100, current (5th) = 200 -> ratio 2.0
    assert result.iloc[4] == 2.0


def test_relative_performance_positive_when_outperforming():
    asset_return = pd.Series([5.0, 3.0, -1.0])
    benchmark_return = pd.Series([2.0, 3.0, 1.0])
    result = stock_features.relative_performance(asset_return, benchmark_return)
    assert result.iloc[0] == 3.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == -2.0
