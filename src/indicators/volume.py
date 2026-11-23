"""Volume indicators: OBV, VWAP, volume moving average, distance from VWAP.

VWAP here is a **running** volume-weighted average price computed over
whatever Series the caller passes in — it does not reset at any particular
boundary on its own. For a conventional daily VWAP, pass in only one
trading day's candles (Phase 10's `TradingCalendar.regular_session_windows`
gives you that day's bounds); Phase 12's feature builder is what will
actually call this per-day. Keeping this function boundary-agnostic keeps
it simple and composable rather than baking in an assumption about how the
caller wants to segment time.
"""

from __future__ import annotations

import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume, added when price rises,
    subtracted when price falls, unchanged when price is flat."""
    direction = close.diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    direction.iloc[0] = 0.0  # no prior price to compare the first bar to
    return (direction * volume).cumsum()


def volume_moving_average(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Running volume-weighted average price. See module docstring for the
    boundary/reset caveat."""
    typical_price = (high + low + close) / 3
    cumulative_volume = volume.cumsum()
    cumulative_turnover = (typical_price * volume).cumsum()
    return cumulative_turnover / cumulative_volume


def price_distance_from_vwap(close: pd.Series, vwap_series: pd.Series) -> pd.Series:
    """Percentage distance of close from the given VWAP series."""
    return (close - vwap_series) / vwap_series * 100
