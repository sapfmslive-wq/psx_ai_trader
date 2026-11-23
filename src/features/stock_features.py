"""Per-stock features not already covered by the Phase 11 indicators
(Section 20). RSI/MACD/ATR/VWAP-distance already exist as indicators and are
deliberately not recomputed here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import trend


def log_return(close: pd.Series, period: int) -> pd.Series:
    """Natural-log return ln(P_t / P_{t-period}). Raw log ratio, not
    multiplied by 100."""
    return np.log(close / close.shift(period))


def volume_change(volume: pd.Series, period: int = 1) -> pd.Series:
    """Percentage change in volume vs `period` bars ago."""
    return volume.pct_change(period) * 100


def volume_acceleration(volume: pd.Series, period: int = 1) -> pd.Series:
    """Change in volume_change — how fast volume growth itself is changing."""
    return volume_change(volume, period).diff()


def ma_distance(close: pd.Series, period: int) -> pd.Series:
    """Percentage distance of close from its SMA (positive = above)."""
    sma_series = trend.sma(close, period)
    return (close - sma_series) / sma_series * 100


def high_low_breakout(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """+1 if close is above the highest high of the PRIOR `period` bars,
    -1 if below the lowest low of the prior `period` bars, else 0.

    The threshold deliberately uses the prior bars only (shifted by one),
    excluding the current bar: including it would make "close > rolling max
    high" nearly meaningless, since the current bar's own high is part of
    that max. NaN until `period` prior bars exist.
    """
    prior_high = high.rolling(window=period, min_periods=period).max().shift(1)
    prior_low = low.rolling(window=period, min_periods=period).min().shift(1)

    score = pd.Series(0.0, index=close.index)
    score[close > prior_high] = 1.0
    score[close < prior_low] = -1.0
    score[prior_high.isna() | prior_low.isna()] = float("nan")
    return score


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(high - low) / close, as a percentage — the bar's own range."""
    return (high - low) / close * 100


def gap(open_: pd.Series, previous_close: pd.Series) -> pd.Series:
    """Percentage gap between this bar's open and the previous bar's close."""
    return (open_ - previous_close) / previous_close * 100


def relative_volume(volume: pd.Series, period: int) -> pd.Series:
    """Current volume divided by the average of the PRIOR `period` bars
    (current bar excluded from its own baseline — same reasoning as
    high_low_breakout: including the current bar in its own average dilutes
    exactly the spike the feature is meant to detect)."""
    average = volume.rolling(window=period, min_periods=period).mean().shift(1)
    return volume / average


def relative_performance(asset_return: pd.Series, benchmark_return: pd.Series) -> pd.Series:
    """Asset return minus benchmark return — positive means the asset
    outperformed the benchmark over that horizon."""
    return asset_return - benchmark_return
