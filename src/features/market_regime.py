"""Market-regime features from an index's candles (Section 19)."""

from __future__ import annotations

import pandas as pd

from src.indicators import momentum, trend, volatility


def index_returns(close: pd.Series, horizons: tuple[int, ...]) -> dict[str, pd.Series]:
    """Percentage return over each horizon (in bars of whatever timeframe
    `close` is at). Reuses the ROC indicator — a horizon-N return IS
    N-bar rate of change."""
    return {f"return_{h}bar": momentum.roc(close, h) for h in horizons}


def index_trend(close: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
    """Trend score: +1 if close > fast SMA > slow SMA, -1 if close < fast
    SMA < slow SMA, 0 otherwise. NaN until both SMAs have a full window
    (never a fabricated 0 during warm-up).

    Encoded numerically because the Feature table stores floats; +1/0/-1 is
    a standard, documented categorical-to-numeric encoding for ML features.
    """
    fast = trend.sma(close, fast_period)
    slow = trend.sma(close, slow_period)

    score = pd.Series(0.0, index=close.index)
    score[(close > fast) & (fast > slow)] = 1.0
    score[(close < fast) & (fast < slow)] = -1.0
    score[fast.isna() | slow.isna()] = float("nan")
    return score


def position_vs_sma(close: pd.Series, period: int) -> pd.Series:
    """Percentage distance of close from its SMA (positive = above)."""
    sma_series = trend.sma(close, period)
    return (close - sma_series) / sma_series * 100


def index_volatility(close: pd.Series, period: int) -> pd.Series:
    return volatility.rolling_volatility(close, period)
