"""Volatility indicators: ATR, Bollinger Bands, rolling volatility."""

from __future__ import annotations

import pandas as pd

from src.indicators.trend import _true_range, sma


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns a DataFrame with columns: middle, upper, lower."""
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def rolling_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation of simple returns — a basic realized-
    volatility measure, not annualized (the caller decides how to scale it
    for their timeframe)."""
    returns = close.pct_change()
    return returns.rolling(window=period, min_periods=period).std()
