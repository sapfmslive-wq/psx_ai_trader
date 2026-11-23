"""Momentum indicators: RSI, ROC, Momentum, Stochastic."""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change, as a percentage."""
    shifted = close.shift(period)
    return (close - shifted) / shifted * 100


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Simple price momentum: close[t] - close[t-period]."""
    return close - close.shift(period)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator. Returns a DataFrame with columns: percent_k, percent_d."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    percent_k = (close - lowest_low) / (highest_high - lowest_low) * 100
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"percent_k": percent_k, "percent_d": percent_d})
