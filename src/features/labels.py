"""Prediction labels / targets (Phase 13, Section 22).

**Labels deliberately use future data** — a label at time T is "what
happened after T". That is the one place in this project where looking
forward is correct, and it is exactly why labels never live alongside
features: they are stored in the separate `labels` table, and nothing in
`src/features` or `src/indicators` can read them.

Every function here aligns the label to the bar it is *about* (the bar at
which a prediction would be made), computing it from bars strictly after
that one. The last `horizon` bars of any series therefore have no label —
the future they describe has not happened yet — and are NaN, never filled.

Targets (all horizons are in bars of the series' own timeframe):
    A. direction_label      1.0 UP / 0.0 DOWN  (NaN if the return is exactly 0)
    B. three_class_label    +1 UP / 0 NEUTRAL / -1 DOWN (NEUTRAL inside a band)
    C. future_return        percent return from this close to the close h bars on
    D. tp_before_sl_label   1.0 if take-profit is reached before stop-loss
                            within the next h bars, else 0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Percent return from close[t] to close[t + horizon] (Target C).
    NaN for the last `horizon` bars.

    NaN where close[t] <= 0: a percentage return divides by the starting
    price, so a zero/negative start silently FLIPS the sign of the move and
    would invert every label built from it. Real prices are positive (the
    quality engine excludes non-positive ticks), so this only ever fires on
    impossible input — where an undefined label is the honest answer."""
    future_close = close.shift(-horizon)
    result = (future_close - close) / close * 100
    result[close <= 0] = np.nan
    return result


def direction_label(future_ret: pd.Series) -> pd.Series:
    """Target A: 1.0 if the future return is positive, 0.0 if negative.

    A return of exactly 0 has no direction, so it is NaN (excluded from
    training) rather than being forced into UP or DOWN.
    """
    label = pd.Series(np.nan, index=future_ret.index)
    label[future_ret > 0] = 1.0
    label[future_ret < 0] = 0.0
    return label


def three_class_label(future_ret: pd.Series, neutral_threshold_pct: float) -> pd.Series:
    """Target B: +1 UP if return > threshold, -1 DOWN if return < -threshold,
    0 NEUTRAL in between (inclusive). NaN where the return is unknown."""
    label = pd.Series(0.0, index=future_ret.index)
    label[future_ret > neutral_threshold_pct] = 1.0
    label[future_ret < -neutral_threshold_pct] = -1.0
    label[future_ret.isna()] = np.nan
    return label


def tp_before_sl_label(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> pd.Series:
    """Target D: did price reach take-profit before stop-loss?

    Entry is close[t]. Over bars t+1 .. t+horizon, the take-profit barrier
    is entry * (1 + tp%) and the stop-loss barrier entry * (1 - sl%). The
    first bar to touch a barrier decides the label:

        1.0  take-profit touched first
        0.0  stop-loss touched first, OR neither touched within the horizon
             (the event "TP reached before SL" did not happen)

    If a single bar touches BOTH barriers, bar-level data cannot say which
    came first, so this takes the pessimistic reading: stop-loss first
    (0.0). Never optimistic.

    NaN for the last `horizon` bars (the full window is not yet known).
    """
    n = len(close)
    highs, lows, closes = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    result = np.full(n, np.nan)

    for t in range(n - horizon):
        entry = closes[t]
        tp = entry * (1 + take_profit_pct / 100)
        sl = entry * (1 - stop_loss_pct / 100)
        outcome = 0.0  # default: neither barrier reached
        for j in range(t + 1, t + horizon + 1):
            hit_tp = highs[j] >= tp
            hit_sl = lows[j] <= sl
            if hit_sl:  # includes the same-bar case: pessimistic
                outcome = 0.0
                break
            if hit_tp:
                outcome = 1.0
                break
        result[t] = outcome

    return pd.Series(result, index=close.index)
