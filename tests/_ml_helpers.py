"""Shared helpers for the Phase 15 ML tests (not collected as tests)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def make_config(**overrides):
    """A lightweight stand-in for AppConfig carrying only what the ML code reads."""
    cfg = SimpleNamespace(
        indicators=SimpleNamespace(
            sma_periods=(5, 10), ema_periods=(5, 10), rsi_period=14,
            macd_fast_period=12, macd_slow_period=26, macd_signal_period=9,
            atr_period=14, adx_period=14, bollinger_period=20, bollinger_num_std=2.0,
            roc_period=12, momentum_period=10, stochastic_k_period=14,
            stochastic_d_period=3, volume_ma_period=20,
        ),
        features=SimpleNamespace(
            return_horizons=(1, 5), high_low_breakout_period=10,
            trend_fast_sma_period=5, trend_slow_sma_period=10, volatility_period=10,
        ),
        baselines=SimpleNamespace(
            ma_fast_period=5, ma_slow_period=20, rsi_oversold=30.0, rsi_overbought=70.0,
            train_fraction=0.7, random_seed=42,
        ),
        models=SimpleNamespace(
            train_fraction=0.6, validation_fraction=0.2, random_seed=42,
            enabled=("logistic", "random_forest", "gradient_boosting", "hist_gradient_boosting"),
            min_rows=100, permutation_repeats=3,
        ),
        symbols=SimpleNamespace(indices=("KSE100", "KMI30"), stocks=("THCCL",)),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def frame(close, volume_seed=0, scale=1.0, start="2026-09-21 09:32", with_volume=True) -> pd.DataFrame:
    close = pd.Series(np.asarray(close, dtype=float)) * scale
    idx = pd.date_range(start, periods=len(close), freq="1min", tz="UTC")
    rng = np.random.default_rng(volume_seed)
    volume = rng.integers(100, 1000, len(close)).astype(float) if with_volume else np.nan
    return pd.DataFrame(
        {"open": close.values, "high": close.values * 1.001, "low": close.values * 0.999,
         "close": close.values, "volume": volume},
        index=idx,
    )


def random_walk(n, seed=1, start=100.0, step=0.3):
    return start + np.cumsum(np.random.default_rng(seed).normal(0, step, n))


def regime_walk(n, seed, block=80, phi=0.7, hi_sd=0.6, lo_sd=0.15, start=100.0):
    """Momentum in high-volatility blocks, mean-reversion in low-volatility
    blocks. "The last move continues" therefore averages out to ~50% — no
    simple baseline can see the structure — but a model that can combine the
    last return with the volatility regime can. Log-price, so always positive."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.integers(0, 2, n // block + 1), block)[:n]  # 1 = high-vol momentum
    r = np.zeros(n)
    for t in range(1, n):
        sign, sd = (1.0, hi_sd) if regime[t] else (-1.0, lo_sd)
        r[t] = sign * phi * r[t - 1] + rng.normal(0, sd)
    return start * np.exp(np.cumsum(r) / 100.0)


def ar_walk(n, seed=1, phi=0.85, step=0.3, start=100.0):
    """Price whose log-RETURNS are autocorrelated — a genuine, learnable signal.

    Built on log-price (start * exp(cumulative return %)), so it can never go
    non-positive. (An earlier additive version drifted below zero, which made
    percentage labels silently invert and cost a debugging session.)"""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, step)
    return start * np.exp(np.cumsum(r) / 100.0)
