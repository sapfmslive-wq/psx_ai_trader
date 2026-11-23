"""Tests for src.indicators.engine.compute_and_store_indicators (Phase 11).

Same fake-repository-module-injected-via-sys.modules technique as the
Phase 10 candle engine DB tests, so this runs fully offline without
SQLAlchemy installed — only pandas/numpy (already required for the pure
indicator math) are needed.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from dataclasses import dataclass

import pytest

from src.config.loader import load_config

SETTINGS = """
market:
  timezone: Asia/Karachi
data:
  provider: mock
collector:
  reconnect_seconds: 5
  stale_timeout_seconds: 30
  poll_interval_seconds: 15
  max_consecutive_failures: 10
prediction:
  horizons: ["5m"]
risk:
  max_risk_per_trade: 0.01
  max_positions: 5
trading:
  mode: RESEARCH
dashboard:
  host: 127.0.0.1
  port: 8000
  allow_external: false
logging:
  level: INFO
database:
  path: database/psx.db
  echo: false
quality:
  max_change_percent: 10.0
  stale_seconds: 300
  future_timestamp_tolerance_seconds: 5
candles:
  volume_convention: cumulative
indicators:
  sma_periods: [5, 10]
  ema_periods: [5, 10]
  rsi_period: 14
  macd_fast_period: 12
  macd_slow_period: 26
  macd_signal_period: 9
  atr_period: 14
  adx_period: 14
  bollinger_period: 20
  bollinger_num_std: 2.0
  roc_period: 12
  momentum_period: 10
  stochastic_k_period: 14
  stochastic_d_period: 3
  volume_ma_period: 20
features:
  return_horizons: [1, 5, 15, 60]
  high_low_breakout_period: 20
  trend_fast_sma_period: 20
  trend_slow_sma_period: 50
  volatility_period: 20
labels:
  horizons: [5, 15, 30]
  neutral_threshold_pct: 0.1
  take_profit_pct: 1.0
  stop_loss_pct: 0.5
baselines:
  ma_fast_period: 10
  ma_slow_period: 30
  rsi_oversold: 30
  rsi_overbought: 70
  train_fraction: 0.7
  random_seed: 42
models:
  train_fraction: 0.6
  validation_fraction: 0.2
  random_seed: 42
  enabled: [logistic, random_forest, gradient_boosting, hist_gradient_boosting]
  min_rows: 300
  permutation_repeats: 5
backtest:
  initial_cash: 1000000
  position_fraction: 1.0
  execution_delay_bars: 1
  slippage_bps: 10.0
  max_participation_pct: 10.0
  stop_loss_pct: 0.0
  take_profit_pct: 0.0
  min_price: 0.0
  min_avg_volume: 0.0
  avg_volume_bars: 20
  risk_free_rate_annual_pct: 0.0
  liquidate_at_end: true
  min_trades_for_confidence: 30
"""

SYMBOLS = """
indices:
  - KSE100
stocks:
  - THCCL
watchlist:
  - THCCL
"""


@dataclass
class _FakeCandle:
    open_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def config(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    symbols_path = tmp_path / "symbols.yaml"
    env_path = tmp_path / ".env"
    settings_path.write_text(SETTINGS, encoding="utf-8")
    symbols_path.write_text(SYMBOLS, encoding="utf-8")
    env_path.write_text("LIVE_TRADING_ENABLED=false\n", encoding="utf-8")
    return load_config(settings_path=settings_path, symbols_path=symbols_path, env_path=env_path)


def _session_factory():
    return _FakeSession()


def _install_fake_repository(monkeypatch, candles):
    inserted = []
    fake_module = types.ModuleType("src.database.repository")

    def get_candles(session, symbol, timeframe, start, end):
        return candles

    def insert_feature(session, **kwargs):
        key = (kwargs["symbol"], kwargs["timeframe"], kwargs["timestamp"], kwargs["feature_name"], kwargs["feature_version"])
        if any(
            (c["symbol"], c["timeframe"], c["timestamp"], c["feature_name"], c["feature_version"]) == key
            for c in inserted
        ):
            return False
        inserted.append(kwargs)
        return True

    fake_module.get_candles = get_candles
    fake_module.insert_feature = insert_feature
    monkeypatch.setitem(sys.modules, "src.database.repository", fake_module)
    return inserted


def _make_candles(n=40):
    import numpy as np

    rng = np.random.default_rng(0)
    base = dt.datetime(2026, 9, 21, 9, 32, tzinfo=dt.timezone.utc)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return [
        _FakeCandle(
            open_time=base + dt.timedelta(minutes=i),
            open=closes[i] - 0.1, high=closes[i] + 0.3, low=closes[i] - 0.3,
            close=closes[i], volume=float(rng.integers(100, 1000)),
        )
        for i in range(n)
    ]


def test_compute_and_store_writes_features_for_every_indicator(config, monkeypatch):
    from src.indicators.engine import compute_and_store_indicators

    candles = _make_candles(40)
    inserted = _install_fake_repository(monkeypatch, candles)

    start = candles[0].open_time
    end = candles[-1].open_time + dt.timedelta(minutes=1)
    result = compute_and_store_indicators(config, _session_factory, "THCCL", "1m", start, end)

    assert result["written"] > 0
    feature_names = {c["feature_name"] for c in inserted}
    assert "sma_5" in feature_names
    assert "rsi_14" in feature_names
    assert "macd_line" in feature_names
    assert "obv" in feature_names
    assert "vwap" in feature_names

    for c in inserted:
        assert c["feature_version"] == "indicators_v1"
        assert c["symbol"] == "THCCL"
        assert c["timeframe"] == "1m"


def test_period_based_features_are_not_stored_before_warmup(config, monkeypatch):
    from src.indicators.engine import compute_and_store_indicators

    # Only 3 candles — far fewer than any rolling-window period (min is 5
    # for sma_5/ema_5). Period-based indicators must not appear at all.
    # OBV and VWAP are cumulative from the first bar and legitimately DO
    # produce values even here — that's correct, not a leak, since they
    # have no fixed warm-up window by definition.
    candles = _make_candles(3)
    inserted = _install_fake_repository(monkeypatch, candles)

    start = candles[0].open_time
    end = candles[-1].open_time + dt.timedelta(minutes=1)
    compute_and_store_indicators(config, _session_factory, "THCCL", "1m", start, end)

    feature_names = {c["feature_name"] for c in inserted}
    assert "sma_5" not in feature_names
    assert "rsi_14" not in feature_names
    assert "atr_14" not in feature_names
    # Cumulative-from-bar-one indicators are legitimately present:
    assert "obv" in feature_names
    assert "vwap" in feature_names


def test_rebuilding_is_idempotent(config, monkeypatch):
    from src.indicators.engine import compute_and_store_indicators

    candles = _make_candles(40)
    inserted = _install_fake_repository(monkeypatch, candles)

    start = candles[0].open_time
    end = candles[-1].open_time + dt.timedelta(minutes=1)
    first = compute_and_store_indicators(config, _session_factory, "THCCL", "1m", start, end)
    second = compute_and_store_indicators(config, _session_factory, "THCCL", "1m", start, end)

    assert first["written"] > 0
    assert second["written"] == 0
    assert second["duplicate"] == first["written"]


def test_empty_candle_list_writes_nothing(config, monkeypatch):
    from src.indicators.engine import compute_and_store_indicators

    inserted = _install_fake_repository(monkeypatch, [])
    result = compute_and_store_indicators(
        config, _session_factory, "THCCL", "1m",
        dt.datetime(2026, 9, 21, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 9, 22, tzinfo=dt.timezone.utc),
    )
    assert result == {"written": 0, "duplicate": 0}
    assert inserted == []
