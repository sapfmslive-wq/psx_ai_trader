"""Tests for src.features.label_engine (Phase 13), via a fake repository
module injected into sys.modules (runs without SQLAlchemy).

The fake repository intentionally exposes NO insert_feature: if the label
engine ever tried to write a label into the features table, these tests
would fail with an AttributeError — enforcing the features/labels split.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from dataclasses import dataclass

import numpy as np
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
  return_horizons: [1, 5]
  high_low_breakout_period: 10
  trend_fast_sma_period: 5
  trend_slow_sma_period: 10
  volatility_period: 10
labels:
  horizons: [3, 5]
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
    (tmp_path / "settings.yaml").write_text(SETTINGS, encoding="utf-8")
    (tmp_path / "symbols.yaml").write_text(SYMBOLS, encoding="utf-8")
    (tmp_path / ".env").write_text("LIVE_TRADING_ENABLED=false\n", encoding="utf-8")
    return load_config(
        settings_path=tmp_path / "settings.yaml",
        symbols_path=tmp_path / "symbols.yaml",
        env_path=tmp_path / ".env",
    )


def _session_factory():
    return _FakeSession()


def _candles(n, seed=1):
    rng = np.random.default_rng(seed)
    base = dt.datetime(2026, 9, 21, 9, 32, tzinfo=dt.timezone.utc)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return [
        _FakeCandle(base + dt.timedelta(minutes=i), closes[i] - 0.1, closes[i] + 0.3,
                    closes[i] - 0.3, closes[i], 100.0)
        for i in range(n)
    ]


def _install(monkeypatch, candles):
    inserted = []
    fake = types.ModuleType("src.database.repository")
    fake.get_candles = lambda session, symbol, timeframe, start, end: candles

    def insert_label(session, **kw):
        key = (kw["symbol"], kw["timeframe"], kw["timestamp"], kw["label_name"], kw["label_version"])
        if any((c["symbol"], c["timeframe"], c["timestamp"], c["label_name"], c["label_version"]) == key
               for c in inserted):
            return False
        inserted.append(kw)
        return True

    fake.insert_label = insert_label  # NOTE: no insert_feature on purpose
    monkeypatch.setitem(sys.modules, "src.database.repository", fake)
    return inserted


def _run(config, candles):
    from src.features.label_engine import compute_and_store_labels

    start = candles[0].open_time
    end = candles[-1].open_time + dt.timedelta(minutes=1)
    return compute_and_store_labels(config, _session_factory, "THCCL", "1m", start, end)


def test_all_four_targets_are_written_for_every_horizon(config, monkeypatch):
    candles = _candles(40)
    inserted = _install(monkeypatch, candles)
    result = _run(config, candles)

    assert result["written"] > 0
    names = {c["label_name"] for c in inserted}
    for h in (3, 5):
        assert f"label_future_return_{h}bar" in names
        assert f"label_direction_{h}bar" in names
        assert f"label_3class_{h}bar" in names
        assert f"label_tp_before_sl_{h}bar" in names
    assert all(c["label_version"] == "labels_v1" for c in inserted)


def test_final_horizon_bars_get_no_label(config, monkeypatch):
    candles = _candles(40)
    inserted = _install(monkeypatch, candles)
    _run(config, candles)

    last_timestamps = {c.open_time for c in candles[-5:]}
    five_bar = [c for c in inserted if c["label_name"] == "label_future_return_5bar"]
    assert five_bar, "expected some 5-bar labels"
    assert not any(c["timestamp"] in last_timestamps for c in five_bar)


def test_labels_never_touch_the_features_table(config, monkeypatch):
    # The fake repository has no insert_feature; reaching it would raise.
    candles = _candles(40)
    _install(monkeypatch, candles)
    _run(config, candles)  # must simply succeed


def test_rebuilding_is_idempotent(config, monkeypatch):
    candles = _candles(40)
    _install(monkeypatch, candles)
    first = _run(config, candles)
    second = _run(config, candles)
    assert first["written"] > 0
    assert second["written"] == 0
    assert second["duplicate"] == first["written"]


def test_empty_candles_write_nothing(config, monkeypatch):
    inserted = _install(monkeypatch, [])
    from src.features.label_engine import compute_and_store_labels

    result = compute_and_store_labels(
        config, _session_factory, "THCCL", "1m",
        dt.datetime(2026, 9, 21, tzinfo=dt.timezone.utc), dt.datetime(2026, 9, 22, tzinfo=dt.timezone.utc),
    )
    assert result == {"written": 0, "duplicate": 0}
    assert inserted == []
