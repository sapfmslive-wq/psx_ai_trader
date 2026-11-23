"""Tests for src.data.validator (Phase 6). No database dependency."""

from __future__ import annotations

import datetime as dt

import pytest

from src.config.loader import load_config
from src.data.validator import (
    DUPLICATE,
    INVALID,
    SUSPECT,
    VALID,
    DataQualityEngine,
    QualityInput,
)

UTC = dt.timezone.utc


def _ts(hour: int, minute: int, second: int = 0, day: int = 14) -> dt.datetime:
    return dt.datetime(2026, 9, day, hour, minute, second, tzinfo=UTC)


def _qi(
    symbol="THCCL",
    timestamp=None,
    price=80.0,
    open=None,
    high=None,
    low=None,
    volume=None,
    market_status=None,
) -> QualityInput:
    return QualityInput(
        symbol=symbol,
        timestamp=timestamp,
        price=price,
        open=open,
        high=high,
        low=low,
        volume=volume,
        market_status=market_status,
    )


VALID_SETTINGS = """
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
  sma_periods: [5, 10, 20]
  ema_periods: [5, 10, 20]
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

VALID_SYMBOLS = """
indices:
  - KSE100
stocks:
  - THCCL
watchlist:
  - THCCL
"""


@pytest.fixture()
def config(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    symbols_path = tmp_path / "symbols.yaml"
    env_path = tmp_path / ".env"
    settings_path.write_text(VALID_SETTINGS, encoding="utf-8")
    symbols_path.write_text(VALID_SYMBOLS, encoding="utf-8")
    env_path.write_text("LIVE_TRADING_ENABLED=false\n", encoding="utf-8")
    return load_config(settings_path=settings_path, symbols_path=symbols_path, env_path=env_path)


def test_first_observation_for_a_symbol_is_valid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    assert result.status == VALID
    assert result.reasons == []


def test_unknown_symbol_is_invalid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(symbol="NOT_CONFIGURED", timestamp=_ts(9, 30), price=80.0))
    assert result.status == INVALID
    assert any("configured universe" in r for r in result.reasons)


def test_missing_timestamp_is_invalid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=None, price=80.0))
    assert result.status == INVALID
    assert any("missing timestamp" in r for r in result.reasons)


def test_non_positive_price_is_invalid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=-5.0))
    assert result.status == INVALID
    assert any("price" in r for r in result.reasons)

    result_zero = engine.evaluate(_qi(timestamp=_ts(9, 31), price=0))
    assert result_zero.status == INVALID


def test_high_below_low_is_invalid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0, high=79.0, low=80.5))
    assert result.status == INVALID
    assert any("high < low" in r for r in result.reasons)


def test_negative_volume_is_invalid(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0, volume=-1))
    assert result.status == INVALID
    assert any("negative volume" in r for r in result.reasons)


def test_far_future_timestamp_is_invalid(config):
    engine = DataQualityEngine(config)
    far_future = dt.datetime.now(UTC) + dt.timedelta(hours=1)
    result = engine.evaluate(_qi(timestamp=far_future, price=80.0))
    assert result.status == INVALID
    assert any("future" in r for r in result.reasons)


def test_duplicate_timestamp_is_flagged_duplicate(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    assert result.status == DUPLICATE


def test_out_of_order_timestamp_is_invalid(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 32), price=80.0))
    result = engine.evaluate(_qi(timestamp=_ts(9, 31), price=80.0))  # earlier than previous
    assert result.status == INVALID
    assert any("out-of-order" in r for r in result.reasons)


def test_extreme_change_is_suspect_not_invalid(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    # +25% in one tick, well beyond the configured 10% threshold.
    result = engine.evaluate(_qi(timestamp=_ts(9, 31), price=100.0))
    assert result.status == SUSPECT
    assert any("extreme change" in r for r in result.reasons)


def test_unrecognized_market_status_is_suspect(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0, market_status="WEIRD_STATUS"))
    assert result.status == SUSPECT
    assert any("unrecognized market_status" in r for r in result.reasons)


def test_known_market_status_is_not_flagged(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0, market_status="SYNTHETIC_OPEN"))
    assert result.status == VALID


def test_stale_price_becomes_suspect_after_threshold(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30, 0), price=80.0))
    # Same price, 301 seconds later (threshold is 300s) -> stale.
    result = engine.evaluate(_qi(timestamp=_ts(9, 35, 1), price=80.0))
    assert result.status == SUSPECT
    assert any("stale price" in r for r in result.reasons)


def test_repeated_price_within_stale_threshold_is_not_flagged(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30, 0), price=80.0))
    # Same price, only 60 seconds later -> not yet stale.
    result = engine.evaluate(_qi(timestamp=_ts(9, 31, 0), price=80.0))
    assert result.status == VALID


def test_invalid_observation_does_not_update_baseline(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    # This one is invalid (negative volume) — must not become the new baseline.
    engine.evaluate(_qi(timestamp=_ts(9, 31), price=999.0, volume=-1))
    # A legitimate small move from the ORIGINAL valid baseline (80.0) should
    # still be evaluated relative to 80.0, not the rejected 999.0.
    result = engine.evaluate(_qi(timestamp=_ts(9, 32), price=80.5))
    assert result.status == VALID


def test_duplicate_observation_does_not_update_baseline(config):
    engine = DataQualityEngine(config)
    engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))
    engine.evaluate(_qi(timestamp=_ts(9, 30), price=80.0))  # duplicate, ignored as baseline
    result = engine.evaluate(_qi(timestamp=_ts(9, 31), price=80.2))
    assert result.status == VALID


def test_index_and_stock_convenience_wrappers(config):
    from src.data.base_provider import IndexObservation, StockObservation

    engine = DataQualityEngine(config)

    index_obs = IndexObservation(
        provider="mock", symbol="KSE100", timestamp=_ts(9, 30), value=45000.0,
        change=None, change_percent=None, high=None, low=None, volume=None,
        market_status="SYNTHETIC_OPEN", source_timestamp=None, is_synthetic=True,
    )
    stock_obs = StockObservation(
        provider="mock", symbol="THCCL", timestamp=_ts(9, 30), last_price=80.0,
        open=80.0, high=80.0, low=80.0, previous_close=80.0, change=None,
        change_percent=None, volume=None, bid=None, bid_volume=None, ask=None,
        ask_volume=None, turnover=None, vwap=None, market_status="SYNTHETIC_OPEN",
        source_timestamp=None, is_synthetic=True,
    )

    assert engine.evaluate_index(index_obs).status == VALID
    assert engine.evaluate_stock(stock_obs).status == VALID


def test_reason_text_joins_multiple_reasons(config):
    engine = DataQualityEngine(config)
    result = engine.evaluate(_qi(timestamp=_ts(9, 30), price=-5.0, volume=-1))
    assert result.reason_text is not None
    assert ";" in result.reason_text or len(result.reasons) == 1
