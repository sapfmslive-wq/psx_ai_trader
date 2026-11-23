"""Tests for src.data.candle_engine.build_candles_for_day (Phase 10).

Uses the same fake-repository-module-injected-via-sys.modules technique as
test_collector.py / test_live_collector.py, so this runs fully offline
without SQLAlchemy installed. build_candles_for_day() imports
src.database.repository lazily inside the function body, which is what
makes this possible.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from zoneinfo import ZoneInfo

import pytest

from src.config.loader import load_config
from src.market.trading_calendar import TradingCalendar, load_market_calendar

TZ = ZoneInfo("Asia/Karachi")

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

SYMBOLS = """
indices:
  - KSE100
stocks:
  - THCCL
watchlist:
  - THCCL
"""

CALENDAR = """
source_urls: []
fetched_date: "2026-09-18"
weekday_sessions:
  monday_to_thursday:
    - regular: {start: "09:32", end: "15:30"}
  friday:
    - regular: {start: "09:17", end: "12:00"}
    - regular: {start: "14:32", end: "16:30"}
  saturday: []
  sunday: []
holidays: []
special_sessions: []
"""


class _FakeIndexTickRow:
    def __init__(self, timestamp, value, volume, provider="mock"):
        self.timestamp = timestamp
        self.value = value
        self.volume = volume
        self.provider = provider


class _FakeStockTickRow:
    def __init__(self, timestamp, last_price, volume, provider="mock"):
        self.timestamp = timestamp
        self.last_price = last_price
        self.volume = volume
        self.provider = provider


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


@pytest.fixture()
def calendar(tmp_path):
    calendar_path = tmp_path / "market_calendar.yaml"
    calendar_path.write_text(CALENDAR, encoding="utf-8")
    return TradingCalendar(load_market_calendar(calendar_path), "Asia/Karachi")


def _session_factory():
    return _FakeSession()


def _install_fake_repository(monkeypatch, index_ticks=None, stock_ticks=None):
    inserted_candles = []

    fake_module = types.ModuleType("src.database.repository")

    def get_index_ticks(session, symbol, start, end):
        return index_ticks or []

    def get_stock_ticks(session, symbol, start, end):
        return stock_ticks or []

    def insert_candle(session, **kwargs):
        key = (kwargs["symbol"], kwargs["timeframe"], kwargs["open_time"])
        if any(c["symbol"] == key[0] and c["timeframe"] == key[1] and c["open_time"] == key[2] for c in inserted_candles):
            return False
        inserted_candles.append(kwargs)
        return True

    fake_module.get_index_ticks = get_index_ticks
    fake_module.get_stock_ticks = get_stock_ticks
    fake_module.insert_candle = insert_candle
    monkeypatch.setitem(sys.modules, "src.database.repository", fake_module)
    return inserted_candles


def test_build_candles_for_stock_writes_all_timeframes(config, calendar, monkeypatch):
    from src.data.candle_engine import build_candles_for_day

    day = dt.date(2026, 9, 21)  # a Monday
    ticks = [
        _FakeStockTickRow(dt.datetime(2026, 9, 21, 9, 32, tzinfo=TZ), 80.0, 1000),
        _FakeStockTickRow(dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ), 81.0, 2000),
        _FakeStockTickRow(dt.datetime(2026, 9, 21, 15, 0, tzinfo=TZ), 79.5, 3000),
    ]
    inserted = _install_fake_repository(monkeypatch, stock_ticks=ticks)

    report = build_candles_for_day(config, _session_factory, calendar, "THCCL", "stock", day)

    assert report.ticks_used == 3
    timeframes_written = {c["timeframe"] for c in inserted}
    assert timeframes_written == {"1m", "5m", "15m", "30m", "60m", "daily"}

    daily = next(c for c in inserted if c["timeframe"] == "daily")
    assert daily["open"] == 80.0
    assert daily["close"] == 79.5
    assert daily["high"] == 81.0
    assert daily["low"] == 79.5
    assert daily["volume"] == 3000  # cumulative, baseline 0


def test_build_candles_excludes_pre_open_ticks(config, monkeypatch, tmp_path):
    from src.data.candle_engine import build_candles_for_day

    calendar_path = tmp_path / "market_calendar.yaml"
    calendar_path.write_text(
        CALENDAR.replace(
            'regular: {start: "09:32", end: "15:30"}',
            'pre_open: {start: "09:15", end: "09:30"}\n'
            '      regular: {start: "09:32", end: "15:30"}',
        ),
        encoding="utf-8",
    )
    calendar = TradingCalendar(load_market_calendar(calendar_path), "Asia/Karachi")

    day = dt.date(2026, 9, 21)
    ticks = [
        _FakeStockTickRow(dt.datetime(2026, 9, 21, 9, 20, tzinfo=TZ), 999.0, 1),  # pre-open, must be excluded
        _FakeStockTickRow(dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ), 80.0, 1000),
    ]
    inserted = _install_fake_repository(monkeypatch, stock_ticks=ticks)

    report = build_candles_for_day(config, _session_factory, calendar, "THCCL", "stock", day, timeframes=("daily",))

    assert report.ticks_used == 1
    daily = inserted[0]
    assert daily["open"] == 80.0  # the pre-open 999.0 never made it in


def test_build_candles_for_index_uses_value_field(config, calendar, monkeypatch):
    from src.data.candle_engine import build_candles_for_day

    day = dt.date(2026, 9, 21)
    ticks = [
        _FakeIndexTickRow(dt.datetime(2026, 9, 21, 9, 32, tzinfo=TZ), 45000.0, 500000),
        _FakeIndexTickRow(dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ), 45100.0, 600000),
    ]
    inserted = _install_fake_repository(monkeypatch, index_ticks=ticks)

    report = build_candles_for_day(config, _session_factory, calendar, "KSE100", "index", day, timeframes=("daily",))

    assert report.ticks_used == 2
    daily = inserted[0]
    assert daily["open"] == 45000.0
    assert daily["close"] == 45100.0


def test_holiday_or_no_data_day_produces_no_daily_candle(config, monkeypatch, tmp_path):
    from src.data.candle_engine import build_candles_for_day

    calendar_path = tmp_path / "market_calendar.yaml"
    calendar_path.write_text(
        CALENDAR.replace('holidays: []', 'holidays:\n  - date: "2026-09-21"\n    name: "Test Holiday"\n    moon_dependent: false'),
        encoding="utf-8",
    )
    calendar = TradingCalendar(load_market_calendar(calendar_path), "Asia/Karachi")

    inserted = _install_fake_repository(monkeypatch, stock_ticks=[])
    report = build_candles_for_day(
        config, _session_factory, calendar, "THCCL", "stock", dt.date(2026, 9, 21), timeframes=("daily",)
    )

    assert report.ticks_used == 0
    assert inserted == []


def test_rebuilding_the_same_day_is_idempotent(config, calendar, monkeypatch):
    from src.data.candle_engine import build_candles_for_day

    day = dt.date(2026, 9, 21)
    ticks = [_FakeStockTickRow(dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ), 80.0, 1000)]
    inserted = _install_fake_repository(monkeypatch, stock_ticks=ticks)

    report1 = build_candles_for_day(config, _session_factory, calendar, "THCCL", "stock", day, timeframes=("daily",))
    report2 = build_candles_for_day(config, _session_factory, calendar, "THCCL", "stock", day, timeframes=("daily",))

    assert report1.written.get("daily") == 1
    assert report2.duplicate.get("daily") == 1
    assert len(inserted) == 1  # second run did not create a new row
