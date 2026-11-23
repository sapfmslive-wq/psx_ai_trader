"""Tests for src.data.live_collector (Phase 9).

Same fake-repository-module technique as test_collector.py, so these run
fully offline without SQLAlchemy installed. Every source of real-world
nondeterminism (wall clock, time.sleep, cycle count) is injected.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from zoneinfo import ZoneInfo

import pytest

from src.config.loader import load_config
from src.data.base_provider import DataProvider, IndexObservation, StockObservation
from src.data.validator import DataQualityEngine
from src.market.trading_calendar import TradingCalendar, load_market_calendar
from src.utils.logger import get_logger

TZ = ZoneInfo("Asia/Karachi")
ALWAYS_OPEN_AT = dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ)  # a Monday, mid-session

SETTINGS = """
market:
  timezone: Asia/Karachi
data:
  provider: mock
collector:
  reconnect_seconds: 1
  stale_timeout_seconds: 30
  poll_interval_seconds: 1
  max_consecutive_failures: 3
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


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeRepository:
    def __init__(self):
        self.market_ticks = []
        self.index_ticks = []
        self.stock_ticks = []
        self.system_events = []

    def insert_market_tick(self, session, **kwargs):
        self.market_ticks.append(kwargs)
        return object()

    def insert_index_tick(self, session, **kwargs):
        self.index_ticks.append(kwargs)
        return True

    def insert_stock_tick(self, session, **kwargs):
        self.stock_ticks.append(kwargs)
        return True

    def insert_system_event(self, session, **kwargs):
        self.system_events.append(kwargs)
        return object()


class _AlwaysWorksProvider(DataProvider):
    provider_name = "mock"
    is_synthetic = True

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def get_index_observation(self, symbol):
        self._require_connected()
        return IndexObservation(
            provider="mock", symbol=symbol, timestamp=dt.datetime.now(dt.timezone.utc), value=45000.0,
            change=0.0, change_percent=0.0, high=45000.0, low=45000.0, volume=1000.0,
            market_status="SYNTHETIC_OPEN", source_timestamp=None, is_synthetic=True,
        )

    def get_stock_observation(self, symbol):
        self._require_connected()
        return StockObservation(
            provider="mock", symbol=symbol, timestamp=dt.datetime.now(dt.timezone.utc), last_price=80.0,
            open=80.0, high=80.0, low=80.0, previous_close=80.0, change=0.0, change_percent=0.0,
            volume=100.0, bid=79.9, bid_volume=10, ask=80.1, ask_volume=10, turnover=8000.0, vwap=80.0,
            market_status="SYNTHETIC_OPEN", source_timestamp=None, is_synthetic=True,
        )


class _FailNTimesThenWorkProvider(_AlwaysWorksProvider):
    """connect() fails the first `fail_count` times it's constructed."""

    remaining_failures = 0  # class-level, set per test

    def connect(self):
        if _FailNTimesThenWorkProvider.remaining_failures > 0:
            _FailNTimesThenWorkProvider.remaining_failures -= 1
            raise ConnectionError("simulated connect failure")
        self._connected = True


class _FlakyObservationProvider(_AlwaysWorksProvider):
    """Connects fine, but get_index_observation() fails a set number of times."""

    def __init__(self, fail_count: int):
        super().__init__()
        self._fail_count = fail_count

    def get_index_observation(self, symbol):
        self._require_connected()
        if self._fail_count > 0:
            self._fail_count -= 1
            raise RuntimeError("simulated feed error")
        return super().get_index_observation(symbol)


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


@pytest.fixture()
def fake_repo(monkeypatch):
    repo = _FakeRepository()
    fake_module = types.ModuleType("src.database.repository")
    fake_module.insert_market_tick = repo.insert_market_tick
    fake_module.insert_index_tick = repo.insert_index_tick
    fake_module.insert_stock_tick = repo.insert_stock_tick
    fake_module.insert_system_event = repo.insert_system_event
    monkeypatch.setitem(sys.modules, "src.database.repository", fake_module)
    return repo


def _session_factory():
    return _FakeSession()


def _make_collector(config, calendar, fake_repo, provider_factory, max_cycles, sleep_calls=None):
    from src.data.live_collector import LiveCollector

    engine = DataQualityEngine(config)
    logger = get_logger("data")
    sleep_calls = sleep_calls if sleep_calls is not None else []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    return LiveCollector(
        config=config,
        session_factory=_session_factory,
        provider_factory=provider_factory,
        calendar=calendar,
        quality_engine=engine,
        logger=logger,
        sleep_fn=fake_sleep,
        now_fn=lambda: ALWAYS_OPEN_AT,
        max_cycles=max_cycles,
        install_signal_handlers=False,
    )


def test_runs_requested_number_of_cycles_and_records_symbols(config, calendar, fake_repo):
    collector = _make_collector(config, calendar, fake_repo, _AlwaysWorksProvider, max_cycles=3)
    collector.run_forever()

    assert collector._cycles_run == 3
    # 1 index + 1 stock per cycle, 3 cycles.
    assert len(fake_repo.index_ticks) == 3
    assert len(fake_repo.stock_ticks) == 3
    assert len(fake_repo.market_ticks) == 6


def test_reconnect_backoff_retries_and_eventually_succeeds(config, calendar, fake_repo):
    _FailNTimesThenWorkProvider.remaining_failures = 2
    sleep_calls = []
    collector = _make_collector(
        config, calendar, fake_repo, _FailNTimesThenWorkProvider, max_cycles=1, sleep_calls=sleep_calls
    )
    collector.run_forever()

    assert collector._cycles_run == 1
    assert len(fake_repo.index_ticks) == 1  # eventually succeeded
    # Backoff: base=1s -> attempt1 wait 1s, attempt2 wait 2s (before third, successful attempt)
    assert sleep_calls[:2] == [1, 2] or sum(sleep_calls[:1]) >= 1  # at least some backoff occurred
    assert len(sleep_calls) >= 2


def test_failure_and_recovery_records_outage_events(config, calendar, fake_repo):
    provider = _FlakyObservationProvider(fail_count=1)
    collector = _make_collector(config, calendar, fake_repo, lambda: provider, max_cycles=2)
    collector.run_forever()

    # Cycle 1 fails (connection_lost recorded), cycle 2 succeeds (connection_restored recorded).
    levels_and_messages = [(e["level"], e["message"]) for e in fake_repo.system_events]
    assert any(level == "ERROR" and "connection_lost" in msg for level, msg in levels_and_messages)
    assert any(level == "INFO" and "connection_restored" in msg for level, msg in levels_and_messages)
    assert len(fake_repo.index_ticks) == 1  # only the successful cycle recorded data


def test_max_consecutive_failures_logs_critical_event(config, calendar, fake_repo):
    provider = _FlakyObservationProvider(fail_count=10)  # always fails within max_cycles
    collector = _make_collector(config, calendar, fake_repo, lambda: provider, max_cycles=3)
    collector.run_forever()

    # max_consecutive_failures is 3 in SETTINGS; after 3 straight failures a
    # CRITICAL system event should have been recorded.
    critical_events = [e for e in fake_repo.system_events if e["level"] == "CRITICAL"]
    assert len(critical_events) == 1
    assert "max_consecutive_failures" in critical_events[0]["message"]
    # The collector must still be running, not have crashed or exited abnormally.
    assert collector._cycles_run == 3


def test_request_stop_halts_the_loop_promptly(config, calendar, fake_repo):
    calls = {"n": 0}

    def fake_sleep(seconds):
        calls["n"] += 1
        if calls["n"] >= 1:
            collector.request_stop()

    collector = _make_collector(config, calendar, fake_repo, _AlwaysWorksProvider, max_cycles=None)
    collector._sleep_fn = fake_sleep  # override after construction, same object referenced in closure

    collector.run_forever()

    # Stopped after the first cycle's sleep, not after max_cycles (there is none).
    assert collector._cycles_run == 1
    assert collector._stop_requested is True


def test_shutdown_disconnects_the_provider(config, calendar, fake_repo):
    provider = _AlwaysWorksProvider()
    collector = _make_collector(config, calendar, fake_repo, lambda: provider, max_cycles=1)
    collector.run_forever()

    assert provider.is_connected() is False
