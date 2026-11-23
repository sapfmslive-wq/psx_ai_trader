"""Tests for src.data.collector (Phase 8).

collect_indices_once() imports src.database.repository lazily, inside the
function body (same pattern as the rest of the project — see main.py and
csv_importer.py). That means we can substitute a fake repository module in
sys.modules before calling it, and exercise the full orchestration logic
(skip-when-closed, force override, per-symbol quality/insert outcomes)
without SQLAlchemy being installed at all.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.config.loader import load_config
from src.data.mock_provider import MockProvider
from src.data.validator import DataQualityEngine
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
  - KMI30
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
    """Records every call instead of touching a real database."""

    def __init__(self):
        self.market_ticks: list[dict] = []
        self.index_ticks: list[dict] = []
        self.already_exists: set[tuple] = set()  # (provider, symbol, timestamp) to simulate DB duplicates

    def insert_market_tick(self, session, **kwargs):
        self.market_ticks.append(kwargs)
        return object()

    def insert_index_tick(self, session, **kwargs):
        key = (kwargs["provider"], kwargs["symbol"], kwargs["timestamp"])
        if key in self.already_exists:
            return False
        self.index_ticks.append(kwargs)
        return True


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
    monkeypatch.setitem(sys.modules, "src.database.repository", fake_module)
    return repo


def _session_factory():
    return _FakeSession()


def test_skips_recording_when_market_is_closed(config, calendar, fake_repo):
    from src.data.collector import collect_indices_once

    provider = MockProvider(seed=1)
    engine = DataQualityEngine(config)

    closed_time = dt.datetime(2026, 9, 21, 20, 0, tzinfo=TZ)  # Monday evening, well after close

    class _FrozenCalendar(TradingCalendar):
        def current_session(self, at=None):
            return super().current_session(closed_time)

        def is_market_open(self, at=None):
            return super().is_market_open(closed_time)

    frozen = _FrozenCalendar(calendar._calendar, "Asia/Karachi")

    report = collect_indices_once(config, _session_factory, provider, frozen, engine, force=False)

    assert report.skipped is True
    assert "not open" in report.skip_reason
    assert len(fake_repo.market_ticks) == 0
    assert len(fake_repo.index_ticks) == 0


def test_force_records_even_when_market_closed(config, calendar, fake_repo):
    from src.data.collector import collect_indices_once

    provider = MockProvider(seed=1)
    engine = DataQualityEngine(config)

    closed_time = dt.datetime(2026, 9, 21, 20, 0, tzinfo=TZ)

    class _FrozenCalendar(TradingCalendar):
        def current_session(self, at=None):
            return super().current_session(closed_time)

        def is_market_open(self, at=None):
            return super().is_market_open(closed_time)

    frozen = _FrozenCalendar(calendar._calendar, "Asia/Karachi")

    report = collect_indices_once(config, _session_factory, provider, frozen, engine, force=True)

    assert report.skipped is False
    assert len(report.outcomes) == 2  # KSE100, KMI30
    assert {o.symbol for o in report.outcomes} == {"KSE100", "KMI30"}
    assert len(fake_repo.market_ticks) == 2
    assert len(fake_repo.index_ticks) == 2


def test_records_all_configured_index_symbols_with_valid_status(config, calendar, fake_repo):
    from src.data.collector import collect_indices_once

    provider = MockProvider(seed=1)
    engine = DataQualityEngine(config)

    report = collect_indices_once(config, _session_factory, provider, calendar, engine, force=True)

    assert report.skipped is False
    for outcome in report.outcomes:
        assert outcome.quality_status == "VALID"  # first-ever observation for each symbol
        assert outcome.inserted is True


def test_db_level_duplicate_is_reported_as_not_inserted(config, calendar, fake_repo):
    from src.data.base_provider import DataProvider, IndexObservation
    from src.data.collector import collect_indices_once

    fixed_timestamp = dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ)

    class _FixedProvider(DataProvider):
        provider_name = "mock"
        is_synthetic = True

        def connect(self):
            self._connected = True

        def disconnect(self):
            self._connected = False

        def get_index_observation(self, symbol):
            self._require_connected()
            return IndexObservation(
                provider="mock", symbol=symbol, timestamp=fixed_timestamp, value=45000.0,
                change=0.0, change_percent=0.0, high=45000.0, low=45000.0, volume=1000.0,
                market_status="SYNTHETIC_OPEN", source_timestamp=fixed_timestamp, is_synthetic=True,
            )

        def get_stock_observation(self, symbol):
            raise NotImplementedError

    engine = DataQualityEngine(config)

    # Pre-seed the fake DB as if KSE100 at this exact timestamp already exists
    # (e.g. from an earlier run), but leave KMI30 untouched.
    fake_repo.already_exists.add(("mock", "KSE100", fixed_timestamp))

    report = collect_indices_once(config, _session_factory, _FixedProvider(), calendar, engine, force=True)

    outcomes_by_symbol = {o.symbol: o for o in report.outcomes}
    assert outcomes_by_symbol["KSE100"].inserted is False  # DB-level duplicate
    assert outcomes_by_symbol["KMI30"].inserted is True  # not seeded, so a fresh insert

    # A MarketTick is still written for both, unconditionally — raw
    # preservation doesn't care whether the cleaned tick was a duplicate.
    assert len(fake_repo.market_ticks) == 2
