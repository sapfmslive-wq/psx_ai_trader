"""Tests for src.data.csv_importer (Phase 5/6).

These exercise the full import path including database writes, so — unlike
tests/test_csv_mapping.py and tests/test_quality_engine.py — they require
SQLAlchemy to be installed.
"""

from __future__ import annotations

import pytest

from src.config.loader import load_config
from src.data.csv_importer import CSVImporter

STOCK_CSV = """Date,Open,High,Low,Close,Volume
2026-09-14 09:30:00,80.0,80.5,79.8,80.2,10000
2026-09-14 09:31:00,80.2,80.6,80.1,80.4,8000
2026-09-14 09:31:00,80.2,80.6,80.1,80.4,8000
2026-09-14 09:32:00,80.4,79.0,80.4,80.9,7000
2026-09-14 09:33:00,80.9,81.0,80.7,-5.0,6000
2026-09-14 09:34:00,80.9,81.0,80.7,81.1,-100
not_a_date,80.9,81.0,80.7,81.1,5000
"""
# Row-by-row expected outcome (verified independently against the real
# validator.py + csv_mapping.py logic before writing these assertions):
#   09:30:00 -> written, VALID (first observation for this symbol)
#   09:31:00 -> written, VALID
#   09:31:00 (again) -> engine-level DUPLICATE, not written
#   09:32:00 -> written, INVALID (high < low: 79.0 < 80.4)
#   09:33:00 -> written, INVALID (close is -5.0, i.e. <= 0)
#   09:34:00 -> written, INVALID (negative volume)
#   not_a_date -> no timestamp at all, cannot be written as a tick
# => rows_read=7, written=5 (VALID=2, SUSPECT=0, INVALID=3),
#    duplicate=1, no_timestamp=1


@pytest.fixture()
def app_config(tmp_path):
    settings = tmp_path / "settings.yaml"
    symbols = tmp_path / "symbols.yaml"
    env = tmp_path / ".env"
    settings.write_text(
        """
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
""",
        encoding="utf-8",
    )
    symbols.write_text(
        """
indices:
  - KSE100
stocks:
  - THCCL
watchlist:
  - THCCL
""",
        encoding="utf-8",
    )
    env.write_text("LIVE_TRADING_ENABLED=false\n", encoding="utf-8")
    return load_config(settings_path=settings, symbols_path=symbols, env_path=env)


@pytest.fixture()
def session_factory(tmp_path):
    from src.database.engine import create_all_tables, create_engine_for_path, get_session_factory

    engine = create_engine_for_path(tmp_path / "test.db")
    create_all_tables(engine)
    return get_session_factory(engine)


def test_stock_csv_import_reports_expected_buckets(tmp_path, app_config, session_factory):
    csv_path = tmp_path / "THCCL.csv"
    csv_path.write_text(STOCK_CSV, encoding="utf-8")

    importer = CSVImporter(app_config, session_factory)
    report = importer.import_file(csv_path, symbol="THCCL", symbol_type="stock", provider_name="csv")

    assert report.rows_read == 7
    assert report.rows_written == 5
    assert report.rows_duplicate == 1
    assert report.rows_no_timestamp == 1
    assert report.quality_counts == {"VALID": 2, "SUSPECT": 0, "INVALID": 3}


def test_stock_csv_import_writes_market_ticks_for_every_row_read(tmp_path, app_config, session_factory):
    from src.database.models import MarketTick
    from src.database.repository import count_rows

    csv_path = tmp_path / "THCCL.csv"
    csv_path.write_text(STOCK_CSV, encoding="utf-8")

    importer = CSVImporter(app_config, session_factory)
    importer.import_file(csv_path, symbol="THCCL", symbol_type="stock", provider_name="csv")

    with session_factory() as session:
        # Raw preservation is unconditional: every row physically read
        # becomes a MarketTick, including the duplicate and the unparseable
        # timestamp row.
        assert count_rows(session, MarketTick) == 7


def test_stock_csv_import_writes_invalid_rows_flagged_not_dropped(tmp_path, app_config, session_factory):
    from src.database.models import StockTick
    from src.database.repository import count_rows

    csv_path = tmp_path / "THCCL.csv"
    csv_path.write_text(STOCK_CSV, encoding="utf-8")

    importer = CSVImporter(app_config, session_factory)
    importer.import_file(csv_path, symbol="THCCL", symbol_type="stock", provider_name="csv")

    with session_factory() as session:
        assert count_rows(session, StockTick) == 5  # matches rows_written
        rows = session.query(StockTick).order_by(StockTick.timestamp).all()
        statuses = [r.quality_status for r in rows]
        assert statuses.count("VALID") == 2
        assert statuses.count("INVALID") == 3
        # An INVALID row for a non-positive close still gets stored with the
        # (nullable) price column carrying whatever was parsed, and the
        # reason recorded — never silently dropped.
        negative_price_row = next(r for r in rows if r.quality_reason and "non-positive" in r.quality_reason)
        assert negative_price_row.last_price == -5.0
        assert negative_price_row.quality_status == "INVALID"


def test_index_csv_import_with_explicit_mapping(tmp_path, app_config, session_factory):
    csv_path = tmp_path / "KSE100.csv"
    csv_path.write_text(
        "WeirdDate,WeirdValue\n"
        "2026-09-14 09:30:00,45000.0\n"
        "2026-09-14 09:31:00,45050.5\n",
        encoding="utf-8",
    )

    importer = CSVImporter(app_config, session_factory)
    report = importer.import_file(
        csv_path,
        symbol="KSE100",
        symbol_type="index",
        provider_name="csv",
        column_mapping={"timestamp": "WeirdDate", "value": "WeirdValue"},
    )

    assert report.rows_read == 2
    assert report.rows_written == 2
    assert report.quality_counts["VALID"] == 2


def test_missing_required_column_raises_before_any_row_is_processed(tmp_path, app_config, session_factory):
    from src.data.csv_mapping import CSVMappingError

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("SomeColumn,AnotherColumn\n1,2\n", encoding="utf-8")

    importer = CSVImporter(app_config, session_factory)
    with pytest.raises(CSVMappingError):
        importer.import_file(csv_path, symbol="THCCL", symbol_type="stock")


def test_importing_the_same_file_twice_is_fully_idempotent(tmp_path, app_config, session_factory):
    from src.database.models import StockTick
    from src.database.repository import count_rows

    csv_path = tmp_path / "THCCL.csv"
    csv_path.write_text(STOCK_CSV, encoding="utf-8")

    importer = CSVImporter(app_config, session_factory)
    importer.import_file(csv_path, symbol="THCCL", symbol_type="stock", provider_name="csv")
    second_report = importer.import_file(csv_path, symbol="THCCL", symbol_type="stock", provider_name="csv")

    # Second run: the 5 previously-written rows are now DB-level duplicates,
    # the 1 in-file duplicate is caught again by the engine, and the
    # unparseable-timestamp row still can't be written. Nothing new appears.
    assert second_report.rows_written == 0
    assert second_report.rows_duplicate == 6
    assert second_report.rows_no_timestamp == 1

    with session_factory() as session:
        assert count_rows(session, StockTick) == 5  # unchanged from the first import


def test_unknown_symbol_is_written_as_invalid_not_rejected(tmp_path, app_config, session_factory):
    """A symbol not present in config/symbols.yaml is a data-quality issue,
    not a reason to refuse the whole file — the row is still recorded so
    the person can see it happened, flagged INVALID."""
    from src.database.models import StockTick
    from src.database.repository import count_rows

    csv_path = tmp_path / "UNKNOWN.csv"
    csv_path.write_text(
        "Date,Close\n2026-09-14 09:30:00,50.0\n",
        encoding="utf-8",
    )

    importer = CSVImporter(app_config, session_factory)
    report = importer.import_file(csv_path, symbol="NOT_IN_CONFIG", symbol_type="stock", provider_name="csv")

    assert report.rows_written == 1
    assert report.quality_counts["INVALID"] == 1

    with session_factory() as session:
        row = session.query(StockTick).one()
        assert row.quality_status == "INVALID"
        assert "configured universe" in (row.quality_reason or "")
