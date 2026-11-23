"""PSX AI Trader — command line entrypoint.

Phase 1-15 scope only. Sixteen commands exist:

    python -m src.main init
    python -m src.main health
    python -m src.main validate-config
    python -m src.main init-db
    python -m src.main provider-demo
    python -m src.main import-csv
    python -m src.main market-status
    python -m src.main record-indices
    python -m src.main record-stocks
    python -m src.main run-collector
    python -m src.main build-candles
    python -m src.main build-indicators
    python -m src.main build-features
    python -m src.main build-labels
    python -m src.main evaluate-baselines
    python -m src.main train

Every other command described in the project README (collect, build-candles,
features, train, predict, backtest, paper-trade, dashboard, report, ...) is
added in the phase that implements the underlying functionality. Adding a
command here before its functionality exists would produce exactly the kind
of impressive-looking-but-empty CLI this project is explicitly avoiding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

from src.config.loader import ConfigValidationError, load_config
from src.utils.file_utils import (
    ensure_all_required_directories,
    missing_required_directories,
    project_root,
    resolve_path,
)
from src.utils.logger import get_logger

MIN_PYTHON_VERSION = (3, 10)


def _run_init() -> int:
    """Create the project directory tree and ensure `.env` exists."""
    logger = get_logger("system")
    logger.info("Running init: ensuring required directories exist.")

    created = ensure_all_required_directories()

    env_path = resolve_path(".env")
    env_example_path = resolve_path(".env.example")
    env_status: str
    if env_path.exists():
        env_status = "already present"
    elif env_example_path.exists():
        shutil.copyfile(env_example_path, env_path)
        env_status = "created from .env.example"
        logger.info("Created .env from .env.example.")
    else:
        env_status = "MISSING (.env.example not found either)"
        logger.error(".env.example not found; cannot create .env.")

    print("PSX AI TRADER — INIT")
    print("---------------------")
    print(f"Project root: {project_root()}")
    print(f"Directories ensured: {len(created)}")
    for path in created:
        print(f"  OK  {path.relative_to(project_root())}")
    print(f".env: {env_status}")
    print()

    if env_status.startswith("MISSING"):
        print("INIT COMPLETED WITH WARNINGS")
        return 1

    print("INIT COMPLETE")
    return 0


def _check_python_version() -> tuple[bool, str]:
    ok = sys.version_info[:2] >= MIN_PYTHON_VERSION
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    detail = f"{current} (>= {required} required)"
    return ok, detail


def _run_validate_config() -> int:
    """Load config/settings.yaml + config/symbols.yaml + .env and report."""
    logger = get_logger("system")
    print("PSX AI TRADER — VALIDATE CONFIG")
    print("---------------------------------")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        logger.error("Configuration validation failed: %s", exc.errors)
        print("CONFIG INVALID:\n")
        for err in exc.errors:
            print(f"  - {err}")
        print()
        print(f"{len(exc.errors)} problem(s) found. Fix config/settings.yaml "
              f"and/or config/symbols.yaml and re-run.")
        return 1

    print(f"market.timezone            {config.market.timezone}")
    print(f"data.provider              {config.data.provider}")
    print(f"collector.reconnect_s      {config.collector.reconnect_seconds}")
    print(f"collector.stale_timeout_s  {config.collector.stale_timeout_seconds}")
    print(f"collector.poll_interval_s  {config.collector.poll_interval_seconds}")
    print(f"collector.max_failures     {config.collector.max_consecutive_failures}")
    print(f"prediction.horizons        {', '.join(config.prediction.horizons)}")
    print(f"risk.max_risk_per_trade    {config.risk.max_risk_per_trade}")
    print(f"risk.max_positions         {config.risk.max_positions}")
    print(f"trading.mode               {config.trading.mode}")
    print(f"trading.live_enabled       {config.trading.live_trading_enabled}")
    print(f"dashboard.host:port        {config.dashboard.host}:{config.dashboard.port}")
    print(f"logging.level              {config.logging.level}")
    print(f"quality.max_change_percent {config.quality.max_change_percent}")
    print(f"quality.stale_seconds      {config.quality.stale_seconds}")
    print(f"quality.future_tolerance_s {config.quality.future_timestamp_tolerance_seconds}")
    print(f"symbols.indices            {', '.join(config.symbols.indices)}")
    print(f"symbols.stocks             {', '.join(config.symbols.stocks)}")
    print(f"symbols.watchlist          {', '.join(config.symbols.watchlist)}")
    print()
    print("CONFIG VALID")
    logger.info("Configuration validated successfully.")
    return 0


def _check_pytest_installed() -> tuple[bool, str]:
    try:
        version = importlib_metadata.version("pytest")
        return True, f"pytest {version} installed"
    except importlib_metadata.PackageNotFoundError:
        return False, "pytest not installed (run: pip install -r requirements.txt)"


def _run_init_db() -> int:
    """Create the SQLite database file and every table defined in models.py."""
    logger = get_logger("system")
    print("PSX AI TRADER — INIT DATABASE")
    print("--------------------------------")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot initialize database: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.database.models import ALL_TABLES
        from src.database.repository import insert_system_event
    except ModuleNotFoundError as exc:
        print(f"Cannot initialize database: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    engine = build_engine()
    create_all_tables(engine)

    session_factory = get_session_factory(engine)
    with session_factory() as session:
        import datetime as _dt

        insert_system_event(
            session,
            timestamp=_dt.datetime.now(_dt.timezone.utc),
            level="INFO",
            category="system",
            message="Database initialized via `init-db`.",
        )

    db_path = resolve_path(config.database.path)
    print(f"Database file: {db_path}")
    print(f"Tables created/verified: {len(ALL_TABLES)}")
    for model in ALL_TABLES:
        print(f"  OK  {model.__tablename__}")
    print()
    print("INIT-DB COMPLETE")
    logger.info("Database initialized at %s with %d tables.", db_path, len(ALL_TABLES))
    return 0


def _check_database() -> tuple[bool, str]:
    try:
        from src.database.engine import build_engine, get_session_factory
        from src.database.models import ALL_TABLES, SystemEvent
        from src.database.repository import count_rows
    except ModuleNotFoundError:
        return False, "SQLAlchemy not installed — run: pip install -r requirements.txt"

    try:
        engine = build_engine()
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())
        expected_tables = {model.__tablename__ for model in ALL_TABLES}
        missing = expected_tables - existing_tables
        if missing:
            return False, f"missing tables {sorted(missing)} — run: python -m src.main init-db"

        session_factory = get_session_factory(engine)
        with session_factory() as session:
            event_count = count_rows(session, SystemEvent)
        return True, f"all {len(expected_tables)} tables present ({event_count} system events logged)"
    except ConfigValidationError:
        return False, "cannot check — configuration is invalid"
    except Exception as exc:  # noqa: BLE001 — health check must never crash the CLI
        return False, f"connection failed: {exc}"


def _run_health() -> int:
    """Run the Phase 1-15 subset of health checks.

    This intentionally does NOT check model files or dashboard reachability,
    or real broker connectivity — those checks are added in the phases that
    create those components (see README.md roadmap). Reporting a false "OK"
    for a component that doesn't exist yet would be dishonest.
    """
    logger = get_logger("system")
    logger.info("Running health check (Phase 15 scope).")

    checks: list[tuple[str, bool, str]] = []

    py_ok, py_detail = _check_python_version()
    checks.append(("Python version", py_ok, py_detail))

    missing_dirs = missing_required_directories()
    dirs_ok = not missing_dirs
    dirs_detail = "all present" if dirs_ok else f"missing: {', '.join(missing_dirs)}"
    checks.append(("Required directories", dirs_ok, dirs_detail))

    env_exists = resolve_path(".env").exists()
    checks.append((
        ".env file",
        env_exists,
        "present" if env_exists else "missing (run: python -m src.main init)",
    ))

    pytest_ok, pytest_detail = _check_pytest_installed()
    checks.append(("pytest available", pytest_ok, pytest_detail))

    try:
        config = load_config()
        checks.append(("Configuration valid", True, "settings.yaml + symbols.yaml + .env OK"))
    except ConfigValidationError as exc:
        config = None
        checks.append((
            "Configuration valid",
            False,
            f"{len(exc.errors)} problem(s) — run: python -m src.main validate-config",
        ))

    db_ok, db_detail = _check_database()
    checks.append(("Database", db_ok, db_detail))

    provider_ok, provider_detail = _check_data_provider()
    provider_label = f"Data provider ({config.data.provider})" if config else "Data provider"
    checks.append((provider_label, provider_ok, provider_detail))

    calendar_ok, calendar_detail = _check_trading_calendar()
    checks.append(("Trading calendar", calendar_ok, calendar_detail))

    print("PSX AI TRADER — HEALTH CHECK (Phase 15 scope)")
    print("---------------------------------------------")
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:4}] {name:24} {detail}")
    print()
    print("NOT YET CHECKED (implemented in later phases):")
    print("  - Broker provider connectivity        (Phase 26)")
    print("  - Model files                          (Phase 15)")
    print("  - Dashboard reachability                 (Phase 21)")
    print()

    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        print("HEALTH CHECK: OK (Phase 15 scope)")
        logger.info("Health check passed (Phase 15 scope).")
        return 0

    print("HEALTH CHECK: FAILED")
    logger.error("Health check failed: %s", [n for n, ok, _ in checks if not ok])
    return 1


def _check_data_provider() -> tuple[bool, str]:
    try:
        config = load_config()
    except ConfigValidationError:
        return False, "cannot check — configuration is invalid"

    provider_name = config.data.provider

    if provider_name == "mock":
        try:
            from src.data.mock_provider import MockProvider
        except ModuleNotFoundError as exc:
            return False, f"missing dependency: {exc}"
        try:
            provider = MockProvider(seed=42)
            with provider:
                sample_symbol = config.symbols.indices[0] if config.symbols.indices else None
                sample_value = (
                    provider.get_index_observation(sample_symbol).value if sample_symbol else None
                )
            return True, f"mock provider OK (SYNTHETIC sample {sample_symbol}={sample_value})"
        except Exception as exc:  # noqa: BLE001 — health check must never crash the CLI
            return False, f"mock provider failed: {exc}"

    if provider_name == "psx":
        from src.data.psx_provider import PSXConnectionError, PSXLicenseRequiredError, PSXProvider

        provider = PSXProvider.from_env()
        if not provider.is_configured():
            return (
                False,
                "not licensed/configured — see .env.example (PSX_DATA_API_BASE_URL/"
                "KEY) and README Phase 7 section",
            )
        try:
            with provider:
                pass
            return True, f"connected to configured licensed endpoint ({provider._base_url})"
        except (PSXLicenseRequiredError, PSXConnectionError) as exc:
            return False, f"configured but unreachable: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"psx provider failed: {exc}"

    return (
        False,
        f"provider '{provider_name}' has no implementation yet (Phase 5/26) "
        f"— set data.provider: mock, csv, or psx in settings.yaml",
    )


def _run_provider_demo() -> int:
    """Connect the configured provider and print one observation per
    configured index/stock symbol. Demonstration only — this is NOT the
    real collector (that is Phase 8/9)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot run provider demo: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    if config.data.provider == "mock":
        return _run_provider_demo_mock(config, logger)
    if config.data.provider == "psx":
        return _run_provider_demo_psx(config, logger)

    print(
        f"data.provider is '{config.data.provider}', which has no implementation "
        f"yet. This demo supports 'mock' and 'psx'."
    )
    return 1


def _run_provider_demo_mock(config, logger) -> int:  # noqa: ANN001
    from src.data.mock_provider import MockProvider

    print("PSX AI TRADER — PROVIDER DEMO (mock)")
    print("--------------------------------------")
    print("*** ALL DATA BELOW IS SYNTHETIC — NOT REAL PSX DATA ***")
    print()

    with MockProvider() as provider:
        for symbol in config.symbols.indices:
            obs = provider.get_index_observation(symbol)
            print(
                f"[SYNTHETIC] {symbol:10} value={obs.value:<12} "
                f"change={obs.change:<10} change%={obs.change_percent:<8} "
                f"status={obs.market_status}"
            )
        for symbol in config.symbols.stocks:
            obs = provider.get_stock_observation(symbol)
            print(
                f"[SYNTHETIC] {symbol:10} last={obs.last_price:<10} "
                f"bid={obs.bid:<10} ask={obs.ask:<10} volume={obs.volume:<12} "
                f"status={obs.market_status}"
            )

    logger.info("Provider demo completed using mock provider (SYNTHETIC data).")
    print()
    print("Reminder: this command only proves the provider interface works.")
    print("It does not store data — that is Phase 8/9 (collector + storage).")
    return 0


def _run_provider_demo_psx(config, logger) -> int:  # noqa: ANN001
    from src.data.psx_provider import PSXConnectionError, PSXLicenseRequiredError, PSXProvider

    provider = PSXProvider.from_env()

    print("PSX AI TRADER — PROVIDER DEMO (psx)")
    print("-------------------------------------")

    try:
        provider.connect()
    except PSXLicenseRequiredError as exc:
        print(str(exc))
        return 1
    except PSXConnectionError as exc:
        print(f"Configured endpoint is unreachable: {exc}")
        return 1

    print(f"*** REAL DATA from your configured licensed endpoint ({provider._base_url}) ***")
    print()
    try:
        for symbol in config.symbols.indices:
            obs = provider.get_index_observation(symbol)
            print(f"[LICENSED] {symbol:10} value={obs.value} status={obs.market_status}")
        for symbol in config.symbols.stocks:
            obs = provider.get_stock_observation(symbol)
            print(f"[LICENSED] {symbol:10} last={obs.last_price} status={obs.market_status}")
    finally:
        provider.disconnect()

    logger.info("Provider demo completed using psx provider (licensed feed).")
    print()
    print("Reminder: this command only proves the provider interface works.")
    print("It does not store data — that is Phase 8/9 (collector + storage).")
    return 0


def _run_import_csv(args: argparse.Namespace) -> int:
    """Import one historical CSV file (Section 95)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot import CSV: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"CSV file not found: {file_path}")
        return 1

    column_mapping = None
    if args.mapping:
        mapping_path = Path(args.mapping)
        if not mapping_path.is_file():
            print(f"Mapping file not found: {mapping_path}")
            return 1
        try:
            column_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Mapping file is not valid JSON: {exc}")
            return 1

    try:
        from src.data.csv_importer import CSVImporter
        from src.data.csv_mapping import CSVMappingError
        from src.database.engine import build_engine, create_all_tables, get_session_factory
    except ModuleNotFoundError as exc:
        print(f"Cannot import CSV: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    engine = build_engine()
    create_all_tables(engine)  # safe/idempotent if init-db was never run
    session_factory = get_session_factory(engine)

    importer = CSVImporter(config, session_factory)

    print("PSX AI TRADER — IMPORT CSV")
    print("-----------------------------")
    try:
        report = importer.import_file(
            file_path=file_path,
            symbol=args.symbol,
            symbol_type=args.symbol_type,
            provider_name=args.provider_name,
            column_mapping=column_mapping,
        )
    except CSVMappingError as exc:
        print(f"Could not import {file_path}: {exc}")
        logger.error("CSV import failed for %s: %s", file_path, exc)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not import {file_path}: {exc}")
        logger.error("CSV import failed for %s: %s", file_path, exc)
        return 1

    for line in report.summary_lines():
        print(line)

    logger.info(
        "CSV import complete: file=%s symbol=%s read=%d written=%d (valid=%d suspect=%d invalid=%d) "
        "duplicate=%d no_timestamp=%d",
        report.file_path, report.symbol, report.rows_read, report.rows_written,
        report.quality_counts["VALID"], report.quality_counts["SUSPECT"], report.quality_counts["INVALID"],
        report.rows_duplicate, report.rows_no_timestamp,
    )
    if report.notable_rows:
        for row_number, status, reason in report.notable_rows:
            logger.warning("CSV import row %d [%s]: %s", row_number, status, reason)

    with session_factory() as session:
        from src.database.repository import insert_system_event
        import datetime as _dt

        insert_system_event(
            session,
            timestamp=_dt.datetime.now(_dt.timezone.utc),
            level="INFO",
            category="data",
            message=(
                f"CSV import: {report.file_path} symbol={report.symbol} "
                f"read={report.rows_read} written={report.rows_written} "
                f"duplicate={report.rows_duplicate} no_timestamp={report.rows_no_timestamp}"
            ),
        )

    print()
    if report.rows_read > 0 and report.rows_written == 0 and report.rows_duplicate == 0:
        print("IMPORT COMPLETED WITH NO USABLE ROWS — check the notable rows above.")
        return 1

    print("IMPORT COMPLETE")
    return 0


def _check_trading_calendar() -> tuple[bool, str]:
    try:
        config = load_config()
    except ConfigValidationError:
        return False, "cannot check — configuration is invalid"

    try:
        from src.market.trading_calendar import MarketCalendarValidationError, TradingCalendar
    except ModuleNotFoundError as exc:
        return False, f"missing dependency: {exc}"

    try:
        calendar = TradingCalendar.from_config(config)
    except MarketCalendarValidationError as exc:
        return False, f"{len(exc.errors)} problem(s) — run: python -m src.main market-status"

    session = calendar.current_session()
    return True, f"loaded OK, current session: {session}"


def _run_market_status() -> int:
    """Show the current PSX market session state (Section 14)."""
    logger = get_logger("system")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot check market status: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        from src.market.trading_calendar import MarketCalendarValidationError, TradingCalendar
    except ModuleNotFoundError as exc:
        print(f"Cannot check market status: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    try:
        calendar = TradingCalendar.from_config(config)
    except MarketCalendarValidationError as exc:
        print("Market calendar is invalid:\n")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    import datetime as _dt
    from zoneinfo import ZoneInfo as _ZoneInfo

    now = _dt.datetime.now(_ZoneInfo(config.market.timezone))
    session = calendar.current_session()
    print("PSX AI TRADER — MARKET STATUS")
    print("--------------------------------")
    print(f"Timezone:        {config.market.timezone}")
    print(f"Current time:    {now.isoformat()}")
    print(f"Current session: {session}")
    print(f"Market open:     {calendar.is_market_open()}")
    if calendar.is_holiday():
        print(f"Holiday today:   {calendar.holiday_name()}")
    print(f"Next open:       {calendar.next_open()}")
    print(f"Next close:      {calendar.next_close()}")
    logger.info("Market status checked: session=%s", session)
    return 0


def _build_provider_for_config(config):  # noqa: ANN001
    """Return a connectable (not yet connected) DataProvider matching
    config.data.provider, or (None, error_message) if that isn't possible
    right now (unimplemented provider, or psx unconfigured)."""
    if config.data.provider == "mock":
        from src.data.mock_provider import MockProvider

        return MockProvider(), None
    if config.data.provider == "psx":
        from src.data.psx_provider import LICENSE_REQUIRED_MESSAGE, PSXProvider

        provider = PSXProvider.from_env()
        if not provider.is_configured():
            return None, LICENSE_REQUIRED_MESSAGE
        return provider, None
    return None, f"data.provider '{config.data.provider}' has no implementation yet."


def _run_record(args: argparse.Namespace, kind: str) -> int:
    """Shared implementation for record-indices / record-stocks.

    Args:
        kind: "index" or "stock".
    """
    label = "INDICES" if kind == "index" else "STOCKS"
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print(f"Cannot record {kind}s: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        from src.data.collector import collect_indices_once, collect_stocks_once
        from src.data.validator import DataQualityEngine
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.market.trading_calendar import MarketCalendarValidationError, TradingCalendar
    except ModuleNotFoundError as exc:
        print(f"Cannot record {kind}s: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    try:
        calendar = TradingCalendar.from_config(config)
    except MarketCalendarValidationError as exc:
        print("Market calendar is invalid:\n")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    provider, error = _build_provider_for_config(config)
    if provider is None:
        print(error)
        return 1

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)
    quality_engine = DataQualityEngine(config)
    collect_fn = collect_indices_once if kind == "index" else collect_stocks_once

    print(f"PSX AI TRADER — RECORD {label}")
    print("---------------------------------")
    try:
        report = collect_fn(config, session_factory, provider, calendar, quality_engine, force=args.force)
    except Exception as exc:  # noqa: BLE001 — never crash the CLI on a provider hiccup
        print(f"Recording failed: {exc}")
        logger.error("record-%ss failed: %s", kind, exc)
        return 1

    if report.skipped:
        print(f"SKIPPED: {report.skip_reason}")
        print("Use --force to record anyway (mock provider only — meaningless for real feeds).")
        return 0

    print(f"Session: {report.session}")
    for outcome in report.outcomes:
        dup_note = "" if outcome.inserted else " (duplicate — already in DB)"
        print(f"  {outcome.symbol:10} [{outcome.quality_status}]{dup_note} {outcome.quality_reason or ''}")
    logger.info(
        "record-%ss: session=%s symbols=%s",
        kind, report.session, [(o.symbol, o.quality_status, o.inserted) for o in report.outcomes],
    )
    print()
    print(f"RECORD-{label} COMPLETE")
    return 0


def _run_record_indices(args: argparse.Namespace) -> int:
    return _run_record(args, "index")


def _run_record_stocks(args: argparse.Namespace) -> int:
    return _run_record(args, "stock")


def _run_collector(args: argparse.Namespace) -> int:
    """Run the continuous, robust collector loop (Phase 9, Section 15)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot start collector: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        from src.data.live_collector import LiveCollector
        from src.data.validator import DataQualityEngine
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.market.trading_calendar import MarketCalendarValidationError, TradingCalendar
    except ModuleNotFoundError as exc:
        print(f"Cannot start collector: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    try:
        calendar = TradingCalendar.from_config(config)
    except MarketCalendarValidationError as exc:
        print("Market calendar is invalid:\n")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    if config.data.provider == "mock":
        def provider_factory():
            from src.data.mock_provider import MockProvider
            return MockProvider()
    elif config.data.provider == "psx":
        from src.data.psx_provider import LICENSE_REQUIRED_MESSAGE, PSXProvider

        probe = PSXProvider.from_env()
        if not probe.is_configured():
            print(LICENSE_REQUIRED_MESSAGE)
            return 1

        def provider_factory():
            from src.data.psx_provider import PSXProvider
            return PSXProvider.from_env()
    else:
        print(f"data.provider '{config.data.provider}' has no implementation yet.")
        return 1

    engine_obj = build_engine()
    create_all_tables(engine_obj)
    session_factory = get_session_factory(engine_obj)
    quality_engine = DataQualityEngine(config)

    print("PSX AI TRADER — LIVE COLLECTOR")
    print("---------------------------------")
    print(f"Provider:       {config.data.provider}")
    print(f"Poll interval:  {config.collector.poll_interval_seconds}s")
    print(f"Symbols:        {', '.join(config.symbols.indices + config.symbols.stocks)}")
    print("Press Ctrl+C to stop gracefully.")
    print()

    collector = LiveCollector(
        config=config,
        session_factory=session_factory,
        provider_factory=provider_factory,
        calendar=calendar,
        quality_engine=quality_engine,
        logger=logger,
        max_cycles=args.max_cycles,
    )
    collector.run_forever()
    return 0


def _fmt(value: float, digits: int = 3) -> str:
    return "  n/a" if value != value else f"{value:.{digits}f}"


def _run_train(args: argparse.Namespace) -> int:
    """Train the configured ML models on one symbol and compare them, on the
    same untouched test rows, against the baselines (Phase 15). Prediction
    quality only — no costs, no P&L."""
    logger = get_logger("model")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot train: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        start_date = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
    except ValueError:
        print("--start-date/--end-date must be YYYY-MM-DD")
        return 1
    if start_date > end_date:
        print("--start-date must not be after --end-date")
        return 1
    if args.symbol not in (config.symbols.indices + config.symbols.stocks):
        print(f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks).")
        return 1
    if args.horizon not in config.labels.horizons:
        print(f"--horizon must be one of the configured labels.horizons: {list(config.labels.horizons)}")
        return 1

    model_names = [m.strip() for m in args.models.split(",")] if args.models else list(config.models.enabled)
    unknown = [m for m in model_names if m not in ("logistic", "random_forest", "gradient_boosting", "hist_gradient_boosting")]
    if unknown:
        print(f"Unknown model(s): {unknown}")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.models.trainer import train_and_store
    except ModuleNotFoundError as exc:
        print(f"Cannot train: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.market.timezone)
    start = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=tz)
    end = _dt.datetime.combine(end_date, _dt.time.min, tzinfo=tz) + _dt.timedelta(days=1)

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    try:
        r = train_and_store(config, session_factory, args.symbol, args.timeframe, start, end, args.horizon, model_names)
    except ValueError as exc:
        print(f"Cannot train: {exc}")
        return 1

    print("PSX AI TRADER — MODEL TRAINING")
    print("----------------------------------")
    print(f"Symbol / timeframe:  {r.symbol} / {r.timeframe}   horizon: {r.horizon} bar(s) ahead (direction)")
    print(f"Usable rows:         {r.n_rows} of {r.n_input_bars} bars "
          f"({r.n_dropped_warmup_rows} warm-up/NaN rows and {r.n_dropped_unlabeled_rows} unlabeled rows dropped)")
    print(f"Features:            {len(r.feature_names)} scale-free features"
          + (f"  (dropped all-NaN: {r.dropped_columns})" if r.dropped_columns else ""))
    print(f"Chronological split: train {r.n_train} | validation {r.n_validation} | test {r.n_test}   "
          f"(purged {r.purged_from_train}+{r.purged_from_validation} rows at the boundaries)")
    print(f"  train      {r.train_period[0]} .. {r.train_period[1]}")
    print(f"  validation {r.validation_period[0]} .. {r.validation_period[1]}")
    print(f"  TEST       {r.test_period[0]} .. {r.test_period[1]}   (untouched by every choice)")
    print(f"Test UP rate:        {_fmt(r.test_up_rate)}     random seed: {r.random_seed}")
    print()
    print("BASELINES on the same test rows")
    print(f"  {'name':<19}{'scored':>7}{'acc':>7}{'balAcc':>8}  verdict")
    for b in r.baselines:
        print(f"  {b.name:<19}{b.n_predicted:>7}{_fmt(b.accuracy):>7}{_fmt(b.balanced_accuracy):>8}  {b.verdict}")
    print(f"  best baseline with enough data: {r.best_baseline_name} (accuracy {_fmt(r.best_baseline_accuracy)})")
    print()
    print(f"MODELS on the test rows  ({r.models_compared} compared -> significance bar {r.alpha:.4f} each)")
    header = (f"  {'model':<24}{'valLL':>7}{'acc':>7}{'balAcc':>8}{'AUC':>7}{'PR-AUC':>8}"
              f"{'Brier':>7}{'logLoss':>9}{'ECE':>7}{'p':>8}  result")
    print(header)
    for res in r.results:
        t = res.test
        c = t.classification
        flag = "CLEARS BASELINES" if res.clears_baselines else "does not clear baselines"
        print(f"  {res.name:<24}{_fmt(res.validation_log_loss):>7}{_fmt(c.accuracy):>7}{_fmt(c.balanced_accuracy):>8}"
              f"{_fmt(t.roc_auc):>7}{_fmt(t.pr_auc):>8}{_fmt(t.brier):>7}{_fmt(t.log_loss):>9}{_fmt(t.ece):>7}"
              f"{_fmt(res.p_value, 4):>8}  {flag}")
        print(f"  {'':<24}{res.verdict}   [params: {res.params}]")
    print()
    chosen = next(res for res in r.results if res.name == r.selected_by_validation)
    print(f"Selected by VALIDATION log loss (not by test results): {chosen.name}")
    if chosen.importances:
        print("  top permutation importances (validation set; what the model leans on, not what causes anything):")
        for imp in chosen.importances[:8]:
            print(f"    {imp['rank']:>2}. {imp['feature']:<34}{imp['importance']:+.4f}")
    print()
    print("HOW TO READ THIS")
    print("  * Hyperparameters and the probability calibrator used only train/validation data.")
    print("    Validation is used twice (to choose, then to calibrate) — disclosed, and the TEST")
    print("    segment influenced nothing. Test results are shown for ALL models, not just the best.")
    print("  * 'CLEARS BASELINES' needs a genuine beat of a constant guess at the tightened bar AND")
    print("    higher accuracy than the best baseline. Several models were tried on the same data,")
    print("    so one apparent winner among them is weaker evidence than it looks.")
    print("  * ECE/Brier/logLoss measure whether the stated probabilities can be trusted, not just")
    print("    whether the guesses were right.")
    print("  * AUC / PR-AUC / ECE come from ONE test segment of overlapping, autocorrelated bars and")
    print("    carry no significance test of their own. Measured on pure random-walk noise (40 series),")
    print("    the test AUC of a model with no real signal ranged from 0.28 to 0.77. A single AUC")
    print("    of 0.7 is therefore NOT evidence of skill; the 'result' column is the safer read.")
    print("  * This is prediction quality only. No fees, tax, slippage or spread are applied")
    print("    (Phase 16), so NOTHING above says any model is profitable.")
    print("  * One symbol, one period, one split. Walk-forward validation is Phase 17.")
    print()
    print(f"Saved {len(r.results)} model(s) under models/trained + models/metadata, recorded model_runs rows")
    print("(status TRAINED) and stored the out-of-sample TEST predictions (not live predictions).")
    logger.info(
        "train: symbol=%s timeframe=%s horizon=%d models=%s selected=%s clears=%s",
        r.symbol, r.timeframe, r.horizon, model_names, r.selected_by_validation,
        [res.name for res in r.results if res.clears_baselines],
    )
    print("TRAIN COMPLETE")
    return 0


def _run_evaluate_baselines(args: argparse.Namespace) -> int:
    """Score the six baseline predictors on one symbol (Phase 14). Prediction
    quality only — no costs, no P&L."""
    logger = get_logger("model")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot evaluate baselines: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        start_date = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
    except ValueError:
        print("--start-date/--end-date must be YYYY-MM-DD")
        return 1
    if start_date > end_date:
        print("--start-date must not be after --end-date")
        return 1
    if args.symbol not in (config.symbols.indices + config.symbols.stocks):
        print(f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks).")
        return 1
    if args.horizon not in config.labels.horizons:
        print(f"--horizon must be one of the configured labels.horizons: {list(config.labels.horizons)}")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.models.baseline_runner import evaluate_and_store_baselines
    except ModuleNotFoundError as exc:
        print(f"Cannot evaluate baselines: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.market.timezone)
    start = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=tz)
    end = _dt.datetime.combine(end_date, _dt.time.min, tzinfo=tz) + _dt.timedelta(days=1)

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    try:
        report = evaluate_and_store_baselines(
            config, session_factory, args.symbol, args.timeframe, start, end, args.horizon
        )
    except ValueError as exc:
        print(f"Cannot evaluate baselines: {exc}")
        return 1

    print("PSX AI TRADER — BASELINE EVALUATION")
    print("---------------------------------------")
    print(f"Symbol / timeframe:  {report.symbol} / {report.timeframe}")
    print(f"Prediction horizon:  {report.horizon} bar(s) ahead (direction)")
    print(f"Bars:                {report.n_bars} total = {report.train_bars} train "
          f"+ {report.horizon} purged + {report.test_bars} test (chronological, never shuffled)")
    print(f"Test segment:        {report.test_start} .. {report.test_end}")
    print(f"Test UP rate:        {_fmt(report.test_up_rate)}   (share of test bars that rose)")
    print(f"Majority class (train): {report.majority_class_from_train}   random seed: {report.random_seed}")
    print()
    header = (f"{'baseline':<19}{'scored':>7}{'nEff':>6}{'cover':>7}{'acc':>7}{'balAcc':>8}"
              f"{'F1(up)':>8}{'null':>7}{'p':>8}  verdict")
    print(header)
    print("-" * len(header))
    for row in report.rows:
        m = row.metrics
        print(f"{row.name:<19}{m.n_predicted:>7}{row.n_effective:>6}{_fmt(m.coverage, 2):>7}{_fmt(m.accuracy):>7}"
              f"{_fmt(m.balanced_accuracy):>8}{_fmt(m.f1_up):>8}{_fmt(m.null_accuracy):>7}"
              f"{_fmt(row.p_value, 4):>8}  {row.verdict}")
    print()
    print(f"Buy-and-hold gross return over the test segment: {report.buy_hold_gross_return_pct:+.2f}% "
          f"(BEFORE any costs or slippage)")
    print()
    print("HOW TO READ THIS")
    print("  * 'null' is what a constant guess would score on the same bars. A baseline only")
    print("    shows skill if it clearly beats null, not just 50%.")
    print("  * nEff ~ scored / horizon: overlapping labels are not independent, so p-values use")
    print("    that smaller count. Still approximate; a small p means 'look closer', not 'proven'.")
    print("  * This is prediction quality only. No fees, tax, slippage or spread are applied")
    print("    (Phase 16), so NOTHING above says any baseline is profitable.")
    print("  * One symbol over one period. Do not generalize from it.")
    print()
    print(f"Recorded {len(report.rows)} model_runs rows (status BASELINE).")
    logger.info(
        "evaluate-baselines: symbol=%s timeframe=%s horizon=%d test_bars=%d",
        report.symbol, report.timeframe, report.horizon, report.test_bars,
    )
    print("EVALUATE-BASELINES COMPLETE")
    return 0


def _run_build_labels(args: argparse.Namespace) -> int:
    """Compute and store prediction labels (targets) for one symbol/timeframe
    over a date range (Phase 13). Labels use FUTURE bars by design; the last
    max(horizon) bars of the range are left unlabeled."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot build labels: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        start_date = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
    except ValueError:
        print("--start-date/--end-date must be YYYY-MM-DD")
        return 1
    if start_date > end_date:
        print("--start-date must not be after --end-date")
        return 1

    if args.symbol not in (config.symbols.indices + config.symbols.stocks):
        print(f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks).")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.features.label_engine import compute_and_store_labels
    except ModuleNotFoundError as exc:
        print(f"Cannot build labels: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.market.timezone)
    start = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=tz)
    end = _dt.datetime.combine(end_date, _dt.time.min, tzinfo=tz) + _dt.timedelta(days=1)

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    print("PSX AI TRADER — BUILD LABELS")
    print("--------------------------------")
    print(f"Symbol:     {args.symbol}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Date range: {start_date} .. {end_date}")
    print(f"Horizons:   {', '.join(str(h) for h in config.labels.horizons)} bar(s)")
    print("NOTE: labels look FORWARD; the final bars of the range stay unlabeled.")

    result = compute_and_store_labels(config, session_factory, args.symbol, args.timeframe, start, end)

    print(f"Written:    {result['written']}")
    print(f"Duplicate:  {result['duplicate']}")
    logger.info(
        "build-labels: symbol=%s timeframe=%s range=%s..%s written=%d duplicate=%d",
        args.symbol, args.timeframe, start_date, end_date, result["written"], result["duplicate"],
    )
    print()
    print("BUILD-LABELS COMPLETE")
    return 0


def _run_build_features(args: argparse.Namespace) -> int:
    """Compute and store stock + market-regime features for one
    symbol/timeframe over a date range (Phase 12)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot build features: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        start_date = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
    except ValueError:
        print("--start-date/--end-date must be YYYY-MM-DD")
        return 1
    if start_date > end_date:
        print("--start-date must not be after --end-date")
        return 1

    if args.symbol not in (config.symbols.indices + config.symbols.stocks):
        print(f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks).")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.features.engine import compute_and_store_features
    except ModuleNotFoundError as exc:
        print(f"Cannot build features: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.market.timezone)
    start = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=tz)
    end = _dt.datetime.combine(end_date, _dt.time.min, tzinfo=tz) + _dt.timedelta(days=1)

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    print("PSX AI TRADER — BUILD FEATURES")
    print("----------------------------------")
    print(f"Symbol:     {args.symbol}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Date range: {start_date} .. {end_date}")
    if args.symbol in config.symbols.indices:
        print("(symbol is an index — only its own regime features will be built)")
    else:
        print(f"(also builds regime features for: {', '.join(config.symbols.indices)})")

    result = compute_and_store_features(config, session_factory, args.symbol, args.timeframe, start, end)

    print(f"Written:    {result['written']}")
    print(f"Duplicate:  {result['duplicate']}")
    logger.info(
        "build-features: symbol=%s timeframe=%s range=%s..%s written=%d duplicate=%d",
        args.symbol, args.timeframe, start_date, end_date, result["written"], result["duplicate"],
    )
    print()
    print("BUILD-FEATURES COMPLETE")
    return 0


def _run_build_indicators(args: argparse.Namespace) -> int:
    """Compute and store technical indicators for one symbol/timeframe over
    a date range (Phase 11)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot build indicators: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        start_date = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
    except ValueError:
        print("--start-date/--end-date must be YYYY-MM-DD")
        return 1
    if start_date > end_date:
        print("--start-date must not be after --end-date")
        return 1

    if args.symbol not in (config.symbols.indices + config.symbols.stocks):
        print(f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks).")
        return 1

    try:
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.indicators.engine import compute_and_store_indicators
    except ModuleNotFoundError as exc:
        print(f"Cannot build indicators: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.market.timezone)
    start = _dt.datetime.combine(start_date, _dt.time.min, tzinfo=tz)
    end = _dt.datetime.combine(end_date, _dt.time.min, tzinfo=tz) + _dt.timedelta(days=1)

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    print("PSX AI TRADER — BUILD INDICATORS")
    print("-----------------------------------")
    print(f"Symbol:     {args.symbol}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Date range: {start_date} .. {end_date}")

    result = compute_and_store_indicators(config, session_factory, args.symbol, args.timeframe, start, end)

    print(f"Written:    {result['written']}")
    print(f"Duplicate:  {result['duplicate']}")
    logger.info(
        "build-indicators: symbol=%s timeframe=%s range=%s..%s written=%d duplicate=%d",
        args.symbol, args.timeframe, start_date, end_date, result["written"], result["duplicate"],
    )
    print()
    print("BUILD-INDICATORS COMPLETE")
    return 0


def _run_build_candles(args: argparse.Namespace) -> int:
    """Build OHLCV candles for one symbol on one trading day (Phase 10)."""
    logger = get_logger("data")
    try:
        config = load_config()
    except ConfigValidationError as exc:
        print("Cannot build candles: configuration is invalid.")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    try:
        import datetime as _dt

        trading_date = _dt.date.fromisoformat(args.date)
    except ValueError:
        print(f"--date must be YYYY-MM-DD (got: {args.date!r})")
        return 1

    if args.symbol in config.symbols.indices:
        symbol_type = "index"
    elif args.symbol in config.symbols.stocks:
        symbol_type = "stock"
    else:
        print(
            f"'{args.symbol}' is not in config/symbols.yaml (indices or stocks). "
            f"Add it there first, or check spelling."
        )
        return 1

    try:
        from src.data.candle_engine import ALL_TIMEFRAMES, build_candles_for_day
        from src.database.engine import build_engine, create_all_tables, get_session_factory
        from src.market.trading_calendar import MarketCalendarValidationError, TradingCalendar
    except ModuleNotFoundError as exc:
        print(f"Cannot build candles: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    try:
        calendar = TradingCalendar.from_config(config)
    except MarketCalendarValidationError as exc:
        print("Market calendar is invalid:\n")
        for err in exc.errors:
            print(f"  - {err}")
        return 1

    timeframes = tuple(args.timeframes.split(",")) if args.timeframes else ALL_TIMEFRAMES
    unknown = set(timeframes) - set(ALL_TIMEFRAMES)
    if unknown:
        print(f"Unknown timeframe(s): {sorted(unknown)}. Valid: {ALL_TIMEFRAMES}")
        return 1

    engine = build_engine()
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    print("PSX AI TRADER — BUILD CANDLES")
    print("--------------------------------")
    report = build_candles_for_day(
        config, session_factory, calendar, args.symbol, symbol_type, trading_date, timeframes=timeframes
    )

    print(f"Symbol:       {report.symbol} ({report.symbol_type})")
    print(f"Trading date: {report.trading_date}")
    print(f"Ticks used:   {report.ticks_used} (REGULAR session, non-INVALID)")
    print()
    for timeframe in timeframes:
        written = report.written.get(timeframe, 0)
        duplicate = report.duplicate.get(timeframe, 0)
        print(f"  {timeframe:6} written={written:<4} duplicate={duplicate}")

    logger.info(
        "build-candles: symbol=%s date=%s ticks_used=%d written=%s duplicate=%s",
        args.symbol, trading_date, report.ticks_used, report.written, report.duplicate,
    )
    print()
    print("BUILD-CANDLES COMPLETE")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "PSX AI Trader CLI (Phase 1-15: init/health/validate-config/init-db/"
            "provider-demo/import-csv/market-status/record-indices/record-stocks/"
            "run-collector/build-candles/build-indicators/build-features/build-labels/"
            "evaluate-baselines/train). See README.md for the full command roadmap."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create required directories and .env file.")
    subparsers.add_parser("health", help="Run Phase 1-15 environment health checks.")
    subparsers.add_parser(
        "validate-config",
        help="Load and validate config/settings.yaml + config/symbols.yaml + .env.",
    )
    subparsers.add_parser(
        "init-db",
        help="Create the SQLite database file and all tables.",
    )
    subparsers.add_parser(
        "provider-demo",
        help="Print one observation per configured symbol from the configured provider (mock or psx).",
    )

    import_csv_parser = subparsers.add_parser(
        "import-csv",
        help="Import a historical CSV file into the database (Section 95).",
    )
    import_csv_parser.add_argument("--file", required=True, help="Path to the CSV file to import.")
    import_csv_parser.add_argument(
        "--symbol", required=True, help="Symbol this file belongs to, e.g. THCCL or KSE100."
    )
    import_csv_parser.add_argument(
        "--symbol-type",
        required=True,
        choices=["stock", "index"],
        help="Whether --symbol is a stock or an index.",
    )
    import_csv_parser.add_argument(
        "--provider-name",
        default="csv",
        help="Label stored in the provider column for imported rows (default: csv).",
    )
    import_csv_parser.add_argument(
        "--mapping",
        default=None,
        help='Path to a JSON file of {"field": "csv_header"} overrides, for files whose '
        "columns cannot be auto-detected.",
    )

    subparsers.add_parser(
        "market-status",
        help="Show the current PSX market session state (open/closed/pre-open/holiday/next open/close).",
    )

    record_indices_parser = subparsers.add_parser(
        "record-indices",
        help="Fetch and store one observation for each configured index symbol (Phase 8).",
    )
    record_indices_parser.add_argument(
        "--force",
        action="store_true",
        help="Record even if the market isn't open (mock provider testing only).",
    )

    record_stocks_parser = subparsers.add_parser(
        "record-stocks",
        help="Fetch and store one observation for each configured stock symbol (Phase 9).",
    )
    record_stocks_parser.add_argument(
        "--force",
        action="store_true",
        help="Record even if the market isn't open (mock provider testing only).",
    )

    run_collector_parser = subparsers.add_parser(
        "run-collector",
        help="Run the continuous collector loop for all configured index and stock symbols (Phase 9).",
    )
    run_collector_parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after this many poll cycles instead of running forever (mainly for testing).",
    )

    build_candles_parser = subparsers.add_parser(
        "build-candles",
        help="Build OHLCV candles for one symbol on one trading day (Phase 10).",
    )
    build_candles_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    build_candles_parser.add_argument("--date", required=True, help="Trading date, YYYY-MM-DD.")
    build_candles_parser.add_argument(
        "--timeframes",
        default=None,
        help="Comma-separated subset of 1m,5m,15m,30m,60m,daily (default: all six).",
    )

    build_indicators_parser = subparsers.add_parser(
        "build-indicators",
        help="Compute and store technical indicators for one symbol/timeframe over a date range (Phase 11).",
    )
    build_indicators_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    build_indicators_parser.add_argument(
        "--timeframe", required=True, help="One of 1m,5m,15m,30m,60m,daily — must have candles already built."
    )
    build_indicators_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD.")
    build_indicators_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")

    build_features_parser = subparsers.add_parser(
        "build-features",
        help="Compute stock + KSE-100/KMI-30 market-regime features for one symbol/timeframe (Phase 12).",
    )
    build_features_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    build_features_parser.add_argument(
        "--timeframe", required=True, help="One of 1m,5m,15m,30m,60m,daily — must have candles already built."
    )
    build_features_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD.")
    build_features_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")

    build_labels_parser = subparsers.add_parser(
        "build-labels",
        help="Compute prediction labels/targets for one symbol/timeframe (Phase 13).",
    )
    build_labels_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    build_labels_parser.add_argument(
        "--timeframe", required=True, help="One of 1m,5m,15m,30m,60m,daily — must have candles already built."
    )
    build_labels_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD.")
    build_labels_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")

    eval_baselines_parser = subparsers.add_parser(
        "evaluate-baselines",
        help="Score the six baseline predictors on one symbol (Phase 14; prediction quality only).",
    )
    eval_baselines_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    eval_baselines_parser.add_argument(
        "--timeframe", required=True, help="One of 1m,5m,15m,30m,60m,daily — must have candles already built."
    )
    eval_baselines_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD.")
    eval_baselines_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    eval_baselines_parser.add_argument(
        "--horizon", required=True, type=int, help="Bars ahead to predict; must be in labels.horizons."
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train the ML models on one symbol and compare them to the baselines (Phase 15; prediction quality only).",
    )
    train_parser.add_argument("--symbol", required=True, help="e.g. KSE100 or THCCL.")
    train_parser.add_argument(
        "--timeframe", required=True, help="One of 1m,5m,15m,30m,60m,daily — must have candles already built "
        "for the symbol and for each index."
    )
    train_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD.")
    train_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    train_parser.add_argument(
        "--horizon", required=True, type=int, help="Bars ahead to predict; must be in labels.horizons."
    )
    train_parser.add_argument(
        "--models", default=None,
        help="Comma-separated subset of logistic,random_forest,gradient_boosting,hist_gradient_boosting "
        "(default: models.enabled in settings.yaml).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return _run_init()
    if args.command == "health":
        return _run_health()
    if args.command == "validate-config":
        return _run_validate_config()
    if args.command == "init-db":
        return _run_init_db()
    if args.command == "provider-demo":
        return _run_provider_demo()
    if args.command == "import-csv":
        return _run_import_csv(args)
    if args.command == "market-status":
        return _run_market_status()
    if args.command == "record-indices":
        return _run_record_indices(args)
    if args.command == "record-stocks":
        return _run_record_stocks(args)
    if args.command == "run-collector":
        return _run_collector(args)
    if args.command == "build-candles":
        return _run_build_candles(args)
    if args.command == "build-indicators":
        return _run_build_indicators(args)
    if args.command == "build-features":
        return _run_build_features(args)
    if args.command == "build-labels":
        return _run_build_labels(args)
    if args.command == "evaluate-baselines":
        return _run_evaluate_baselines(args)
    if args.command == "train":
        return _run_train(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
