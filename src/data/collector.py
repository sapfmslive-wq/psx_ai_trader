"""Single-shot recording (Phase 8/9): wires a `DataProvider`, the
`DataQualityEngine`, and the database together for the configured index and
stock symbols, gated by the `TradingCalendar`.

Three entry points:

    collect_indices_once()  - KSE-100/KMI-30 only (Phase 8)
    collect_stocks_once()    - configured stocks only (Phase 9)
    collect_once()            - both, in a single market-open check and a
                                 single provider connection (what the live
                                 collector in live_collector.py actually
                                 calls every cycle)

These are deliberately single-shot, not a polling loop — see
`live_collector.py` for the continuous, reconnect/backoff/heartbeat-aware
loop built around `collect_once()` (Section 15 of the project spec).

By default every function here refuses to record when the market isn't
open — the whole point of the trading calendar is to stop the system from
treating a CLOSED/HOLIDAY/WEEKEND read as a legitimate new observation.
`force=True` overrides this for testing (e.g. with the mock provider, at
any time of day); it is not meant for real provider use.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass, field

from src.config.loader import AppConfig
from src.data.base_provider import DataProvider
from src.market.trading_calendar import TradingCalendar


@dataclass
class SymbolOutcome:
    symbol: str
    quality_status: str
    quality_reason: str | None
    inserted: bool  # False means it was a duplicate (already in the DB)
    kind: str = "index"  # "index" or "stock"


@dataclass
class CollectionReport:
    skipped: bool
    skip_reason: str | None = None
    session: str | None = None
    outcomes: list[SymbolOutcome] = field(default_factory=list)


def _record_index_symbol(session_factory, provider, quality_engine, symbol: str, now: dt.datetime) -> SymbolOutcome:  # noqa: ANN001
    from src.database.repository import insert_index_tick, insert_market_tick

    observation = provider.get_index_observation(symbol)
    result = quality_engine.evaluate_index(observation)

    with session_factory() as db_session:
        insert_market_tick(
            db_session,
            provider=observation.provider,
            symbol=symbol,
            local_timestamp=now,
            source_timestamp=observation.source_timestamp,
            raw_payload=json.dumps(dataclasses.asdict(observation), default=str),
        )
        inserted = insert_index_tick(
            db_session,
            provider=observation.provider,
            symbol=symbol,
            timestamp=observation.timestamp,
            value=observation.value,
            change=observation.change,
            change_percent=observation.change_percent,
            high=observation.high,
            low=observation.low,
            volume=observation.volume,
            market_status=observation.market_status,
            source_timestamp=observation.source_timestamp,
            local_timestamp=now,
            quality_status=result.status,
            quality_reason=result.reason_text,
        )

    return SymbolOutcome(
        symbol=symbol, kind="index", quality_status=result.status,
        quality_reason=result.reason_text, inserted=inserted,
    )


def _record_stock_symbol(session_factory, provider, quality_engine, symbol: str, now: dt.datetime) -> SymbolOutcome:  # noqa: ANN001
    from src.database.repository import insert_market_tick, insert_stock_tick

    observation = provider.get_stock_observation(symbol)
    result = quality_engine.evaluate_stock(observation)

    with session_factory() as db_session:
        insert_market_tick(
            db_session,
            provider=observation.provider,
            symbol=symbol,
            local_timestamp=now,
            source_timestamp=observation.source_timestamp,
            raw_payload=json.dumps(dataclasses.asdict(observation), default=str),
        )
        inserted = insert_stock_tick(
            db_session,
            provider=observation.provider,
            symbol=symbol,
            timestamp=observation.timestamp,
            last_price=observation.last_price,
            open=observation.open,
            high=observation.high,
            low=observation.low,
            previous_close=observation.previous_close,
            change=observation.change,
            change_percent=observation.change_percent,
            volume=observation.volume,
            bid=observation.bid,
            bid_volume=observation.bid_volume,
            ask=observation.ask,
            ask_volume=observation.ask_volume,
            turnover=observation.turnover,
            vwap=observation.vwap,
            market_status=observation.market_status,
            source_timestamp=observation.source_timestamp,
            local_timestamp=now,
            quality_status=result.status,
            quality_reason=result.reason_text,
        )

    return SymbolOutcome(
        symbol=symbol, kind="stock", quality_status=result.status,
        quality_reason=result.reason_text, inserted=inserted,
    )


def _check_market_open(calendar: TradingCalendar, force: bool, at: dt.datetime | None) -> tuple[bool, str]:
    session = calendar.current_session(at)
    if not force and not calendar.is_market_open(at):
        return False, session
    return True, session


def collect_indices_once(
    config: AppConfig,
    session_factory,  # noqa: ANN001
    provider: DataProvider,
    calendar: TradingCalendar,
    quality_engine,  # noqa: ANN001 — src.data.validator.DataQualityEngine
    force: bool = False,
    at: dt.datetime | None = None,
) -> CollectionReport:
    """Fetch and store one observation for every configured index symbol.

    See the module docstring for `force`/gating behavior. `at` overrides
    the reference time used for the market-open check — omit it to use the
    real wall clock (the live collector always passes it explicitly, for
    determinism and testability; one-off CLI use leaves it as None).
    """
    should_run, session = _check_market_open(calendar, force, at)
    if not should_run:
        return CollectionReport(skipped=True, skip_reason=f"market not open (session={session})", session=session)

    now = dt.datetime.now(dt.timezone.utc)
    outcomes = []
    with provider:
        for symbol in config.symbols.indices:
            outcomes.append(_record_index_symbol(session_factory, provider, quality_engine, symbol, now))

    return CollectionReport(skipped=False, session=session, outcomes=outcomes)


def collect_stocks_once(
    config: AppConfig,
    session_factory,  # noqa: ANN001
    provider: DataProvider,
    calendar: TradingCalendar,
    quality_engine,  # noqa: ANN001
    force: bool = False,
    at: dt.datetime | None = None,
) -> CollectionReport:
    """Fetch and store one observation for every configured stock symbol.

    Same gating/force/at semantics as `collect_indices_once`.
    """
    should_run, session = _check_market_open(calendar, force, at)
    if not should_run:
        return CollectionReport(skipped=True, skip_reason=f"market not open (session={session})", session=session)

    now = dt.datetime.now(dt.timezone.utc)
    outcomes = []
    with provider:
        for symbol in config.symbols.stocks:
            outcomes.append(_record_stock_symbol(session_factory, provider, quality_engine, symbol, now))

    return CollectionReport(skipped=False, session=session, outcomes=outcomes)


def collect_once(
    config: AppConfig,
    session_factory,  # noqa: ANN001
    provider: DataProvider,
    calendar: TradingCalendar,
    quality_engine,  # noqa: ANN001
    force: bool = False,
    at: dt.datetime | None = None,
) -> CollectionReport:
    """Fetch and store one observation for every configured index AND stock
    symbol, in a single market-open check and a single provider connection.

    This is what `live_collector.py`'s continuous loop calls every cycle —
    one connect, one calendar check, indices then stocks, one disconnect.
    """
    should_run, session = _check_market_open(calendar, force, at)
    if not should_run:
        return CollectionReport(skipped=True, skip_reason=f"market not open (session={session})", session=session)

    now = dt.datetime.now(dt.timezone.utc)
    outcomes = []
    with provider:
        for symbol in config.symbols.indices:
            outcomes.append(_record_index_symbol(session_factory, provider, quality_engine, symbol, now))
        for symbol in config.symbols.stocks:
            outcomes.append(_record_stock_symbol(session_factory, provider, quality_engine, symbol, now))

    return CollectionReport(skipped=False, session=session, outcomes=outcomes)
