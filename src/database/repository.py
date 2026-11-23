"""Repository helpers: duplicate-safe inserts and simple queries.

`insert_*` functions for IndexTick/StockTick/Candle use SQLite's
`INSERT ... ON CONFLICT DO NOTHING` against each table's unique constraint,
so replaying the same record (a reconnected collector re-sending the last
few seconds of data, a re-imported CSV, etc.) never creates a duplicate row
— it is silently a no-op, and the function tells the caller whether a row
was actually inserted so callers can log/count duplicates if they want to.

`insert_market_tick` and `insert_system_event` are plain inserts: raw
provider payloads and operational events are append-only logs, not
deduplicated at this layer (see models.py for the reasoning).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.database.models import (
    Candle,
    Feature,
    IndexTick,
    Label,
    MarketTick,
    ModelRun,
    Prediction,
    StockTick,
    SystemEvent,
)


def insert_market_tick(
    session: Session,
    *,
    provider: str,
    symbol: str,
    local_timestamp: dt.datetime,
    raw_payload: str,
    source_timestamp: dt.datetime | None = None,
) -> MarketTick:
    """Append a raw provider payload. Never deduplicated — see module docstring."""
    tick = MarketTick(
        provider=provider,
        symbol=symbol,
        local_timestamp=local_timestamp,
        source_timestamp=source_timestamp,
        raw_payload=raw_payload,
        processed=False,
    )
    session.add(tick)
    session.commit()
    return tick


def insert_index_tick(session: Session, **fields) -> bool:
    """Insert an IndexTick, ignoring it if (provider, symbol, timestamp)
    already exists.

    Returns:
        True if a new row was inserted, False if it was a duplicate.
    """
    stmt = (
        sqlite_insert(IndexTick)
        .values(**fields)
        .on_conflict_do_nothing(
            index_elements=["provider", "symbol", "timestamp"]
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_stock_tick(session: Session, **fields) -> bool:
    """Insert a StockTick, ignoring it if (provider, symbol, timestamp)
    already exists.

    Returns:
        True if a new row was inserted, False if it was a duplicate.
    """
    stmt = (
        sqlite_insert(StockTick)
        .values(**fields)
        .on_conflict_do_nothing(
            index_elements=["provider", "symbol", "timestamp"]
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_candle(session: Session, **fields) -> bool:
    """Insert a Candle, ignoring it if (symbol, timeframe, open_time)
    already exists.

    Returns:
        True if a new row was inserted, False if it was a duplicate.
    """
    stmt = (
        sqlite_insert(Candle)
        .values(**fields)
        .on_conflict_do_nothing(
            index_elements=["symbol", "timeframe", "open_time"]
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_system_event(
    session: Session,
    *,
    timestamp: dt.datetime,
    level: str,
    category: str,
    message: str,
    details_json: str | None = None,
) -> SystemEvent:
    """Append an operational event. Never deduplicated, never edited."""
    event_row = SystemEvent(
        timestamp=timestamp,
        level=level,
        category=category,
        message=message,
        details_json=details_json,
    )
    session.add(event_row)
    session.commit()
    return event_row


def get_index_ticks(session: Session, symbol: str, start: dt.datetime, end: dt.datetime):
    """Return non-INVALID IndexTicks for `symbol` in [start, end), ordered
    by timestamp. Used by the candle engine — INVALID ticks are excluded
    here so a derived candle never gets built from data already flagged as
    unreliable (see src/data/candle_engine.py's module docstring)."""
    return (
        session.query(IndexTick)
        .filter(
            IndexTick.symbol == symbol,
            IndexTick.timestamp >= start,
            IndexTick.timestamp < end,
            IndexTick.quality_status != "INVALID",
        )
        .order_by(IndexTick.timestamp)
        .all()
    )


def get_stock_ticks(session: Session, symbol: str, start: dt.datetime, end: dt.datetime):
    """Return non-INVALID StockTicks for `symbol` in [start, end), ordered
    by timestamp. See get_index_ticks for why INVALID is excluded."""
    return (
        session.query(StockTick)
        .filter(
            StockTick.symbol == symbol,
            StockTick.timestamp >= start,
            StockTick.timestamp < end,
            StockTick.quality_status != "INVALID",
        )
        .order_by(StockTick.timestamp)
        .all()
    )


def get_candles(session: Session, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime):
    """Return Candles for `symbol`/`timeframe` in [start, end), ordered by
    open_time. Used by the indicator engine (Phase 11) to build the
    price/volume series it computes indicators from."""
    return (
        session.query(Candle)
        .filter(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
            Candle.open_time >= start,
            Candle.open_time < end,
        )
        .order_by(Candle.open_time)
        .all()
    )


def insert_feature(session: Session, **fields) -> bool:
    """Insert a Feature row, ignoring it if (symbol, timeframe, timestamp,
    feature_name, feature_version) already exists.

    Returns:
        True if a new row was inserted, False if it was a duplicate.
    """
    stmt = (
        sqlite_insert(Feature)
        .values(**fields)
        .on_conflict_do_nothing(
            index_elements=["symbol", "timeframe", "timestamp", "feature_name", "feature_version"]
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_label(session: Session, **fields) -> bool:
    """Insert a Label row, ignoring it if (symbol, timeframe, timestamp,
    label_name, label_version) already exists. Returns True if inserted."""
    stmt = (
        sqlite_insert(Label)
        .values(**fields)
        .on_conflict_do_nothing(
            index_elements=["symbol", "timeframe", "timestamp", "label_name", "label_version"]
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_model_run(session: Session, **fields) -> bool:
    """Insert a ModelRun, ignoring it if `model_id` already exists.
    Returns True if a new row was inserted."""
    stmt = sqlite_insert(ModelRun).values(**fields).on_conflict_do_nothing(index_elements=["model_id"])
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def insert_prediction(session: Session, **fields) -> bool:
    """Insert a Prediction, ignoring it if (symbol, timeframe, timestamp,
    model_version) already exists. Returns True if inserted."""
    stmt = (
        sqlite_insert(Prediction)
        .values(**fields)
        .on_conflict_do_nothing(index_elements=["symbol", "timeframe", "timestamp", "model_version"])
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount > 0


def count_rows(session: Session, model) -> int:  # noqa: ANN001
    """Return the total row count for a given ORM model. Small convenience
    used by health checks and tests; not intended for large tables."""
    return session.query(model).count()
