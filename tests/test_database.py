"""Tests for src.database (Phase 3)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.engine import create_all_tables, create_engine_for_path, get_session_factory
from src.database.models import ALL_TABLES, Fill, MarketTick, ModelRun, Order, SystemEvent
from src.database.repository import (
    count_rows,
    insert_candle,
    insert_index_tick,
    insert_market_tick,
    insert_stock_tick,
    insert_system_event,
)

UTC = dt.timezone.utc


@pytest.fixture()
def session(tmp_path) -> Session:
    engine = create_engine_for_path(tmp_path / "test.db", echo=False)

    # Mirror the FK-enforcement pragma that engine.build_engine() sets up in
    # the real application, so this test suite exercises the same behavior.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    create_all_tables(engine)
    factory = get_session_factory(engine)
    sess = factory()
    yield sess
    sess.close()


def test_create_all_tables_creates_every_expected_table(session: Session):
    inspector = inspect(session.get_bind())
    existing_tables = set(inspector.get_table_names())
    expected_tables = {model.__tablename__ for model in ALL_TABLES}
    assert expected_tables.issubset(existing_tables)


def test_index_tick_duplicate_insert_is_ignored(session: Session):
    from src.database.models import IndexTick

    ts = dt.datetime(2026, 9, 14, 9, 32, tzinfo=UTC)
    fields = dict(
        provider="mock",
        symbol="KSE100",
        timestamp=ts,
        value=45000.0,
        local_timestamp=ts,
    )

    first = insert_index_tick(session, **fields)
    second = insert_index_tick(session, **fields)

    assert first is True
    assert second is False
    assert count_rows(session, IndexTick) == 1


def test_stock_tick_duplicate_insert_is_ignored(session: Session):
    from src.database.models import StockTick

    ts = dt.datetime(2026, 9, 14, 9, 33, tzinfo=UTC)
    fields = dict(
        provider="mock",
        symbol="THCCL",
        timestamp=ts,
        last_price=80.5,
        local_timestamp=ts,
    )

    first = insert_stock_tick(session, **fields)
    second = insert_stock_tick(session, **fields)

    assert first is True
    assert second is False
    assert count_rows(session, StockTick) == 1


def test_candle_duplicate_insert_is_ignored(session: Session):
    from src.database.models import Candle

    open_time = dt.datetime(2026, 9, 14, 9, 30, tzinfo=UTC)
    close_time = dt.datetime(2026, 9, 14, 9, 31, tzinfo=UTC)
    fields = dict(
        symbol="THCCL",
        timeframe="1m",
        open_time=open_time,
        close_time=close_time,
        open=80.0,
        high=80.6,
        low=79.9,
        close=80.3,
        volume=12000,
        provider="mock",
        is_synthetic=True,
    )

    first = insert_candle(session, **fields)
    second = insert_candle(session, **fields)

    assert first is True
    assert second is False
    assert count_rows(session, Candle) == 1


def test_market_tick_is_never_deduplicated(session: Session):
    ts = dt.datetime(2026, 9, 14, 9, 34, tzinfo=UTC)
    for _ in range(2):
        insert_market_tick(
            session,
            provider="mock",
            symbol="THCCL",
            local_timestamp=ts,
            raw_payload='{"last_price": 80.5}',
        )
    assert count_rows(session, MarketTick) == 2


def test_system_event_is_never_deduplicated(session: Session):
    ts = dt.datetime(2026, 9, 14, 9, 35, tzinfo=UTC)
    for _ in range(2):
        insert_system_event(
            session,
            timestamp=ts,
            level="INFO",
            category="system",
            message="init complete",
        )
    assert count_rows(session, SystemEvent) == 2


def test_foreign_key_is_enforced_between_fill_and_order(session: Session):
    bad_fill = Fill(
        order_id=999999,  # does not exist
        filled_at=dt.datetime(2026, 9, 14, 9, 36, tzinfo=UTC),
        price=80.0,
        quantity=10,
    )
    session.add(bad_fill)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_fill_succeeds_against_a_real_order(session: Session):
    order = Order(
        symbol="THCCL",
        side="BUY",
        order_type="MARKET",
        quantity=10,
        status="FILLED",
        mode="PAPER",
    )
    session.add(order)
    session.commit()

    fill = Fill(
        order_id=order.id,
        filled_at=dt.datetime(2026, 9, 14, 9, 37, tzinfo=UTC),
        price=80.1,
        quantity=10,
        fees=1.5,
        slippage=0.05,
    )
    session.add(fill)
    session.commit()

    assert count_rows(session, Fill) == 1
    assert order.fills[0].price == 80.1


def test_model_run_model_id_must_be_unique(session: Session):
    run_a = ModelRun(
        model_id="THCCL_15M_RF_20260914_001",
        symbol="THCCL",
        timeframe="15m",
        model_type="random_forest",
    )
    session.add(run_a)
    session.commit()

    run_b = ModelRun(
        model_id="THCCL_15M_RF_20260914_001",  # duplicate on purpose
        symbol="THCCL",
        timeframe="15m",
        model_type="random_forest",
    )
    session.add(run_b)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
