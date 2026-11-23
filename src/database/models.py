"""SQLAlchemy ORM models.

Table groups, matching the project's DATA / ANALYSIS / PREDICTION / SIGNAL /
BACKTEST / PAPER TRADE / LIVE TRADE separation:

    RAW / DATA:       MarketTick, IndexTick, StockTick
    DERIVED DATA:     Candle
    ANALYSIS:         Feature
    PREDICTION:       Prediction
    SIGNAL:           Signal
    ORDER/EXECUTION:  Order, Fill, Position, PortfolioSnapshot
    RESEARCH:         ModelRun, BacktestRun
    OPERATIONS:       SystemEvent

Every table that can plausibly receive the same event twice (replayed feed,
reconnected collector, re-imported CSV) has an explicit UniqueConstraint so
duplicate-safe inserts (see repository.py) actually mean something at the
database level, not just in application code.

`quality_status` columns use the vocabulary from the data-quality engine
(Phase 6): VALID, SUSPECT, INVALID, DUPLICATE. Phase 3 only defines the
column; Phase 6 is what actually populates it with real logic.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# RAW DATA (immutable — never overwritten, never deduplicated at this layer;
# deduplication into IndexTick/StockTick happens downstream in Phase 6)
# ---------------------------------------------------------------------------


class MarketTick(Base):
    """Raw, as-received provider payload, before typing/validation.

    This is the layer that satisfies the "never destroy original data" rule
    (Section 11 of the project spec). Nothing here is ever deleted or edited
    by later phases.
    """

    __tablename__ = "market_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    source_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    local_timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)  # original JSON/text as received
    processed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    processing_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        Index("ix_market_ticks_symbol_local_ts", "symbol", "local_timestamp"),
        Index("ix_market_ticks_processed", "processed"),
    )


# ---------------------------------------------------------------------------
# CLEANED / TYPED DATA
# ---------------------------------------------------------------------------


class IndexTick(Base):
    """A single observation of KSE-100 or KMI-30 (or any future index)."""

    __tablename__ = "index_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g. KSE100, KMI30
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    change: Mapped[float | None] = mapped_column(Float)
    change_percent: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    market_status: Mapped[str | None] = mapped_column(String(32))
    source_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    local_timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False, default="VALID")
    quality_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint("provider", "symbol", "timestamp", name="uq_index_ticks_provider_symbol_ts"),
        Index("ix_index_ticks_symbol_timestamp", "symbol", "timestamp"),
    )


class StockTick(Base):
    """A single observation of a configured PSX stock symbol."""

    __tablename__ = "stock_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_price: Mapped[float | None] = mapped_column(Float)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    previous_close: Mapped[float | None] = mapped_column(Float)
    change: Mapped[float | None] = mapped_column(Float)
    change_percent: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float)
    bid_volume: Mapped[float | None] = mapped_column(Float)
    ask: Mapped[float | None] = mapped_column(Float)
    ask_volume: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    vwap: Mapped[float | None] = mapped_column(Float)
    market_status: Mapped[str | None] = mapped_column(String(32))
    source_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    local_timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False, default="VALID")
    quality_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint("provider", "symbol", "timestamp", name="uq_stock_ticks_provider_symbol_ts"),
        Index("ix_stock_ticks_symbol_timestamp", "symbol", "timestamp"),
    )


# ---------------------------------------------------------------------------
# DERIVED DATA
# ---------------------------------------------------------------------------


class Candle(Base):
    """OHLCV candle for a symbol/timeframe (Phase 10 builds these)."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)  # 1m/5m/15m/30m/60m/daily
    open_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    trade_count: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "open_time", name="uq_candles_symbol_timeframe_open"),
        Index("ix_candles_symbol_timeframe_open", "symbol", "timeframe", "open_time"),
    )


# ---------------------------------------------------------------------------
# ANALYSIS / PREDICTION / SIGNAL
# ---------------------------------------------------------------------------


class Feature(Base):
    """A single named feature value at a point in time (Phase 12/13).

    Stored in long format (one row per feature per timestamp) rather than
    one wide table, so adding a new feature never requires a schema
    migration — it is just a new `feature_name` value.
    """

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Float)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "timestamp", "feature_name", "feature_version",
            name="uq_features_identity",
        ),
        Index("ix_features_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
    )


class Label(Base):
    """A prediction target for a symbol/timeframe/timestamp (Phase 13).

    Deliberately a separate table from `Feature`: a label is built from
    data AFTER its timestamp, a feature only from data at or before it.
    Keeping them apart means a model-building query for features cannot
    return a label by accident (Sections 21/101).
    """

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label_name: Mapped[str] = mapped_column(String(64), nullable=False)
    label_value: Mapped[float | None] = mapped_column(Float)
    label_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "timestamp", "label_name", "label_version",
            name="uq_labels_identity",
        ),
        Index("ix_labels_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
    )


class Prediction(Base):
    """Model output for a symbol/timeframe/timestamp (Phase 15)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prediction: Mapped[str] = mapped_column(String(16), nullable=False)  # UP/DOWN/NEUTRAL
    probability_up: Mapped[float | None] = mapped_column(Float)
    probability_down: Mapped[float | None] = mapped_column(Float)
    probability_neutral: Mapped[float | None] = mapped_column(Float)
    expected_return: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "timestamp", "model_version",
            name="uq_predictions_identity",
        ),
        Index("ix_predictions_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
    )


class Signal(Base):
    """A research signal emitted by the signal engine (Phase 18)."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    price: Mapped[float | None] = mapped_column(Float)
    market_regime: Mapped[str | None] = mapped_column(String(32))
    model_version: Mapped[str | None] = mapped_column(String(64))
    prediction: Mapped[str | None] = mapped_column(String(16))
    probability: Mapped[float | None] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)  # STRONG_BUY..STRONG_SELL
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    risk_reward: Mapped[float | None] = mapped_column(Float)
    position_size: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp"),
    )


# ---------------------------------------------------------------------------
# ORDERS / EXECUTION (paper today, live in Phase 26 behind the same schema)
# ---------------------------------------------------------------------------


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"))
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY/SELL
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # MARKET/LIMIT/STOP
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    mode: Mapped[str] = mapped_column(String(8), nullable=False)  # PAPER/LIVE
    reason: Mapped[str | None] = mapped_column(Text)

    fills: Mapped[list["Fill"]] = relationship(back_populates="order")

    __table_args__ = (
        Index("ix_orders_symbol_created_at", "symbol", "created_at"),
    )


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    filled_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    order: Mapped["Order"] = relationship(back_populates="fills")

    __table_args__ = (
        Index("ix_fills_order_id", "order_id"),
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    mode: Mapped[str] = mapped_column(String(8), nullable=False)  # PAPER/LIVE
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    average_entry_price: Mapped[float | None] = mapped_column(Float)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="CURRENT_TIMESTAMP",
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )

    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
    )


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mode: Mapped[str] = mapped_column(String(8), nullable=False)  # PAPER/LIVE
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        UniqueConstraint("timestamp", "mode", name="uq_portfolio_snapshots_timestamp_mode"),
        Index("ix_portfolio_snapshots_timestamp", "timestamp"),
    )


# ---------------------------------------------------------------------------
# RESEARCH TRACKING
# ---------------------------------------------------------------------------


class ModelRun(Base):
    """One trained model's full provenance (Phase 14/15/34/59/100)."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    training_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    training_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    validation_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    validation_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    test_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    test_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    hyperparameters_json: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    feature_list_json: Mapped[str | None] = mapped_column(Text)
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    random_seed: Mapped[int | None] = mapped_column(Integer)
    python_version: Mapped[str | None] = mapped_column(String(32))
    package_versions_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="TRAINED")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        Index("ix_model_runs_symbol_timeframe", "symbol", "timeframe"),
    )


class BacktestRun(Base):
    """One backtest execution's full provenance (Phase 16/17/80/100)."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    parameters_json: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    benchmark_metrics_json: Mapped[str | None] = mapped_column(Text)
    costs_config_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="COMPLETED")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        Index("ix_backtest_runs_symbol_timeframe", "symbol", "timeframe"),
    )


# ---------------------------------------------------------------------------
# OPERATIONS
# ---------------------------------------------------------------------------


class SystemEvent(Base):
    """Application lifecycle / operational events (outages, restarts, etc.).

    This is where Section 71's "record connection_lost / connection_restored"
    requirement will be satisfied once the collector exists (Phase 15+... /
    actually Phase 15 collector work happens Phase 4-9). Deliberately append-
    only: never updated, never deleted by application code.
    """

    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)  # INFO/WARNING/ERROR/CRITICAL
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    __table_args__ = (
        Index("ix_system_events_timestamp", "timestamp"),
        Index("ix_system_events_category", "category"),
    )


ALL_TABLES: tuple[type[Base], ...] = (
    MarketTick,
    IndexTick,
    StockTick,
    Candle,
    Feature,
    Label,
    Prediction,
    Signal,
    Order,
    Fill,
    Position,
    PortfolioSnapshot,
    ModelRun,
    BacktestRun,
    SystemEvent,
)
