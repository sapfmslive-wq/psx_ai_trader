"""The data provider abstraction.

Every concrete provider (mock, CSV, PSX, broker) implements this interface.
Nothing outside `src/data/` should import a concrete provider class directly
in application code that isn't explicitly choosing/wiring up a provider —
everything else should type-hint against `DataProvider`.

`IndexObservation` and `StockObservation` are provider-agnostic value
objects, independent of the SQLAlchemy models in `src/database/models.py`.
Keeping them separate means a provider never needs to know anything about
how (or whether) its output gets persisted, and the database layer never
needs to know anything about where data came from.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderNotConnectedError(RuntimeError):
    """Raised when a provider method that requires a connection is called
    before `connect()` (or after `disconnect()`)."""


@dataclass(frozen=True)
class IndexObservation:
    """One observation of an index (KSE-100, KMI-30, ...).

    Fields mirror Section 8 of the project spec. Any field the provider
    cannot supply must be `None` — never fabricated.
    """

    provider: str
    symbol: str
    timestamp: dt.datetime
    value: float | None
    change: float | None
    change_percent: float | None
    high: float | None
    low: float | None
    volume: float | None
    market_status: str | None
    source_timestamp: dt.datetime | None
    is_synthetic: bool


@dataclass(frozen=True)
class StockObservation:
    """One observation of a stock symbol.

    Fields mirror Section 9 of the project spec. Any field the provider
    cannot supply must be `None` — never fabricated.
    """

    provider: str
    symbol: str
    timestamp: dt.datetime
    last_price: float | None
    open: float | None
    high: float | None
    low: float | None
    previous_close: float | None
    change: float | None
    change_percent: float | None
    volume: float | None
    bid: float | None
    bid_volume: float | None
    ask: float | None
    ask_volume: float | None
    turnover: float | None
    vwap: float | None
    market_status: str | None
    source_timestamp: dt.datetime | None
    is_synthetic: bool


class DataProvider(ABC):
    """Abstract base class every concrete data provider implements.

    Usage pattern (also works as a context manager):

        provider = MockProvider()
        with provider:
            obs = provider.get_index_observation("KSE100")

    Concrete providers must never silently invent a value that the
    underlying source did not actually supply — return `None` for that
    field instead.
    """

    #: Short machine-readable name, e.g. "mock", "csv", "psx", "broker".
    #: Stored verbatim in `MarketTick.provider` / `IndexTick.provider` / etc.
    provider_name: str = "unknown"

    #: True if this provider generates or replays non-live data. Every
    #: observation from a synthetic provider must have `is_synthetic=True`,
    #: and any code displaying such data must label it SYNTHETIC (Section 115).
    is_synthetic: bool = False

    def __init__(self) -> None:
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish whatever connection/session the provider needs.

        Must be idempotent: calling `connect()` twice in a row is not an
        error.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection/session. Must be idempotent and must
        never raise if already disconnected."""

    def is_connected(self) -> bool:
        return self._connected

    def _require_connected(self) -> None:
        if not self._connected:
            raise ProviderNotConnectedError(
                f"{self.provider_name} provider is not connected — call connect() first."
            )

    @abstractmethod
    def get_index_observation(self, symbol: str) -> IndexObservation:
        """Return the latest available observation for an index symbol."""

    @abstractmethod
    def get_stock_observation(self, symbol: str) -> StockObservation:
        """Return the latest available observation for a stock symbol."""

    def __enter__(self) -> "DataProvider":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.disconnect()
