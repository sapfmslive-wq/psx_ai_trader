"""MockProvider — synthetic market data, no external dependency.

This provider exists specifically so the rest of the system (candles,
indicators, features, backtesting, dashboard) can be built and tested
without any PSX connectivity, per Section 94 of the project spec. It is a
random walk, not a model of PSX — nobody should ever mistake its numbers for
real market behavior. Every observation it produces:

    - carries `is_synthetic=True`
    - carries `market_status="SYNTHETIC_OPEN"` (never a real session status)
    - carries `provider="mock"`

so downstream code and the eventual dashboard can always tell synthetic data
apart from anything real, and never accidentally present it as such
(Section 115: "Never present synthetic data as real PSX data").
"""

from __future__ import annotations

import datetime as dt
import random

from src.data.base_provider import DataProvider, IndexObservation, StockObservation

# Starting values chosen to be roughly plausible in scale for the symbols in
# the default config/symbols.yaml. These are NOT real PSX values — they only
# need to be a believable order of magnitude for a random walk to look like
# a price series during testing.
DEFAULT_INDEX_BASE_VALUES: dict[str, float] = {
    "KSE100": 45000.0,
    "KMI30": 70000.0,
}
DEFAULT_STOCK_BASE_PRICES: dict[str, float] = {
    "THCCL": 80.0,
}
FALLBACK_INDEX_BASE_VALUE = 10000.0
FALLBACK_STOCK_BASE_PRICE = 100.0


class MockProvider(DataProvider):
    """Generates a per-symbol random walk for indices and stocks.

    Args:
        seed: Random seed. Pass the same seed to get an identical sequence
            of observations — used by tests and by anyone who wants a
            reproducible demo run.
        volatility: Standard deviation of the per-call percentage change
            (e.g. 0.0015 = roughly 0.15% typical step).
        index_base_values: Override/extend the default starting index
            values, e.g. {"KSE100": 44000.0}.
        stock_base_prices: Override/extend the default starting stock
            prices, e.g. {"OGDC": 95.0}.
    """

    provider_name = "mock"
    is_synthetic = True

    def __init__(
        self,
        seed: int | None = None,
        volatility: float = 0.0015,
        index_base_values: dict[str, float] | None = None,
        stock_base_prices: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._volatility = volatility

        self._index_base_values = dict(DEFAULT_INDEX_BASE_VALUES)
        if index_base_values:
            self._index_base_values.update(index_base_values)

        self._stock_base_prices = dict(DEFAULT_STOCK_BASE_PRICES)
        if stock_base_prices:
            self._stock_base_prices.update(stock_base_prices)

        self._index_last_value: dict[str, float] = {}
        self._stock_state: dict[str, dict[str, float]] = {}

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _base_index_value(self, symbol: str) -> float:
        return self._index_base_values.get(symbol, FALLBACK_INDEX_BASE_VALUE)

    def _base_stock_price(self, symbol: str) -> float:
        return self._stock_base_prices.get(symbol, FALLBACK_STOCK_BASE_PRICE)

    def get_index_observation(self, symbol: str) -> IndexObservation:
        self._require_connected()

        previous = self._index_last_value.get(symbol, self._base_index_value(symbol))
        pct_change = self._rng.gauss(0, self._volatility)
        new_value = max(previous * (1 + pct_change), 0.01)
        self._index_last_value[symbol] = new_value

        change = new_value - previous
        change_percent = (change / previous * 100) if previous else None
        high = max(new_value, previous)
        low = min(new_value, previous)
        volume = abs(self._rng.gauss(500_000, 150_000))

        now = dt.datetime.now(dt.timezone.utc)
        return IndexObservation(
            provider=self.provider_name,
            symbol=symbol,
            timestamp=now,
            value=round(new_value, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 4) if change_percent is not None else None,
            high=round(high, 2),
            low=round(low, 2),
            volume=round(volume, 0),
            market_status="SYNTHETIC_OPEN",
            source_timestamp=now,
            is_synthetic=True,
        )

    def get_stock_observation(self, symbol: str) -> StockObservation:
        self._require_connected()

        state = self._stock_state.get(symbol)
        if state is None:
            base_price = self._base_stock_price(symbol)
            state = {
                "last": base_price,
                "open": base_price,
                "high": base_price,
                "low": base_price,
                "previous_close": base_price,
                "cumulative_volume": 0.0,
                "cumulative_turnover": 0.0,
            }
            self._stock_state[symbol] = state

        pct_change = self._rng.gauss(0, self._volatility)
        new_last = max(state["last"] * (1 + pct_change), 0.01)
        volume_increment = abs(self._rng.gauss(1000, 300))

        state["high"] = max(state["high"], new_last)
        state["low"] = min(state["low"], new_last)
        state["cumulative_volume"] += volume_increment
        state["cumulative_turnover"] += new_last * volume_increment
        state["last"] = new_last

        spread = max(new_last * 0.0005, 0.01)
        bid = new_last - spread / 2
        ask = new_last + spread / 2
        vwap = (
            state["cumulative_turnover"] / state["cumulative_volume"]
            if state["cumulative_volume"] > 0
            else new_last
        )
        change = new_last - state["previous_close"]
        change_percent = (
            (change / state["previous_close"] * 100) if state["previous_close"] else None
        )

        now = dt.datetime.now(dt.timezone.utc)
        return StockObservation(
            provider=self.provider_name,
            symbol=symbol,
            timestamp=now,
            last_price=round(new_last, 2),
            open=round(state["open"], 2),
            high=round(state["high"], 2),
            low=round(state["low"], 2),
            previous_close=round(state["previous_close"], 2),
            change=round(change, 2),
            change_percent=round(change_percent, 4) if change_percent is not None else None,
            volume=round(state["cumulative_volume"], 0),
            bid=round(bid, 2),
            bid_volume=round(abs(self._rng.gauss(500, 150)), 0),
            ask=round(ask, 2),
            ask_volume=round(abs(self._rng.gauss(500, 150)), 0),
            turnover=round(state["cumulative_turnover"], 2),
            vwap=round(vwap, 2),
            market_status="SYNTHETIC_OPEN",
            source_timestamp=now,
            is_synthetic=True,
        )
