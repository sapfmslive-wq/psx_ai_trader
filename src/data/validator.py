"""The data quality engine (Section 13 of the project spec).

Classifies a single incoming observation as one of:

    VALID     — no issues detected
    SUSPECT    — plausible but anomalous; worth flagging, not necessarily wrong
    INVALID     — structurally or logically broken; still recorded, never fabricated-over
    DUPLICATE    — same symbol + timestamp already seen in this session

**Records are never deleted or silently dropped by this engine.** Every
classification is stored (`quality_status`) alongside the reason(s) that
led to it (`quality_reason`) — see `csv_importer.py` for how the two are
threaded into `IndexTick`/`StockTick` rows. Deciding what a downstream
consumer (a candle builder, a feature, a signal) does with a SUSPECT or
INVALID row is explicitly out of scope for this module.

This engine has no database dependency — it operates on the plain
`QualityInput` value object, not on ORM models or a live provider. That
keeps it usable from the CSV importer, a future live collector, and its own
test suite without pulling in SQLAlchemy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from src.config.loader import AppConfig
from src.data.base_provider import IndexObservation, StockObservation

VALID = "VALID"
SUSPECT = "SUSPECT"
INVALID = "INVALID"
DUPLICATE = "DUPLICATE"

# market_status values this engine recognizes as well-formed. Anything else
# present (but non-None) is flagged SUSPECT — not rejected outright, since a
# provider may legitimately introduce a new status we haven't seen before,
# and rejecting the whole tick over a label is disproportionate. A real
# trading-calendar-aware session engine arrives in Phase 8; this is
# deliberately just a sanity check against known vocabulary until then.
KNOWN_MARKET_STATUSES = frozenset(
    {"OPEN", "CLOSED", "PRE_OPEN", "POST_MARKET", "SYNTHETIC_OPEN"}
)


@dataclass(frozen=True)
class QualityInput:
    """Everything the engine needs to evaluate one observation.

    Every field except `symbol` is `Optional` on purpose: a garbled or
    partial source record is exactly the kind of input this engine exists
    to classify, not something callers should have to pre-filter.
    """

    symbol: str
    timestamp: dt.datetime | None
    price: float | None  # `value` for an index, `last_price` for a stock
    open: float | None
    high: float | None
    low: float | None
    volume: float | None
    market_status: str | None


@dataclass(frozen=True)
class QualityResult:
    status: str
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str | None:
        return "; ".join(self.reasons) if self.reasons else None


def from_index_observation(obs: IndexObservation) -> QualityInput:
    return QualityInput(
        symbol=obs.symbol,
        timestamp=obs.timestamp,
        price=obs.value,
        open=None,
        high=obs.high,
        low=obs.low,
        volume=obs.volume,
        market_status=obs.market_status,
    )


def from_stock_observation(obs: StockObservation) -> QualityInput:
    return QualityInput(
        symbol=obs.symbol,
        timestamp=obs.timestamp,
        price=obs.last_price,
        open=obs.open,
        high=obs.high,
        low=obs.low,
        volume=obs.volume,
        market_status=obs.market_status,
    )


class DataQualityEngine:
    """Stateful, per-symbol quality classifier.

    State (last-seen timestamp/price per symbol, and how long a price has
    been unchanged) is only updated after a VALID or SUSPECT result — an
    INVALID observation's price/timestamp is never trusted enough to become
    the new baseline for the next comparison, and a DUPLICATE observation
    carries no new information at all.

    One engine instance should track one coherent stream (e.g. one CSV file
    for one symbol, or one live collector run) — construct a fresh instance
    for a fresh, unrelated stream rather than reusing state across runs.
    """

    def __init__(self, config: AppConfig) -> None:
        self._known_symbols = set(config.symbols.indices) | set(config.symbols.stocks)
        self._max_change_percent = config.quality.max_change_percent
        self._stale_seconds = config.quality.stale_seconds
        self._future_tolerance = dt.timedelta(
            seconds=config.quality.future_timestamp_tolerance_seconds
        )

        self._last_timestamp: dict[str, dt.datetime] = {}
        self._last_price: dict[str, float] = {}
        self._stale_since: dict[str, dt.datetime] = {}

    def evaluate_index(self, obs: IndexObservation) -> QualityResult:
        return self.evaluate(from_index_observation(obs))

    def evaluate_stock(self, obs: StockObservation) -> QualityResult:
        return self.evaluate(from_stock_observation(obs))

    def evaluate(self, qi: QualityInput) -> QualityResult:
        last_timestamp = self._last_timestamp.get(qi.symbol)

        # --- DUPLICATE: cheapest and most definitive check, first. -------
        if qi.timestamp is not None and last_timestamp is not None and qi.timestamp == last_timestamp:
            return QualityResult(status=DUPLICATE, reasons=["duplicate timestamp for symbol (already seen)"])

        # --- INVALID: structural/logical problems. ------------------------
        invalid_reasons: list[str] = []

        if qi.timestamp is None:
            invalid_reasons.append("missing timestamp")

        if qi.price is None or qi.price <= 0:
            invalid_reasons.append(f"non-positive or missing price: {qi.price!r}")

        if qi.high is not None and qi.low is not None and qi.high < qi.low:
            invalid_reasons.append(f"impossible OHLC: high < low ({qi.high} < {qi.low})")
        if qi.high is not None and qi.open is not None and qi.high < qi.open:
            invalid_reasons.append(f"impossible OHLC: high < open ({qi.high} < {qi.open})")
        if qi.low is not None and qi.open is not None and qi.low > qi.open:
            invalid_reasons.append(f"impossible OHLC: low > open ({qi.low} > {qi.open})")

        if qi.volume is not None and qi.volume < 0:
            invalid_reasons.append(f"negative volume: {qi.volume}")

        if qi.symbol not in self._known_symbols:
            invalid_reasons.append(
                f"symbol '{qi.symbol}' is not in the configured universe "
                f"(config/symbols.yaml)"
            )

        if qi.timestamp is not None:
            now = dt.datetime.now(dt.timezone.utc)
            if qi.timestamp > now + self._future_tolerance:
                invalid_reasons.append(
                    f"timestamp {qi.timestamp.isoformat()} is in the future "
                    f"beyond the configured tolerance"
                )
            if last_timestamp is not None and qi.timestamp < last_timestamp:
                invalid_reasons.append(
                    f"out-of-order timestamp ({qi.timestamp.isoformat()} is "
                    f"before the last-seen {last_timestamp.isoformat()})"
                )

        if invalid_reasons:
            # Deliberately do NOT update _last_timestamp/_last_price here —
            # an invalid observation must never become the trusted baseline
            # for the next comparison.
            return QualityResult(status=INVALID, reasons=invalid_reasons)

        # --- SUSPECT: plausible but anomalous. -----------------------------
        suspect_reasons: list[str] = []

        if qi.market_status is not None and qi.market_status not in KNOWN_MARKET_STATUSES:
            suspect_reasons.append(f"unrecognized market_status: {qi.market_status!r}")

        last_price = self._last_price.get(qi.symbol)
        if last_price is not None and last_price != 0:
            pct_change = abs((qi.price - last_price) / last_price * 100)
            if pct_change > self._max_change_percent:
                suspect_reasons.append(
                    f"extreme change: {pct_change:.2f}% "
                    f"(limit {self._max_change_percent}%)"
                )

        if last_price is not None and qi.price == last_price:
            anchor = self._stale_since.get(qi.symbol)
            if anchor is None:
                # First observation where the price has stopped moving:
                # anchor the streak at the previous observation's timestamp,
                # since that's when the price actually stopped changing.
                anchor = last_timestamp if last_timestamp else qi.timestamp
                self._stale_since[qi.symbol] = anchor
            elapsed = (qi.timestamp - anchor).total_seconds()
            if elapsed >= self._stale_seconds:
                suspect_reasons.append(
                    f"stale price: unchanged for {elapsed:.0f}s "
                    f"(limit {self._stale_seconds}s)"
                )
        else:
            self._stale_since.pop(qi.symbol, None)

        # Trustworthy enough to become the new baseline.
        self._last_timestamp[qi.symbol] = qi.timestamp
        self._last_price[qi.symbol] = qi.price

        if suspect_reasons:
            return QualityResult(status=SUSPECT, reasons=suspect_reasons)

        return QualityResult(status=VALID, reasons=[])
