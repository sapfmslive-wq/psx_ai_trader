"""Candle engine (Phase 10, Section 17 of the project spec).

Converts recorded ticks into OHLCV candles at 1m/5m/15m/30m/60m/daily
timeframes.

Design decisions, stated up front because they materially affect what a
candle means:

- **Only REGULAR-session ticks are used.** Pre-open, pre-open-break, and
  post-close observations are excluded from every candle (both price and
  volume). Candles represent continuous-trading price action; pre-open/
  post-close auction-style prices behave differently and mixing them in
  would silently distort OHLC. (A dedicated opening-auction analysis, if
  ever built, is Section 50's job, not this one's.)
- **Only non-INVALID ticks are used.** A tick flagged INVALID by the
  quality engine (Section 13) — bad price, impossible OHLC, negative
  volume, unknown symbol — is preserved in the database for audit but
  never allowed to distort a derived candle. SUSPECT ticks ARE included
  (they're a real, if flagged-unusual, observation, not a known-bad one).
- **Volume convention is a single global config setting**
  (`candles.volume_convention`), not inferred per-tick. Section 17
  explicitly requires distinguishing per-tick volume from a reported
  cumulative running total — this project does not have a way to detect
  which one a given feed uses automatically, so the person configuring it
  states it explicitly. See `config/settings.yaml` for the two options.
- **Known limitation on cumulative volume**: for the first REGULAR-session
  tick of a day, this engine assumes the cumulative counter's value AT
  that tick is entirely attributable to that tick's candle (baseline 0).
  If a real feed's cumulative counter already included pre-open volume by
  that point, the first candle of the day will overstate volume. There is
  no field in the current schema to separate that out; this is a stated
  limitation, not a hidden bug.

The bucketing/aggregation math (`build_intraday_candles`,
`build_daily_candle`) is pure — no database, no config object, just plain
tick-like inputs — so it is fully unit-testable without SQLAlchemy. The
orchestration function (`build_candles_for_day`) is what actually reads
ticks from and writes candles to the database.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
}
ALL_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "60m", "daily")


@dataclass(frozen=True)
class TickLike:
    """The minimal shape the candle engine needs from a tick, independent
    of whether it came from an IndexTick or a StockTick row."""

    timestamp: dt.datetime
    price: float
    volume: float | None
    provider: str


@dataclass(frozen=True)
class CandleData:
    symbol: str
    timeframe: str
    open_time: dt.datetime
    close_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    trade_count: int
    provider: str
    is_synthetic: bool


def _bucket_start(timestamp: dt.datetime, timeframe_seconds: int, tz: ZoneInfo) -> dt.datetime:
    local = timestamp.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_midnight = (local - midnight).total_seconds()
    bucket_index = int(seconds_since_midnight // timeframe_seconds)
    return midnight + dt.timedelta(seconds=bucket_index * timeframe_seconds)


def _dominant_provider(ticks: list[TickLike]) -> str:
    """The provider of the last tick in the group — matches the 'close'
    convention (the most recent observation wins) and avoids ambiguity if
    a bucket somehow mixed providers."""
    return ticks[-1].provider


def _aggregate(ticks: list[TickLike], volume_convention: str, previous_cumulative: float) -> tuple[float, float, float, float, float | None, float]:
    """Returns (open, high, low, close, volume, new_running_cumulative).

    `previous_cumulative` is only meaningful for volume_convention ==
    "cumulative"; pass 0.0 for the first group of a day.
    """
    prices = [t.price for t in ticks]
    open_, close = prices[0], prices[-1]
    high, low = max(prices), min(prices)

    if volume_convention == "cumulative":
        last_with_volume = next((t.volume for t in reversed(ticks) if t.volume is not None), None)
        if last_with_volume is None:
            return open_, high, low, close, None, previous_cumulative
        delta = last_with_volume - previous_cumulative
        if delta < 0:
            # The counter went backwards — a reset or data anomaly we
            # can't interpret. Don't fabricate a volume; don't crash either.
            return open_, high, low, close, None, previous_cumulative
        return open_, high, low, close, delta, last_with_volume
    else:  # per_tick
        volumes = [t.volume for t in ticks if t.volume is not None]
        volume = sum(volumes) if volumes else None
        return open_, high, low, close, volume, previous_cumulative


def build_intraday_candles(
    ticks: list[TickLike],
    symbol: str,
    timeframe: str,
    tz: ZoneInfo,
    volume_convention: str,
) -> list[CandleData]:
    """Build candles for one intraday timeframe from an already-filtered,
    time-ordered list of ticks (REGULAR session only, non-INVALID only —
    filtering is the orchestration layer's job, not this function's).
    """
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported intraday timeframe: {timeframe!r}")
    if not ticks:
        return []

    seconds = TIMEFRAME_SECONDS[timeframe]
    groups: dict[dt.datetime, list[TickLike]] = {}
    for tick in ticks:
        start = _bucket_start(tick.timestamp, seconds, tz)
        groups.setdefault(start, []).append(tick)

    candles = []
    running_cumulative = 0.0
    for start in sorted(groups):
        group = groups[start]
        open_, high, low, close, volume, running_cumulative = _aggregate(
            group, volume_convention, running_cumulative
        )
        candles.append(
            CandleData(
                symbol=symbol,
                timeframe=timeframe,
                open_time=start,
                close_time=start + dt.timedelta(seconds=seconds),
                open=open_, high=high, low=low, close=close,
                volume=volume,
                trade_count=len(group),
                provider=_dominant_provider(group),
                is_synthetic=(_dominant_provider(group) == "mock"),
            )
        )
    return candles


def build_daily_candle(
    ticks: list[TickLike],
    symbol: str,
    open_time: dt.datetime,
    close_time: dt.datetime,
    volume_convention: str,
) -> CandleData | None:
    """Build the single daily candle from an already-filtered, time-ordered
    list of ticks for one trading day. Returns None if there are no ticks
    (e.g. a holiday, or a symbol with no observations that day).

    `open_time`/`close_time` come from the trading calendar's actual
    session bounds for the day (first session's open .. last session's
    close — meaningful on a Friday with two sessions), not midnight.
    """
    if not ticks:
        return None

    open_, high, low, close, volume, _ = _aggregate(ticks, volume_convention, previous_cumulative=0.0)
    return CandleData(
        symbol=symbol,
        timeframe="daily",
        open_time=open_time,
        close_time=close_time,
        open=open_, high=high, low=low, close=close,
        volume=volume,
        trade_count=len(ticks),
        provider=_dominant_provider(ticks),
        is_synthetic=(_dominant_provider(ticks) == "mock"),
    )


def filter_regular_session_ticks(ticks: list[TickLike], calendar, at_getter=None) -> list[TickLike]:  # noqa: ANN001
    """Keep only ticks observed during the REGULAR session.

    `calendar` is a TradingCalendar. Ticks are assumed already filtered to
    non-INVALID and sorted by timestamp by the caller (the orchestration
    layer, which reads them from the database that way).
    """
    return [t for t in ticks if calendar.current_session(t.timestamp) == "REGULAR"]


@dataclass
class CandleBuildReport:
    symbol: str
    symbol_type: str
    trading_date: dt.date
    ticks_used: int = 0
    written: dict[str, int] | None = None
    duplicate: dict[str, int] | None = None

    def __post_init__(self):
        if self.written is None:
            self.written = {}
        if self.duplicate is None:
            self.duplicate = {}

    def record(self, timeframe: str, inserted: bool) -> None:
        if inserted:
            self.written[timeframe] = self.written.get(timeframe, 0) + 1
        else:
            self.duplicate[timeframe] = self.duplicate.get(timeframe, 0) + 1


def build_candles_for_day(
    config,  # noqa: ANN001 — src.config.loader.AppConfig
    session_factory,  # noqa: ANN001
    calendar,  # noqa: ANN001 — src.market.trading_calendar.TradingCalendar
    symbol: str,
    symbol_type: str,
    trading_date: dt.date,
    timeframes: tuple[str, ...] | None = None,
) -> CandleBuildReport:
    """Read one trading day's ticks for `symbol` and write candles for
    every requested timeframe (default: all six). This is the only
    function in this module that touches the database — everything it
    delegates to (`filter_regular_session_ticks`, `build_intraday_candles`,
    `build_daily_candle`) is pure and independently tested.
    """
    if symbol_type not in ("index", "stock"):
        raise ValueError(f"symbol_type must be 'index' or 'stock' (got: {symbol_type!r})")

    from src.database.repository import get_index_ticks, get_stock_ticks, insert_candle

    timeframes = timeframes or ALL_TIMEFRAMES
    tz = ZoneInfo(config.market.timezone)
    day_start = dt.datetime.combine(trading_date, dt.time.min, tzinfo=tz)
    day_end = day_start + dt.timedelta(days=1)

    with session_factory() as session:
        if symbol_type == "index":
            raw_ticks = get_index_ticks(session, symbol, day_start, day_end)
            price_of = lambda t: t.value  # noqa: E731
        else:
            raw_ticks = get_stock_ticks(session, symbol, day_start, day_end)
            price_of = lambda t: t.last_price  # noqa: E731

        tick_likes = [
            TickLike(timestamp=t.timestamp, price=price_of(t), volume=t.volume, provider=t.provider)
            for t in raw_ticks
            if price_of(t) is not None
        ]

    filtered = filter_regular_session_ticks(tick_likes, calendar)
    report = CandleBuildReport(symbol=symbol, symbol_type=symbol_type, trading_date=trading_date, ticks_used=len(filtered))

    with session_factory() as session:
        for timeframe in timeframes:
            if timeframe == "daily":
                windows = calendar.regular_session_windows(trading_date)
                if not windows:
                    continue
                candle = build_daily_candle(
                    filtered, symbol, windows[0][0], windows[-1][1], config.candles.volume_convention
                )
                candles = [candle] if candle else []
            else:
                candles = build_intraday_candles(
                    filtered, symbol, timeframe, tz, config.candles.volume_convention
                )

            for candle in candles:
                inserted = insert_candle(
                    session,
                    symbol=candle.symbol,
                    timeframe=candle.timeframe,
                    open_time=candle.open_time,
                    close_time=candle.close_time,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    trade_count=candle.trade_count,
                    provider=candle.provider,
                    is_synthetic=candle.is_synthetic,
                )
                report.record(timeframe, inserted)

    return report
