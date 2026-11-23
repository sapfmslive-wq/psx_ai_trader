"""PSX trading calendar (Phase 8, Section 14 of the project spec).

Everything about *when* the market is open lives in
`config/market_calendar.yaml` — this module only knows how to interpret
that file. It hardcodes no PSX-specific time or date, per Section 14's
explicit instruction not to assume a schedule without researching the
current official one, and Section 10's requirement to handle open/close/
breaks/Friday sessions/holidays/special sessions through a *configurable*
calendar, not a hardcoded Monday-Friday assumption.

Two things live here:

    load_market_calendar()  - parses and validates market_calendar.yaml
    TradingCalendar          - the public query API (Section 14):
        is_market_open() / is_market_closed() / is_pre_open() /
        is_post_market() / current_session() / next_open() / next_close() /
        is_holiday()

Validation follows the same "collect every problem, raise once" pattern as
`src.config.loader`, for the same reason: fixing a calendar file one error
at a time is miserable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.config.loader import AppConfig
from src.utils.file_utils import resolve_path

_WEEKDAY_GROUP_NAMES = ("saturday", "sunday")  # monday_to_thursday/friday handled separately


class MarketCalendarValidationError(Exception):
    """Raised when market_calendar.yaml fails validation. Carries every
    problem found, not just the first, same as ConfigValidationError."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Market calendar validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


@dataclass(frozen=True)
class SessionWindow:
    start: dt.time
    end: dt.time


@dataclass(frozen=True)
class SessionBlock:
    """One contiguous session block within a trading day.

    Only `regular` (continuous trading) is required — `pre_open`,
    `pre_open_break`, and `post_close` are optional, since not every
    source of session times publishes all four (see the Ramadan example
    in market_calendar.yaml, which only has `regular`).
    """

    regular: SessionWindow
    pre_open: SessionWindow | None = None
    pre_open_break: SessionWindow | None = None
    post_close: SessionWindow | None = None


@dataclass(frozen=True)
class Holiday:
    date: dt.date
    name: str
    moon_dependent: bool


@dataclass(frozen=True)
class SpecialSession:
    start_date: dt.date
    end_date: dt.date
    description: str
    weekday_blocks: dict[int, list[SessionBlock]]  # 0=Monday .. 6=Sunday


@dataclass(frozen=True)
class MarketCalendarData:
    source_urls: tuple[str, ...]
    fetched_date: dt.date
    weekday_blocks: dict[int, list[SessionBlock]]  # 0=Monday .. 6=Sunday
    holidays: tuple[Holiday, ...]
    special_sessions: tuple[SpecialSession, ...]


def _parse_time(raw: object, errors: list[str], context: str) -> dt.time | None:
    if not isinstance(raw, str):
        errors.append(f"{context}: expected a 'HH:MM' string (got: {raw!r})")
        return None
    try:
        return dt.datetime.strptime(raw, "%H:%M").time()
    except ValueError:
        errors.append(f"{context}: '{raw}' is not a valid HH:MM time")
        return None


def _parse_window(raw: object, errors: list[str], context: str) -> SessionWindow | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or "start" not in raw or "end" not in raw:
        errors.append(f"{context}: must be a mapping with 'start' and 'end'")
        return None
    start = _parse_time(raw.get("start"), errors, f"{context}.start")
    end = _parse_time(raw.get("end"), errors, f"{context}.end")
    if start is None or end is None:
        return None
    if start >= end:
        errors.append(f"{context}: start ({start}) must be before end ({end})")
        return None
    return SessionWindow(start=start, end=end)


def _parse_block(raw: object, errors: list[str], context: str) -> SessionBlock | None:
    if not isinstance(raw, dict):
        errors.append(f"{context}: must be a mapping")
        return None

    regular = _parse_window(raw.get("regular"), errors, f"{context}.regular")
    if raw.get("regular") is None:
        errors.append(f"{context}: missing required 'regular' window")
    if regular is None:
        return None

    pre_open = _parse_window(raw.get("pre_open"), errors, f"{context}.pre_open")
    pre_open_break = _parse_window(raw.get("pre_open_break"), errors, f"{context}.pre_open_break")
    post_close = _parse_window(raw.get("post_close"), errors, f"{context}.post_close")

    ordered = [w for w in (pre_open, pre_open_break, regular, post_close) if w is not None]
    for a, b in zip(ordered, ordered[1:]):
        if a.end > b.start:
            errors.append(f"{context}: windows overlap or are out of order ({a.end} > {b.start})")

    return SessionBlock(regular=regular, pre_open=pre_open, pre_open_break=pre_open_break, post_close=post_close)


def _parse_block_list(raw: object, errors: list[str], context: str) -> list[SessionBlock]:
    if not isinstance(raw, list):
        errors.append(f"{context}: must be a list (use [] for no sessions)")
        return []
    blocks = []
    for i, item in enumerate(raw):
        block = _parse_block(item, errors, f"{context}[{i}]")
        if block is not None:
            blocks.append(block)
    return blocks


def _parse_weekday_sessions(raw: object, errors: list[str], context: str) -> dict[int, list[SessionBlock]]:
    if not isinstance(raw, dict):
        errors.append(f"{context}: must be a mapping")
        return {i: [] for i in range(7)}

    result: dict[int, list[SessionBlock]] = {}
    mon_thu_blocks = _parse_block_list(
        raw.get("monday_to_thursday", []), errors, f"{context}.monday_to_thursday"
    )
    for day in range(4):  # Monday=0 .. Thursday=3
        result[day] = mon_thu_blocks
    result[4] = _parse_block_list(raw.get("friday", []), errors, f"{context}.friday")  # Friday
    for name, index in zip(_WEEKDAY_GROUP_NAMES, (5, 6)):  # Saturday=5, Sunday=6
        result[index] = _parse_block_list(raw.get(name, []), errors, f"{context}.{name}")
    return result


def _parse_holiday(raw: object, errors: list[str], index: int) -> Holiday | None:
    if not isinstance(raw, dict):
        errors.append(f"holidays[{index}]: must be a mapping")
        return None

    date_raw = raw.get("date")
    if not isinstance(date_raw, str):
        errors.append(f"holidays[{index}]: 'date' must be a YYYY-MM-DD string")
        return None
    try:
        date = dt.date.fromisoformat(date_raw)
    except ValueError:
        errors.append(f"holidays[{index}]: invalid date {date_raw!r}")
        return None

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"holidays[{index}]: 'name' must be a non-empty string")
        name = "Unnamed holiday"

    moon_dependent = raw.get("moon_dependent", False)
    if not isinstance(moon_dependent, bool):
        errors.append(f"holidays[{index}]: 'moon_dependent' must be true or false")
        moon_dependent = False

    return Holiday(date=date, name=name, moon_dependent=moon_dependent)


def _parse_special_session(raw: object, errors: list[str], index: int) -> SpecialSession | None:
    if not isinstance(raw, dict):
        errors.append(f"special_sessions[{index}]: must be a mapping")
        return None

    context = f"special_sessions[{index}]"

    try:
        start_date = dt.date.fromisoformat(raw.get("start_date", ""))
    except ValueError:
        errors.append(f"{context}: invalid start_date {raw.get('start_date')!r}")
        return None
    try:
        end_date = dt.date.fromisoformat(raw.get("end_date", ""))
    except ValueError:
        errors.append(f"{context}: invalid end_date {raw.get('end_date')!r}")
        return None
    if start_date > end_date:
        errors.append(f"{context}: start_date {start_date} is after end_date {end_date}")

    weekday_blocks = _parse_weekday_sessions(
        raw.get("weekday_sessions"), errors, f"{context}.weekday_sessions"
    )

    return SpecialSession(
        start_date=start_date,
        end_date=end_date,
        description=str(raw.get("description", "")),
        weekday_blocks=weekday_blocks,
    )


def load_market_calendar(path: Path | None = None) -> MarketCalendarData:
    """Load, validate, and return `config/market_calendar.yaml`.

    Raises:
        MarketCalendarValidationError: with every problem found, if the
            file is missing, unparsable, or fails any structural check
            (bad time format, overlapping windows, duplicate holiday
            dates, overlapping special-session ranges, ...).
    """
    path = path or resolve_path("config/market_calendar.yaml")
    errors: list[str] = []

    if not path.is_file():
        raise MarketCalendarValidationError([f"Market calendar file not found: {path}"])

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise MarketCalendarValidationError([f"Failed to parse {path}: {exc}"])

    if not isinstance(raw, dict):
        raise MarketCalendarValidationError([f"{path} must contain a YAML mapping at the top level."])

    source_urls = raw.get("source_urls", [])
    if not isinstance(source_urls, list) or not all(isinstance(u, str) for u in source_urls):
        errors.append("source_urls must be a list of strings")
        source_urls = []

    fetched_date_raw = raw.get("fetched_date")
    try:
        fetched_date = dt.date.fromisoformat(fetched_date_raw)
    except (TypeError, ValueError):
        errors.append(f"fetched_date must be a YYYY-MM-DD string (got: {fetched_date_raw!r})")
        fetched_date = dt.date(1970, 1, 1)

    weekday_blocks = _parse_weekday_sessions(raw.get("weekday_sessions"), errors, "weekday_sessions")

    holidays_raw = raw.get("holidays", [])
    if not isinstance(holidays_raw, list):
        errors.append("holidays must be a list")
        holidays_raw = []
    holidays: list[Holiday] = []
    seen_dates: set[dt.date] = set()
    for i, item in enumerate(holidays_raw):
        holiday = _parse_holiday(item, errors, i)
        if holiday is None:
            continue
        if holiday.date in seen_dates:
            errors.append(f"holidays[{i}]: duplicate date {holiday.date.isoformat()} ({holiday.name})")
        else:
            seen_dates.add(holiday.date)
            holidays.append(holiday)

    special_raw = raw.get("special_sessions", [])
    if not isinstance(special_raw, list):
        errors.append("special_sessions must be a list")
        special_raw = []
    special_sessions = [
        s for s in (_parse_special_session(item, errors, i) for i, item in enumerate(special_raw)) if s is not None
    ]
    sorted_special = sorted(special_sessions, key=lambda s: s.start_date)
    for a, b in zip(sorted_special, sorted_special[1:]):
        if a.end_date >= b.start_date:
            errors.append(
                f"special_sessions: overlapping ranges {a.start_date}..{a.end_date} "
                f"and {b.start_date}..{b.end_date}"
            )

    if errors:
        raise MarketCalendarValidationError(errors)

    return MarketCalendarData(
        source_urls=tuple(source_urls),
        fetched_date=fetched_date,
        weekday_blocks=weekday_blocks,
        holidays=tuple(holidays),
        special_sessions=tuple(special_sessions),
    )


class TradingCalendar:
    """Query API for market session state (Section 14).

    Every method accepts an optional `at` (a timezone-aware or naive
    datetime, interpreted in the calendar's configured timezone if naive)
    so behavior is fully testable without depending on the wall clock.
    Omitting `at` uses the current time.
    """

    def __init__(self, calendar: MarketCalendarData, timezone: str) -> None:
        self._calendar = calendar
        self._tz = ZoneInfo(timezone)
        self._holiday_dates = {h.date for h in calendar.holidays}

    @classmethod
    def from_config(cls, config: AppConfig, calendar_path: Path | None = None) -> "TradingCalendar":
        return cls(load_market_calendar(calendar_path), config.market.timezone)

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def _normalize(self, at: dt.datetime | None) -> dt.datetime:
        if at is None:
            return self._now()
        if at.tzinfo is None:
            return at.replace(tzinfo=self._tz)
        return at.astimezone(self._tz)

    def _to_date(self, at: dt.date | dt.datetime | None) -> dt.date:
        if at is None:
            return self._now().date()
        if isinstance(at, dt.datetime):
            return self._normalize(at).date()
        return at

    def is_holiday(self, at: dt.date | dt.datetime | None = None) -> bool:
        return self._to_date(at) in self._holiday_dates

    def holiday_name(self, at: dt.date | dt.datetime | None = None) -> str | None:
        date = self._to_date(at)
        for holiday in self._calendar.holidays:
            if holiday.date == date:
                return holiday.name
        return None

    def _blocks_for_date(self, date: dt.date) -> list[SessionBlock]:
        if date in self._holiday_dates:
            return []
        for special in self._calendar.special_sessions:
            if special.start_date <= date <= special.end_date:
                return special.weekday_blocks.get(date.weekday(), [])
        return self._calendar.weekday_blocks.get(date.weekday(), [])

    def regular_session_windows(self, date: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
        """The (open, close) datetime pairs of every 'regular' trading
        window on `date`, in chronological order (empty on a non-trading
        day). Used by the candle engine to bound daily candles and to know
        which windows to bucket intraday candles within, without needing
        to know about holidays/special-sessions/Friday's split itself.
        """
        return [
            (
                dt.datetime.combine(date, block.regular.start, tzinfo=self._tz),
                dt.datetime.combine(date, block.regular.end, tzinfo=self._tz),
            )
            for block in self._blocks_for_date(date)
        ]

    def current_session(self, at: dt.datetime | None = None) -> str:
        """One of: PRE_OPEN, PRE_OPEN_BREAK, REGULAR, POST_CLOSE, CLOSED,
        WEEKEND, HOLIDAY."""
        at = self._normalize(at)
        date = at.date()

        if date in self._holiday_dates:
            return "HOLIDAY"

        blocks = self._blocks_for_date(date)
        if not blocks:
            return "WEEKEND" if at.weekday() >= 5 else "CLOSED"

        current_time = at.time()
        for block in blocks:
            if block.pre_open and block.pre_open.start <= current_time < block.pre_open.end:
                return "PRE_OPEN"
            if block.pre_open_break and block.pre_open_break.start <= current_time < block.pre_open_break.end:
                return "PRE_OPEN_BREAK"
            if block.regular.start <= current_time < block.regular.end:
                return "REGULAR"
            if block.post_close and block.post_close.start <= current_time < block.post_close.end:
                return "POST_CLOSE"
        return "CLOSED"

    def is_market_open(self, at: dt.datetime | None = None) -> bool:
        return self.current_session(at) == "REGULAR"

    def is_market_closed(self, at: dt.datetime | None = None) -> bool:
        return not self.is_market_open(at)

    def is_pre_open(self, at: dt.datetime | None = None) -> bool:
        return self.current_session(at) in ("PRE_OPEN", "PRE_OPEN_BREAK")

    def is_post_market(self, at: dt.datetime | None = None) -> bool:
        return self.current_session(at) == "POST_CLOSE"

    def _iter_regular_windows(self, from_date: dt.date, max_days: int = 60):
        for offset in range(max_days):
            date = from_date + dt.timedelta(days=offset)
            for block in self._blocks_for_date(date):
                open_dt = dt.datetime.combine(date, block.regular.start, tzinfo=self._tz)
                close_dt = dt.datetime.combine(date, block.regular.end, tzinfo=self._tz)
                yield open_dt, close_dt

    def next_open(self, at: dt.datetime | None = None) -> dt.datetime:
        """The next time continuous ('regular') trading begins, strictly
        after `at`."""
        at = self._normalize(at)
        for open_dt, _close_dt in self._iter_regular_windows(at.date()):
            if open_dt > at:
                return open_dt
        raise RuntimeError(f"No trading session found within the lookahead window from {at}")

    def next_close(self, at: dt.datetime | None = None) -> dt.datetime:
        """The next time continuous ('regular') trading ends, strictly
        after `at`."""
        at = self._normalize(at)
        for _open_dt, close_dt in self._iter_regular_windows(at.date()):
            if close_dt > at:
                return close_dt
        raise RuntimeError(f"No trading session found within the lookahead window from {at}")
