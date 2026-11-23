"""Tests for src.market.trading_calendar (Phase 8). No database dependency."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.market.trading_calendar import (
    MarketCalendarValidationError,
    TradingCalendar,
    load_market_calendar,
)

TZ = ZoneInfo("Asia/Karachi")

VALID_CALENDAR = """
source_urls:
  - https://www.psx.com.pk/psx/exchange/general/trading-hours
fetched_date: "2026-09-18"
weekday_sessions:
  monday_to_thursday:
    - pre_open: {start: "09:15", end: "09:30"}
      pre_open_break: {start: "09:30", end: "09:32"}
      regular: {start: "09:32", end: "15:30"}
      post_close: {start: "15:35", end: "15:50"}
  friday:
    - pre_open: {start: "09:00", end: "09:15"}
      pre_open_break: {start: "09:15", end: "09:17"}
      regular: {start: "09:17", end: "12:00"}
    - pre_open: {start: "14:15", end: "14:30"}
      pre_open_break: {start: "14:30", end: "14:32"}
      regular: {start: "14:32", end: "16:30"}
      post_close: {start: "16:35", end: "16:50"}
  saturday: []
  sunday: []
holidays:
  - date: "2026-08-14"
    name: "Independence Day"
    moon_dependent: false
special_sessions: []
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "market_calendar.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _calendar(tmp_path: Path, content: str = VALID_CALENDAR) -> TradingCalendar:
    data = load_market_calendar(_write(tmp_path, content))
    return TradingCalendar(data, "Asia/Karachi")


def test_valid_calendar_loads(tmp_path):
    data = load_market_calendar(_write(tmp_path, VALID_CALENDAR))
    assert data.fetched_date == dt.date(2026, 9, 18)
    assert len(data.holidays) == 1
    assert len(data.weekday_blocks[0]) == 1  # Monday
    assert len(data.weekday_blocks[4]) == 2  # Friday
    assert data.weekday_blocks[5] == []  # Saturday


def test_missing_file_raises(tmp_path):
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(tmp_path / "does_not_exist.yaml")


def test_bad_time_format_is_rejected(tmp_path):
    bad = VALID_CALENDAR.replace('"09:32"', '"9:32am"')
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(_write(tmp_path, bad))


def test_start_after_end_is_rejected(tmp_path):
    bad = VALID_CALENDAR.replace(
        'regular: {start: "09:32", end: "15:30"}', 'regular: {start: "16:00", end: "15:30"}'
    )
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(_write(tmp_path, bad))


def test_missing_regular_window_is_rejected(tmp_path):
    bad = VALID_CALENDAR.replace(
        'regular: {start: "09:32", end: "15:30"}\n      post_close', "post_close"
    )
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(_write(tmp_path, bad))


def test_duplicate_holiday_dates_rejected(tmp_path):
    bad = VALID_CALENDAR.replace(
        'holidays:\n  - date: "2026-08-14"\n    name: "Independence Day"\n    moon_dependent: false',
        'holidays:\n  - date: "2026-08-14"\n    name: "Independence Day"\n    moon_dependent: false\n'
        '  - date: "2026-08-14"\n    name: "Some Other Holiday"\n    moon_dependent: false',
    )
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(_write(tmp_path, bad))


def test_overlapping_special_sessions_rejected(tmp_path):
    bad = VALID_CALENDAR.replace(
        "special_sessions: []",
        """special_sessions:
  - start_date: "2026-03-01"
    end_date: "2026-03-20"
    description: "A"
    weekday_sessions:
      monday_to_thursday:
        - regular: {start: "09:15", end: "13:30"}
      friday:
        - regular: {start: "09:00", end: "12:30"}
      saturday: []
      sunday: []
  - start_date: "2026-03-10"
    end_date: "2026-03-25"
    description: "B (overlaps A)"
    weekday_sessions:
      monday_to_thursday:
        - regular: {start: "09:15", end: "13:30"}
      friday:
        - regular: {start: "09:00", end: "12:30"}
      saturday: []
      sunday: []
""",
    )
    with pytest.raises(MarketCalendarValidationError):
        load_market_calendar(_write(tmp_path, bad))


# --- TradingCalendar behavior ---------------------------------------------


def test_monday_regular_session(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ)  # a Monday
    assert cal.current_session(at) == "REGULAR"
    assert cal.is_market_open(at) is True
    assert cal.is_market_closed(at) is False


def test_monday_pre_open_and_break(tmp_path):
    cal = _calendar(tmp_path)
    pre_open = dt.datetime(2026, 9, 21, 9, 20, tzinfo=TZ)
    break_time = dt.datetime(2026, 9, 21, 9, 31, tzinfo=TZ)
    assert cal.current_session(pre_open) == "PRE_OPEN"
    assert cal.is_pre_open(pre_open) is True
    assert cal.current_session(break_time) == "PRE_OPEN_BREAK"
    assert cal.is_pre_open(break_time) is True


def test_monday_post_close(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 21, 15, 40, tzinfo=TZ)
    assert cal.current_session(at) == "POST_CLOSE"
    assert cal.is_post_market(at) is True


def test_monday_outside_all_sessions_is_closed(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 21, 20, 0, tzinfo=TZ)
    assert cal.current_session(at) == "CLOSED"
    assert cal.is_market_open(at) is False


def test_friday_two_sessions_and_lunch_gap(tmp_path):
    cal = _calendar(tmp_path)
    session_1 = dt.datetime(2026, 9, 25, 10, 0, tzinfo=TZ)  # a Friday
    lunch_gap = dt.datetime(2026, 9, 25, 13, 0, tzinfo=TZ)
    session_2 = dt.datetime(2026, 9, 25, 15, 0, tzinfo=TZ)
    assert cal.is_market_open(session_1) is True
    assert cal.is_market_open(lunch_gap) is False
    assert cal.current_session(lunch_gap) == "CLOSED"
    assert cal.is_market_open(session_2) is True


def test_weekend_is_weekend_not_closed(tmp_path):
    cal = _calendar(tmp_path)
    saturday = dt.datetime(2026, 9, 26, 10, 0, tzinfo=TZ)
    sunday = dt.datetime(2026, 9, 27, 10, 0, tzinfo=TZ)
    assert cal.current_session(saturday) == "WEEKEND"
    assert cal.current_session(sunday) == "WEEKEND"


def test_holiday_overrides_a_weekday_session(tmp_path):
    cal = _calendar(tmp_path)
    # 2026-08-14 is a Friday in the fixture's holiday list.
    at = dt.datetime(2026, 8, 14, 10, 0, tzinfo=TZ)
    assert cal.is_holiday(at) is True
    assert cal.current_session(at) == "HOLIDAY"
    assert cal.is_market_open(at) is False


def test_holiday_name_lookup(tmp_path):
    cal = _calendar(tmp_path)
    assert cal.holiday_name(dt.date(2026, 8, 14)) == "Independence Day"
    assert cal.holiday_name(dt.date(2026, 8, 15)) is None


def test_next_open_from_after_close_returns_next_trading_day(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 21, 16, 0, tzinfo=TZ)  # Monday after close
    next_open = cal.next_open(at)
    assert next_open == dt.datetime(2026, 9, 22, 9, 32, tzinfo=TZ)  # Tuesday


def test_next_close_from_during_session(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 21, 10, 0, tzinfo=TZ)  # during Monday's session
    next_close = cal.next_close(at)
    assert next_close == dt.datetime(2026, 9, 21, 15, 30, tzinfo=TZ)


def test_next_open_and_close_from_friday_lunch_gap(tmp_path):
    cal = _calendar(tmp_path)
    at = dt.datetime(2026, 9, 25, 13, 0, tzinfo=TZ)
    assert cal.next_open(at) == dt.datetime(2026, 9, 25, 14, 32, tzinfo=TZ)
    assert cal.next_close(at) == dt.datetime(2026, 9, 25, 16, 30, tzinfo=TZ)


def test_next_open_skips_a_holiday(tmp_path):
    cal = _calendar(tmp_path)
    # 2026-08-13 is a Thursday; 2026-08-14 (Friday) is a holiday in the
    # fixture, so next_open from Thursday's close should skip straight to
    # the following Monday (2026-08-17).
    at = dt.datetime(2026, 8, 13, 16, 0, tzinfo=TZ)
    next_open = cal.next_open(at)
    assert next_open.date() == dt.date(2026, 8, 17)


def test_naive_datetime_is_localized_to_calendar_timezone(tmp_path):
    cal = _calendar(tmp_path)
    naive = dt.datetime(2026, 9, 21, 10, 0)  # no tzinfo
    assert cal.is_market_open(naive) is True


def test_special_session_overrides_regular_hours(tmp_path):
    content = VALID_CALENDAR.replace(
        "special_sessions: []",
        """special_sessions:
  - start_date: "2026-03-01"
    end_date: "2026-03-20"
    description: "Shortened hours example"
    weekday_sessions:
      monday_to_thursday:
        - regular: {start: "09:15", end: "13:30"}
      friday:
        - regular: {start: "09:00", end: "12:30"}
      saturday: []
      sunday: []
""",
    )
    cal = _calendar(tmp_path, content)

    # Inside the special-session window: shortened hours apply.
    inside = dt.datetime(2026, 3, 10, 13, 0, tzinfo=TZ)  # a Tuesday
    assert cal.is_market_open(inside) is True
    after_shortened_close = dt.datetime(2026, 3, 10, 14, 0, tzinfo=TZ)
    assert cal.is_market_open(after_shortened_close) is False

    # Outside the window: normal hours apply again.
    outside = dt.datetime(2026, 3, 25, 14, 0, tzinfo=TZ)  # a Wednesday, normal 09:32-15:30
    assert cal.is_market_open(outside) is True
