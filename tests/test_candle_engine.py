"""Tests for src.data.candle_engine (Phase 10). No database dependency for
the pure bucketing/aggregation functions."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from src.data.candle_engine import (
    TickLike,
    build_daily_candle,
    build_intraday_candles,
    filter_regular_session_ticks,
)
from src.market.trading_calendar import TradingCalendar, load_market_calendar

TZ = ZoneInfo("Asia/Karachi")

CALENDAR = """
source_urls: []
fetched_date: "2026-09-18"
weekday_sessions:
  monday_to_thursday:
    - regular: {start: "09:32", end: "15:30"}
  friday:
    - regular: {start: "09:17", end: "12:00"}
    - regular: {start: "14:32", end: "16:30"}
  saturday: []
  sunday: []
holidays: []
special_sessions: []
"""


def _ts(hour, minute, second=0, day=21):
    return dt.datetime(2026, 9, day, hour, minute, second, tzinfo=TZ)  # a Monday


def _tick(hour, minute, price, volume=None, second=0, provider="mock"):
    return TickLike(timestamp=_ts(hour, minute, second), price=price, volume=volume, provider=provider)


def test_intraday_bucketing_groups_ticks_into_correct_windows():
    ticks = [
        _tick(9, 32, 100.0),
        _tick(9, 33, 101.0),
        _tick(9, 34, 99.0),
        _tick(9, 37, 102.0),  # falls in the next 5m bucket
    ]
    candles = build_intraday_candles(ticks, "THCCL", "5m", TZ, "per_tick")

    # 09:32-09:34 fall in the 09:30-09:35 bucket; 09:37 falls in 09:35-09:40.
    assert len(candles) == 2
    first, second = candles
    assert first.open_time == _ts(9, 30)
    assert first.close_time == _ts(9, 35)
    assert first.open == 100.0
    assert first.high == 101.0
    assert first.low == 99.0
    assert first.close == 99.0
    assert first.trade_count == 3

    assert second.open_time == _ts(9, 35)
    assert second.open == second.close == 102.0
    assert second.trade_count == 1


def test_per_tick_volume_is_summed_within_bucket():
    ticks = [_tick(9, 32, 100.0, volume=1000), _tick(9, 33, 101.0, volume=500)]
    candles = build_intraday_candles(ticks, "THCCL", "5m", TZ, "per_tick")
    assert candles[0].volume == 1500


def test_cumulative_volume_uses_delta_from_previous_bucket():
    ticks = [
        _tick(9, 32, 100.0, volume=1000),  # cumulative: 1000
        _tick(9, 34, 101.0, volume=1500),  # cumulative: 1500 (bucket 1 total: 1500)
        _tick(9, 37, 102.0, volume=1800),  # next bucket, cumulative: 1800 (delta: 300)
    ]
    candles = build_intraday_candles(ticks, "THCCL", "5m", TZ, "cumulative")
    assert len(candles) == 2
    assert candles[0].volume == 1500  # first bucket: baseline 0, so full 1500
    assert candles[1].volume == 300  # 1800 - 1500


def test_cumulative_volume_going_backwards_is_treated_as_unknown_not_negative():
    ticks = [
        _tick(9, 32, 100.0, volume=5000),
        _tick(9, 37, 101.0, volume=100),  # counter reset/anomaly — went down
    ]
    candles = build_intraday_candles(ticks, "THCCL", "5m", TZ, "cumulative")
    assert candles[0].volume == 5000
    assert candles[1].volume is None  # never a negative volume


def test_is_synthetic_reflects_mock_provider():
    ticks = [_tick(9, 32, 100.0, provider="mock")]
    candles = build_intraday_candles(ticks, "THCCL", "5m", TZ, "per_tick")
    assert candles[0].is_synthetic is True

    ticks_real = [_tick(9, 32, 100.0, provider="csv")]
    candles_real = build_intraday_candles(ticks_real, "THCCL", "5m", TZ, "per_tick")
    assert candles_real[0].is_synthetic is False


def test_empty_ticks_produce_no_candles():
    assert build_intraday_candles([], "THCCL", "5m", TZ, "per_tick") == []


def test_daily_candle_spans_the_whole_day():
    ticks = [
        _tick(9, 32, 100.0, volume=1000),
        _tick(12, 0, 105.0, volume=2000),
        _tick(15, 29, 98.0, volume=3000),
    ]
    open_time = _ts(9, 32)
    close_time = _ts(15, 30)
    candle = build_daily_candle(ticks, "THCCL", open_time, close_time, "cumulative")

    assert candle.open == 100.0
    assert candle.close == 98.0
    assert candle.high == 105.0
    assert candle.low == 98.0
    assert candle.volume == 3000  # baseline 0 for the day, last cumulative value
    assert candle.trade_count == 3
    assert candle.open_time == open_time
    assert candle.close_time == close_time


def test_daily_candle_returns_none_for_no_ticks():
    assert build_daily_candle([], "THCCL", _ts(9, 32), _ts(15, 30), "cumulative") is None


def test_filter_regular_session_excludes_pre_open_and_post_close(tmp_path):
    calendar_path = tmp_path / "market_calendar.yaml"
    calendar_path.write_text(
        CALENDAR.replace(
            'regular: {start: "09:32", end: "15:30"}',
            'pre_open: {start: "09:15", end: "09:30"}\n'
            '      regular: {start: "09:32", end: "15:30"}\n'
            '      post_close: {start: "15:35", end: "15:50"}',
        ),
        encoding="utf-8",
    )
    calendar = TradingCalendar(load_market_calendar(calendar_path), "Asia/Karachi")

    ticks = [
        _tick(9, 20, 100.0),  # pre-open
        _tick(10, 0, 101.0),  # regular
        _tick(15, 40, 102.0),  # post-close
    ]
    filtered = filter_regular_session_ticks(ticks, calendar)
    assert len(filtered) == 1
    assert filtered[0].price == 101.0


def test_friday_two_sessions_produce_separate_daily_windows_from_calendar(tmp_path):
    calendar_path = tmp_path / "market_calendar.yaml"
    calendar_path.write_text(CALENDAR, encoding="utf-8")
    calendar = TradingCalendar(load_market_calendar(calendar_path), "Asia/Karachi")

    friday = dt.date(2026, 9, 25)
    windows = calendar.regular_session_windows(friday)
    assert len(windows) == 2

    # Ticks from both Friday sessions should all end up in one daily candle
    # spanning session 1's open to session 2's close.
    ticks = [
        TickLike(timestamp=dt.datetime(2026, 9, 25, 10, 0, tzinfo=TZ), price=80.0, volume=100, provider="mock"),
        TickLike(timestamp=dt.datetime(2026, 9, 25, 15, 0, tzinfo=TZ), price=82.0, volume=200, provider="mock"),
    ]
    filtered = filter_regular_session_ticks(ticks, calendar)
    candle = build_daily_candle(filtered, "THCCL", windows[0][0], windows[-1][1], "cumulative")

    assert candle.open == 80.0
    assert candle.close == 82.0
    assert candle.open_time == windows[0][0]
    assert candle.close_time == windows[-1][1]
