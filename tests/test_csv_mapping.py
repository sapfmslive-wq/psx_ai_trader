"""Tests for src.data.csv_mapping (Phase 5). No database dependency."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from src.data.csv_mapping import (
    CSVMappingError,
    autodetect_mapping,
    normalize_header,
    parse_float,
    parse_timestamp,
    resolve_mapping,
)

KARACHI = ZoneInfo("Asia/Karachi")


def test_normalize_header_handles_spaces_dashes_and_percent():
    assert normalize_header("Last Price") == "last_price"
    assert normalize_header("Change %") == "change_pct"
    assert normalize_header("prev-close") == "prev_close"
    assert normalize_header("  Volume  ") == "volume"


def test_autodetect_mapping_finds_common_stock_headers():
    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    from src.data.csv_mapping import STOCK_FIELD_ALIASES

    mapping = autodetect_mapping(headers, STOCK_FIELD_ALIASES)

    assert mapping["timestamp"] == "Date"
    assert mapping["last_price"] == "Close"
    assert mapping["open"] == "Open"
    assert mapping["high"] == "High"
    assert mapping["low"] == "Low"
    assert mapping["volume"] == "Volume"


def test_resolve_mapping_succeeds_for_stock_with_standard_headers():
    headers = ["Date", "Close", "Volume"]
    mapping = resolve_mapping(headers, "stock")
    assert mapping["timestamp"] == "Date"
    assert mapping["last_price"] == "Close"


def test_resolve_mapping_succeeds_for_index_with_standard_headers():
    headers = ["Date", "Index Value", "High", "Low"]
    # "Index Value" normalizes to "index_value", which is a recognized alias.
    mapping = resolve_mapping(headers, "index")
    assert mapping["timestamp"] == "Date"
    assert mapping["value"] == "Index Value"


def test_resolve_mapping_raises_when_required_field_missing():
    headers = ["SomeWeirdColumn", "AnotherOne"]
    with pytest.raises(CSVMappingError):
        resolve_mapping(headers, "stock")


def test_resolve_mapping_honors_explicit_override():
    headers = ["WeirdDateCol", "WeirdPriceCol"]
    mapping = resolve_mapping(
        headers,
        "stock",
        overrides={"timestamp": "WeirdDateCol", "last_price": "WeirdPriceCol"},
    )
    assert mapping["timestamp"] == "WeirdDateCol"
    assert mapping["last_price"] == "WeirdPriceCol"


def test_resolve_mapping_invalid_symbol_type_raises():
    with pytest.raises(ValueError):
        resolve_mapping(["Date", "Close"], "crypto")


def test_parse_float_handles_commas_and_blank():
    assert parse_float("1,234.56") == 1234.56
    assert parse_float("") is None
    assert parse_float(None) is None
    assert parse_float("not_a_number") is None
    assert parse_float("80.5") == 80.5


def test_parse_timestamp_common_formats_localized_to_default_tz():
    ts = parse_timestamp("2026-09-14 09:32:00", KARACHI)
    assert ts is not None
    assert ts.year == 2026 and ts.month == 9 and ts.day == 14
    assert ts.tzinfo is not None
    assert ts.utcoffset() == dt.timedelta(hours=5)  # Asia/Karachi is UTC+5


def test_parse_timestamp_date_only_format():
    ts = parse_timestamp("14/09/2026", KARACHI)
    assert ts is not None
    assert ts.day == 14 and ts.month == 9 and ts.year == 2026


def test_parse_timestamp_preserves_explicit_timezone():
    ts = parse_timestamp("2026-09-14T09:32:00+00:00", KARACHI)
    assert ts is not None
    assert ts.utcoffset() == dt.timedelta(0)  # explicit UTC, not overridden


def test_parse_timestamp_unparseable_returns_none():
    assert parse_timestamp("not a date", KARACHI) is None
    assert parse_timestamp("", KARACHI) is None
    assert parse_timestamp(None, KARACHI) is None
