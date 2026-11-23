"""CSV column auto-detection and value parsing.

Deliberately has zero dependency on SQLAlchemy or the database layer — it
only knows how to turn "a CSV header row + a raw string" into "a canonical
field name + a typed value". `csv_importer.py` is what wires this to actual
database writes. Keeping this pure makes it trivially unit-testable and
reusable (e.g. a future CSVProvider replay implementation can reuse it
without pulling in the database).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

# Canonical field -> list of header spellings we'll recognize, after
# normalization (lowercased, spaces/dashes -> underscore, % -> pct).
# This is intentionally conservative: if a header doesn't match one of
# these, we do NOT guess — the importer will report it as unmapped and
# require an explicit --mapping instead of risking silently importing the
# wrong column as a price (Section 3: "Do NOT assume ... exists").
STOCK_FIELD_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "date", "time"],
    "last_price": ["last_price", "close", "price", "ltp", "last"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "previous_close": ["previous_close", "prev_close", "prevclose", "pclose"],
    "change": ["change", "chg", "net_change"],
    "change_percent": ["change_percent", "chg_pct", "pct_change", "change_pct"],
    "volume": ["volume", "vol", "qty", "quantity", "shares_traded"],
    "bid": ["bid", "bid_price"],
    "ask": ["ask", "ask_price", "offer"],
    "turnover": ["turnover", "value_traded"],
    "vwap": ["vwap"],
}

INDEX_FIELD_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "date", "time"],
    "value": ["value", "close", "index_value", "points", "last"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "change": ["change", "chg", "net_change"],
    "change_percent": ["change_percent", "chg_pct", "pct_change", "change_pct"],
    "volume": ["volume", "vol"],
}

_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "stock": ("timestamp", "last_price"),
    "index": ("timestamp", "value"),
}

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%Y-%m-%d",
)


class CSVMappingError(ValueError):
    """Raised when required columns cannot be determined from the header."""


def normalize_header(name: str) -> str:
    """Normalize a CSV header for alias matching.

    "Last Price" -> "last_price", "Change %" -> "change_pct", etc.
    """
    normalized = name.strip().lower()
    normalized = normalized.replace("%", "pct")
    for ch in (" ", "-", "."):
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def autodetect_mapping(headers: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    """Return {canonical_field: actual_header} for every canonical field
    whose alias matches one of the given headers. Fields with no match are
    simply absent from the result — callers decide whether that's fatal."""
    normalized_to_original: dict[str, str] = {}
    for header in headers:
        normalized_to_original.setdefault(normalize_header(header), header)

    mapping: dict[str, str] = {}
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias in normalized_to_original:
                mapping[canonical] = normalized_to_original[alias]
                break
    return mapping


def resolve_mapping(
    headers: list[str],
    symbol_type: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Auto-detect a column mapping, apply any explicit overrides, and
    validate that every required field for `symbol_type` is present.

    Args:
        headers: The CSV file's header row.
        symbol_type: "stock" or "index".
        overrides: Explicit {canonical_field: actual_header} entries that
            take priority over auto-detection (from a user-supplied
            mapping file).

    Raises:
        CSVMappingError: if a required field could not be determined.
    """
    if symbol_type not in _REQUIRED_FIELDS:
        raise ValueError(f"symbol_type must be 'stock' or 'index' (got: {symbol_type!r})")

    aliases = STOCK_FIELD_ALIASES if symbol_type == "stock" else INDEX_FIELD_ALIASES
    mapping = autodetect_mapping(headers, aliases)
    if overrides:
        mapping.update(overrides)

    missing = [f for f in _REQUIRED_FIELDS[symbol_type] if f not in mapping]
    if missing:
        raise CSVMappingError(
            f"Could not determine column(s) for {missing}. "
            f"Detected headers: {headers}. Auto-detected mapping so far: {mapping}. "
            f"Provide --mapping with an explicit {{\"field\": \"header\"}} entry for "
            f"the missing field(s)."
        )
    return mapping


def parse_float(raw: object) -> float | None:
    """Parse a CSV cell as a float. Returns None for empty/unparseable
    input rather than raising — the caller decides whether that's fatal
    for the field in question."""
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_timestamp(raw: object, default_tz: ZoneInfo) -> dt.datetime | None:
    """Parse a CSV timestamp cell using a small set of common formats.

    A naive (timezone-less) result is localized to `default_tz` — per
    Section 10, every stored timestamp must be timezone-aware, and the
    project's primary market timezone is the correct default for a PSX
    historical file that doesn't specify one.

    Returns None if no format matches, rather than raising.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    for fmt in _TIMESTAMP_FORMATS:
        try:
            parsed = dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=default_tz)
        return parsed

    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed
