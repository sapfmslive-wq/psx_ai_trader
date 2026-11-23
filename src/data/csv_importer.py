"""Historical CSV importer (Phase 5/6, Section 95 + Section 13 of the spec).

Reads a CSV file of historical index or stock observations and writes:

    - one MarketTick per row read, unconditionally (raw, immutable, exactly
      as read — even rows that turn out to be unusable, per Section 11's
      "never destroy original data" rule)
    - one IndexTick / StockTick for every row where a timestamp could be
      determined, tagged with the VALID/SUSPECT/INVALID/DUPLICATE status
      the data-quality engine (`src/data/validator.py`) assigns it, via the
      same duplicate-safe repository functions the rest of the project uses

This module owns CSV-specific orchestration only. Column mapping/parsing
(turning a string cell into a float or a timestamp) lives in
`csv_mapping.py`; all business-rule validation (bad OHLC, negative price,
extreme moves, staleness, unknown symbols, ...) lives in `validator.py`.
This importer does not duplicate either — it wires them together.

**A row without a determinable timestamp is the one case that can never
become a tick row**, regardless of quality status: `IndexTick.timestamp`
and `StockTick.timestamp` are non-nullable, so there is no key to store it
under. Everything else the quality engine flags — bad price, bad OHLC,
negative volume, unknown symbol — still gets written, with that status
attached, per Section 13's "flag, never silently delete" rule.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config.loader import AppConfig
from src.data.csv_mapping import parse_float, parse_timestamp, resolve_mapping
from src.data.validator import DUPLICATE, DataQualityEngine, QualityInput


@dataclass
class ImportReport:
    file_path: str
    symbol: str
    symbol_type: str
    provider_name: str
    rows_read: int = 0
    rows_written: int = 0  # new IndexTick/StockTick rows created, any quality status
    rows_duplicate: int = 0  # exact repeat, either within this file or already in the DB
    rows_no_timestamp: int = 0  # no tick possible at all — no valid key to store under
    quality_counts: dict[str, int] = field(default_factory=lambda: {"VALID": 0, "SUSPECT": 0, "INVALID": 0})
    notable_rows: list[tuple[int, str, str]] = field(default_factory=list)  # (line, status, reason)

    def summary_lines(self, max_notable: int = 15) -> list[str]:
        lines = [
            f"File:              {self.file_path}",
            f"Symbol:            {self.symbol} ({self.symbol_type})",
            f"Provider label:    {self.provider_name}",
            f"Rows read:         {self.rows_read}",
            f"Rows written:      {self.rows_written}  "
            f"(VALID={self.quality_counts['VALID']}, "
            f"SUSPECT={self.quality_counts['SUSPECT']}, "
            f"INVALID={self.quality_counts['INVALID']})",
            f"Rows duplicate:    {self.rows_duplicate}",
            f"Rows no timestamp: {self.rows_no_timestamp}",
        ]
        if self.notable_rows:
            lines.append("")
            lines.append(
                f"First {min(max_notable, len(self.notable_rows))} notable row(s) "
                f"(anything not a clean VALID write):"
            )
            for row_number, status, reason in self.notable_rows[:max_notable]:
                lines.append(f"  line {row_number} [{status}]: {reason}")
            if len(self.notable_rows) > max_notable:
                lines.append(f"  ... and {len(self.notable_rows) - max_notable} more (see logs/data.log)")
        return lines


class CSVImporter:
    """Imports one CSV file at a time for one symbol.

    A single file is assumed to contain observations for exactly one
    symbol — mixed-symbol files should be split before import, since the
    column mapping is resolved once per file.

    A fresh `DataQualityEngine` is created per `import_file()` call, scoped
    to that one file — see `validator.py`'s class docstring for why
    cross-run duplicate/out-of-order detection is deliberately left to the
    database's unique constraints instead of engine state persisting across
    independent imports.
    """

    def __init__(self, config: AppConfig, session_factory) -> None:  # noqa: ANN001
        self._config = config
        self._session_factory = session_factory
        self._tz = ZoneInfo(config.market.timezone)

    def import_file(
        self,
        file_path: Path,
        symbol: str,
        symbol_type: str,
        provider_name: str = "csv",
        column_mapping: dict[str, str] | None = None,
    ) -> ImportReport:
        if symbol_type not in ("stock", "index"):
            raise ValueError(f"symbol_type must be 'stock' or 'index' (got: {symbol_type!r})")
        if not file_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Imported here, not at module level, so csv_mapping.py/validator.py
        # (and this module's pure orchestration logic) stay importable and
        # testable without SQLAlchemy installed — only actually running an
        # import requires it.
        from src.database.repository import insert_index_tick, insert_market_tick, insert_stock_tick

        report = ImportReport(
            file_path=str(file_path),
            symbol=symbol,
            symbol_type=symbol_type,
            provider_name=provider_name,
        )
        engine = DataQualityEngine(self._config)

        with file_path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            mapping = resolve_mapping(headers, symbol_type, overrides=column_mapping)

            with self._session_factory() as session:
                for row_number, row in enumerate(reader, start=2):  # header is line 1
                    report.rows_read += 1
                    self._import_row(
                        session=session,
                        row=row,
                        row_number=row_number,
                        mapping=mapping,
                        symbol=symbol,
                        symbol_type=symbol_type,
                        provider_name=provider_name,
                        report=report,
                        engine=engine,
                        insert_market_tick=insert_market_tick,
                        insert_index_tick=insert_index_tick,
                        insert_stock_tick=insert_stock_tick,
                    )

        return report

    def _get(self, row: dict[str, str], mapping: dict[str, str], field_name: str) -> float | None:
        return parse_float(row.get(mapping[field_name])) if field_name in mapping else None

    def _import_row(
        self,
        *,
        session,  # noqa: ANN001
        row: dict[str, str],
        row_number: int,
        mapping: dict[str, str],
        symbol: str,
        symbol_type: str,
        provider_name: str,
        report: ImportReport,
        engine: DataQualityEngine,
        insert_market_tick,  # noqa: ANN001
        insert_index_tick,  # noqa: ANN001
        insert_stock_tick,  # noqa: ANN001
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        price_field = "last_price" if symbol_type == "stock" else "value"

        timestamp = parse_timestamp(row.get(mapping["timestamp"]), self._tz)
        price = parse_float(row.get(mapping[price_field]))
        open_ = self._get(row, mapping, "open") if symbol_type == "stock" else None
        high = self._get(row, mapping, "high")
        low = self._get(row, mapping, "low")
        volume = self._get(row, mapping, "volume")

        # Raw preservation is unconditional: every row physically read from
        # the file gets a MarketTick, regardless of what happens next.
        insert_market_tick(
            session,
            provider=provider_name,
            symbol=symbol,
            local_timestamp=now,
            source_timestamp=timestamp,
            raw_payload=json.dumps(row),
        )

        if timestamp is None:
            report.rows_no_timestamp += 1
            report.notable_rows.append(
                (row_number, "NO_TIMESTAMP", f"unparseable timestamp: {row.get(mapping['timestamp'])!r}")
            )
            return

        quality_input = QualityInput(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            open=open_,
            high=high,
            low=low,
            volume=volume,
            market_status=None,  # historical CSV files don't usually carry a session status
        )
        result = engine.evaluate(quality_input)

        if result.status == DUPLICATE:
            report.rows_duplicate += 1
            report.notable_rows.append((row_number, DUPLICATE, result.reason_text or "duplicate"))
            return

        if symbol_type == "stock":
            inserted = insert_stock_tick(
                session,
                provider=provider_name,
                symbol=symbol,
                timestamp=timestamp,
                last_price=price,
                open=open_,
                high=high,
                low=low,
                previous_close=self._get(row, mapping, "previous_close"),
                change=self._get(row, mapping, "change"),
                change_percent=self._get(row, mapping, "change_percent"),
                volume=volume,
                bid=self._get(row, mapping, "bid"),
                bid_volume=None,
                ask=self._get(row, mapping, "ask"),
                ask_volume=None,
                turnover=self._get(row, mapping, "turnover"),
                vwap=self._get(row, mapping, "vwap"),
                market_status=None,
                source_timestamp=timestamp,
                local_timestamp=now,
                quality_status=result.status,
                quality_reason=result.reason_text,
            )
        else:
            inserted = insert_index_tick(
                session,
                provider=provider_name,
                symbol=symbol,
                timestamp=timestamp,
                value=price,
                change=self._get(row, mapping, "change"),
                change_percent=self._get(row, mapping, "change_percent"),
                high=high,
                low=low,
                volume=volume,
                market_status=None,
                source_timestamp=timestamp,
                local_timestamp=now,
                quality_status=result.status,
                quality_reason=result.reason_text,
            )

        if inserted:
            report.rows_written += 1
            report.quality_counts[result.status] += 1
            if result.status != "VALID":
                report.notable_rows.append((row_number, result.status, result.reason_text or ""))
        else:
            # Parseable and passed engine checks, but the database already
            # has this exact (provider, symbol, timestamp) — a cross-run
            # duplicate the in-memory engine couldn't have known about.
            report.rows_duplicate += 1
            report.notable_rows.append((row_number, DUPLICATE, "already present in the database"))
