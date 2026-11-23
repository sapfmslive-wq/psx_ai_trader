"""LiveCollector — the continuous, robust polling loop (Phase 9, Section 15).

Wraps `collect_once()` (indices + stocks together) in everything Section 15
requires of a real collector:

    - reconnect automatically, with exponential backoff
    - never crash on one bad cycle — log and keep going
    - graceful shutdown on Ctrl+C / SIGTERM (finishes the current cycle,
      disconnects cleanly, does not leave the provider or DB mid-write)
    - record connection_lost / connection_restored events (with duration)
      into `system_events` — Section 71's requirement, satisfied here for
      the first time since Phase 3 created the table
    - a retry limit that logs CRITICAL after repeated consecutive failures,
      but — per Section 15's "the collector should NOT crash because of one
      bad record" — never gives up permanently on its own; it keeps
      retrying with capped backoff indefinitely, since the person running
      it, not the collector itself, decides when a persistent outage
      warrants stopping the process

Testability: every source of real-world nondeterminism — wall-clock time,
`time.sleep`, and how many cycles to run — is injectable. Production code
never passes these; tests always do, so the loop can be exercised fully
without actually running forever or actually sleeping.
"""

from __future__ import annotations

import datetime as dt
import logging
import signal
import time
from typing import Callable

from src.config.loader import AppConfig
from src.data.base_provider import DataProvider
from src.data.collector import collect_once
from src.market.trading_calendar import TradingCalendar


class _StopRequestedDuringReconnect(Exception):
    """Internal signal: stop was requested while waiting to reconnect."""


class LiveCollector:
    def __init__(
        self,
        config: AppConfig,
        session_factory,  # noqa: ANN001
        provider_factory: Callable[[], DataProvider],
        calendar: TradingCalendar,
        quality_engine,  # noqa: ANN001 — src.data.validator.DataQualityEngine
        logger: logging.Logger,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
        max_cycles: int | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        self._config = config
        self._session_factory = session_factory
        self._provider_factory = provider_factory
        self._calendar = calendar
        self._quality_engine = quality_engine
        self._logger = logger
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        self._max_cycles = max_cycles  # None = run forever; tests set a small int

        self._poll_interval_seconds = config.collector.poll_interval_seconds
        self._base_reconnect_seconds = config.collector.reconnect_seconds
        self._max_backoff_seconds = max(60, config.collector.reconnect_seconds * 8)
        self._max_consecutive_failures = config.collector.max_consecutive_failures

        self._provider: DataProvider | None = None
        self._stop_requested = False
        self._cycles_run = 0
        self._consecutive_failures = 0
        self._outage_started_at: dt.datetime | None = None

        if install_signal_handlers:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001
        self._logger.info("Received signal %s — shutting down gracefully after this cycle.", signum)
        self.request_stop()

    def request_stop(self) -> None:
        self._stop_requested = True

    def run_forever(self) -> None:
        self._logger.info(
            "Live collector starting (poll_interval=%ss, max_consecutive_failures=%s).",
            self._poll_interval_seconds, self._max_consecutive_failures,
        )
        while not self._stop_requested:
            self._run_one_cycle()
            self._cycles_run += 1
            if self._max_cycles is not None and self._cycles_run >= self._max_cycles:
                break
            if self._stop_requested:
                break
            self._interruptible_sleep(self._poll_interval_seconds)
        self._shutdown()

    def _interruptible_sleep(self, seconds: float) -> None:
        remaining = seconds
        step = min(1.0, seconds) if seconds > 0 else 0.0
        while remaining > 0 and not self._stop_requested:
            chunk = min(step, remaining)
            self._sleep_fn(chunk)
            remaining -= chunk

    def _shutdown(self) -> None:
        if self._provider is not None:
            try:
                self._provider.disconnect()
            except Exception as exc:  # noqa: BLE001 — shutdown must never raise
                self._logger.warning("Error disconnecting provider during shutdown: %s", exc)
        self._logger.info("Live collector stopped after %d cycle(s).", self._cycles_run)

    def _run_one_cycle(self) -> None:
        try:
            self._ensure_connected()
            at = self._now_fn()
            report = collect_once(
                self._config, self._session_factory, self._provider,
                self._calendar, self._quality_engine, force=False, at=at,
            )
            self._on_success(report)
        except _StopRequestedDuringReconnect:
            pass
        except Exception as exc:  # noqa: BLE001 — one bad cycle must never crash the loop
            self._on_failure(exc)

    def _ensure_connected(self) -> None:
        if self._provider is not None and self._provider.is_connected():
            return
        self._connect_with_backoff()

    def _connect_with_backoff(self) -> None:
        attempt = 0
        while not self._stop_requested:
            try:
                provider = self._provider_factory()
                provider.connect()
                self._provider = provider
                return
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                wait = min(self._base_reconnect_seconds * (2 ** (attempt - 1)), self._max_backoff_seconds)
                self._logger.warning(
                    "Connect attempt %d failed: %s — retrying in %.0fs", attempt, exc, wait
                )
                self._interruptible_sleep(wait)
        raise _StopRequestedDuringReconnect()

    def _on_success(self, report) -> None:  # noqa: ANN001
        if self._outage_started_at is not None:
            duration = (self._now_fn() - self._outage_started_at).total_seconds()
            self._record_system_event(
                "INFO", f"connection_restored after {duration:.0f}s outage "
                        f"({self._consecutive_failures} failed cycle(s))"
            )
            self._outage_started_at = None
        self._consecutive_failures = 0

        if report.skipped:
            self._logger.info("Cycle skipped: %s", report.skip_reason)
        else:
            self._logger.info(
                "Cycle recorded %d symbol(s) (session=%s).", len(report.outcomes), report.session
            )

    def _on_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        if self._outage_started_at is None:
            self._outage_started_at = self._now_fn()
            self._record_system_event("ERROR", f"connection_lost: {exc}")

        self._logger.error("Cycle failed (%d consecutive): %s", self._consecutive_failures, exc)

        if self._provider is not None:
            try:
                self._provider.disconnect()
            except Exception:  # noqa: BLE001 — best-effort cleanup only
                pass
            self._provider = None

        if self._consecutive_failures == self._max_consecutive_failures:
            self._logger.critical(
                "Max consecutive failures (%d) reached — still retrying with backoff.",
                self._max_consecutive_failures,
            )
            self._record_system_event(
                "CRITICAL",
                f"max_consecutive_failures reached ({self._max_consecutive_failures}); "
                f"collector is still running and will keep retrying",
            )

    def _record_system_event(self, level: str, message: str) -> None:
        try:
            from src.database.repository import insert_system_event

            with self._session_factory() as session:
                insert_system_event(
                    session, timestamp=self._now_fn(), level=level, category="data", message=message
                )
        except Exception as exc:  # noqa: BLE001 — logging the outage must never itself crash the loop
            self._logger.error("Failed to record system event %r: %s", message, exc)
