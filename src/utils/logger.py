"""Structured logging for PSX AI Trader.

Per project requirements, logging must never silently swallow errors and
must be split into separate streams:

    logs/system.log    general application lifecycle events
    logs/data.log       data collection / validation events (Phase 4+)
    logs/model.log       training / prediction events (Phase 14+)
    logs/trading.log      signal / order / paper-trade events (Phase 18+)
    logs/error.log         ERROR and CRITICAL from every stream, duplicated here

At Phase 1 only "system" is actually emitted to by application code (via
`main.py`); the other categories exist and are wired up now so later phases
just call `get_logger("data")`, `get_logger("model")`, etc. without touching
this file again.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.utils.file_utils import ensure_directory

_LOG_CATEGORIES: tuple[str, ...] = ("system", "data", "model", "trading")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file before rotation
_BACKUP_COUNT = 5

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: dict[str, logging.Logger] = {}


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)


def _build_file_handler(filename: str, level: int) -> RotatingFileHandler:
    logs_dir = ensure_directory("logs")
    handler = RotatingFileHandler(
        logs_dir / filename,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    return handler


def get_logger(category: str) -> logging.Logger:
    """Return a configured logger for the given category.

    Args:
        category: One of "system", "data", "model", "trading". Any other
            value is still accepted (so a future module can introduce a new
            category without editing this file) but will only write to
            `logs/<category>.log` and `logs/error.log`, not to `system.log`.

    Returns:
        A `logging.Logger` with a category-specific file handler, a shared
        error-file handler, and a console handler, all attached exactly
        once (safe to call repeatedly).
    """
    if category in _configured_loggers:
        return _configured_loggers[category]

    logger = logging.getLogger(f"psx_ai_trader.{category}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # never double-log via the root logger

    # Category-specific file (INFO and above).
    category_filename = f"{category}.log" if category in _LOG_CATEGORIES else f"{category}.log"
    logger.addHandler(_build_file_handler(category_filename, logging.INFO))

    # Shared error file (ERROR and CRITICAL from every category).
    logger.addHandler(_build_file_handler("error.log", logging.ERROR))

    # Console (INFO and above) so `python -m src.main ...` is legible.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_build_formatter())
    logger.addHandler(console_handler)

    _configured_loggers[category] = logger
    return logger


def reset_loggers_for_tests() -> None:
    """Remove all handlers and cached loggers.

    Only intended for use by the test suite, so each test can assert against
    a clean logger without file handles leaking across test cases.
    """
    for logger in _configured_loggers.values():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    _configured_loggers.clear()
