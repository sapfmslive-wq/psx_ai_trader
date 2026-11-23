"""Database engine and session construction.

This is the only module allowed to call `sqlalchemy.create_engine`. Every
other module gets a session via `get_session_factory()` from here.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.loader import load_config
from src.database.models import Base
from src.utils.file_utils import resolve_path


def _enable_foreign_keys(engine: Engine) -> None:
    """SQLite does not enforce FOREIGN KEY constraints unless told to.

    Without this, the Fill -> Order relationship's integrity is not actually
    enforced at the database level, only conceptually. Enable it explicitly.
    """

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def build_engine(db_path: Path | None = None, echo: bool | None = None) -> Engine:
    """Build a SQLite engine.

    Args:
        db_path: Absolute path to the .db file. If omitted, resolved from
            `config/settings.yaml` (`database.path`).
        echo: Whether SQLAlchemy should log every SQL statement. If omitted,
            resolved from config (`database.echo`).

    Returns:
        A configured SQLAlchemy Engine with foreign-key enforcement enabled.
    """
    if db_path is None or echo is None:
        config = load_config()
        if db_path is None:
            db_path = resolve_path(config.database.path)
        if echo is None:
            echo = config.database.echo

    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine_for_path(db_path, echo=echo)
    _enable_foreign_keys(engine)
    return engine


def create_engine_for_path(db_path: Path, echo: bool = False) -> Engine:
    """Low-level engine constructor, separated out so tests can pass an
    in-memory or temp-file path without touching config/env at all."""
    from sqlalchemy import create_engine

    return create_engine(f"sqlite:///{db_path}", echo=echo, future=True)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


def create_all_tables(engine: Engine) -> None:
    """Create every table defined in models.py if it does not already exist.

    Safe to call repeatedly (idempotent) — SQLAlchemy checks for existing
    tables before issuing CREATE TABLE.
    """
    Base.metadata.create_all(engine)
