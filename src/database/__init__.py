"""SQLite database layer (Phase 3).

- `models.py` — SQLAlchemy ORM table definitions.
- `engine.py` — engine/session construction, bound to config.database.path.
- `repository.py` — insert/query helpers, in particular duplicate-safe
  inserts for the tables that receive repeated/replayed data (ticks,
  candles).

No other module should import `sqlalchemy` directly to talk to the
database — go through this package so every query and every constraint
lives in one place.
"""
