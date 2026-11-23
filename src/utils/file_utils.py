"""File and directory utilities.

Centralizes "where do project files live" so every later phase (data
collector, candle engine, model trainer, dashboard, ...) resolves paths the
same way instead of re-implementing path logic per module.
"""

from __future__ import annotations

from pathlib import Path

# The project root is defined as the parent of the `src/` package that this
# file lives in. This works regardless of the current working directory the
# CLI is invoked from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Directories required by the project at Phase 1. Later phases may add
# subdirectories (e.g. data/raw/<date>/) at runtime, but these top-level
# directories must always exist.
REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "config",
    "data/raw",
    "data/cleaned",
    "data/candles",
    "data/features",
    "data/predictions",
    "data/reports",
    "database",
    "logs",
    "models/trained",
    "models/scalers",
    "models/metadata",
    "scripts",
)


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root.

    Args:
        relative_path: A path such as "data/raw" or "logs/system.log".

    Returns:
        The absolute Path, not guaranteed to exist.
    """
    return PROJECT_ROOT / relative_path


def ensure_directory(relative_path: str) -> Path:
    """Create a directory (and parents) if it does not already exist.

    This is idempotent: calling it repeatedly on an already-existing
    directory is a safe no-op.

    Args:
        relative_path: Directory path relative to the project root.

    Returns:
        The absolute Path to the (now guaranteed to exist) directory.
    """
    path = resolve_path(relative_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_all_required_directories() -> list[Path]:
    """Create every directory required by the current phase of the project.

    Returns:
        List of absolute Paths that were ensured to exist.
    """
    return [ensure_directory(rel) for rel in REQUIRED_DIRECTORIES]


def missing_required_directories() -> list[str]:
    """Return the subset of REQUIRED_DIRECTORIES that do not currently exist.

    Used by the `health` CLI command to report problems without side effects
    (unlike `ensure_all_required_directories`, this never creates anything).
    """
    return [
        rel for rel in REQUIRED_DIRECTORIES if not resolve_path(rel).is_dir()
    ]
