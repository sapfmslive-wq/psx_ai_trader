"""Tests for src.utils.file_utils (Phase 1)."""

from __future__ import annotations

import shutil

import pytest

from src.utils.file_utils import (
    REQUIRED_DIRECTORIES,
    ensure_all_required_directories,
    missing_required_directories,
    project_root,
    resolve_path,
)


def test_project_root_points_to_repo_root():
    root = project_root()
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "README.md").is_file()


def test_resolve_path_is_relative_to_root():
    resolved = resolve_path("data/raw")
    assert resolved == project_root() / "data" / "raw"


def test_ensure_all_required_directories_creates_everything(tmp_path, monkeypatch):
    # Point the module at a throwaway directory so this test never touches
    # the real project tree, then verify every required directory appears.
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    created = file_utils_module.ensure_all_required_directories()

    assert len(created) == len(REQUIRED_DIRECTORIES)
    for rel in REQUIRED_DIRECTORIES:
        assert (tmp_path / rel).is_dir()

    # Calling it again must be a safe no-op (idempotent).
    created_again = file_utils_module.ensure_all_required_directories()
    assert len(created_again) == len(REQUIRED_DIRECTORIES)


def test_missing_required_directories_detects_gaps(tmp_path, monkeypatch):
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    # Nothing created yet -> everything should be reported missing.
    missing = file_utils_module.missing_required_directories()
    assert set(missing) == set(REQUIRED_DIRECTORIES)

    file_utils_module.ensure_all_required_directories()
    assert file_utils_module.missing_required_directories() == []
