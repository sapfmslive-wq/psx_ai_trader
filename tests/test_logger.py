"""Tests for src.utils.logger (Phase 1)."""

from __future__ import annotations

import logging

import pytest

from src.utils import logger as logger_module


@pytest.fixture(autouse=True)
def _reset_logging():
    """Ensure each test starts and ends with no cached loggers/handlers."""
    logger_module.reset_loggers_for_tests()
    yield
    logger_module.reset_loggers_for_tests()


def test_get_logger_creates_expected_log_files(tmp_path, monkeypatch):
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    log = logger_module.get_logger("system")
    log.info("test message")

    assert (tmp_path / "logs" / "system.log").is_file()
    assert (tmp_path / "logs" / "error.log").is_file()


def test_error_messages_are_duplicated_into_error_log(tmp_path, monkeypatch):
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    log = logger_module.get_logger("data")
    log.error("something went wrong")

    for handler in log.handlers:
        handler.flush()

    error_log_content = (tmp_path / "logs" / "error.log").read_text(encoding="utf-8")
    assert "something went wrong" in error_log_content

    data_log_content = (tmp_path / "logs" / "data.log").read_text(encoding="utf-8")
    assert "something went wrong" in data_log_content


def test_get_logger_is_idempotent_no_duplicate_handlers(tmp_path, monkeypatch):
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    log_a = logger_module.get_logger("model")
    handler_count_first_call = len(log_a.handlers)

    log_b = logger_module.get_logger("model")
    handler_count_second_call = len(log_b.handlers)

    assert log_a is log_b
    assert handler_count_first_call == handler_count_second_call


def test_logger_does_not_propagate_to_root(tmp_path, monkeypatch):
    import src.utils.file_utils as file_utils_module

    monkeypatch.setattr(file_utils_module, "PROJECT_ROOT", tmp_path)

    log = logger_module.get_logger("trading")
    assert log.propagate is False
    assert log.level == logging.DEBUG
