#!/usr/local/bin/python3
"""The project logger initializes from football_predictions.toml and writes to its log file."""

import shutil

import pytest

import custom_logger
from football_predictions.config import DEFAULT_CONFIG_PATH

TEST_MESSAGE = "logger smoke test"


@pytest.fixture
def fresh_logger(tmp_path, monkeypatch):
    """Point the logger at a temp folder holding a copy of the real config; restore handlers afterwards."""
    shutil.copy(DEFAULT_CONFIG_PATH, tmp_path / DEFAULT_CONFIG_PATH.name)
    monkeypatch.setattr(custom_logger, "PROJECT_DIR", tmp_path)
    saved_handler_lst = custom_logger.custom_logger.handlers[:]
    custom_logger.custom_logger.handlers.clear()
    yield tmp_path
    for handler in custom_logger.custom_logger.handlers:
        handler.close()
    custom_logger.custom_logger.handlers[:] = saved_handler_lst


def test_setup_logger_uses_config_level_and_writes_log_file(fresh_logger):
    prog_name = DEFAULT_CONFIG_PATH.stem
    logger = custom_logger.setup_logger(prog_name)
    custom_logger.setup_logger(prog_name)                      # idempotent: no duplicate handlers
    assert len(logger.handlers) == 2
    assert logger.level == custom_logger.logging.getLevelName(custom_logger._default_log_level(DEFAULT_CONFIG_PATH))
    logger.warning(TEST_MESSAGE)
    for handler in logger.handlers:
        handler.flush()
    assert TEST_MESSAGE in (fresh_logger / f"{prog_name}.log").read_text()
