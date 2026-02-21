from __future__ import annotations

import io

import pytest

from agri_auditor.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TIMEOUT_SEC,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    RuntimeConfig,
    load_runtime_config,
)
from agri_auditor.logging_config import configure_logging, get_logger, log_event, resolve_log_format


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "AGRI_AUDITOR_LOG_LEVEL",
        "AGRI_AUDITOR_LOG_FORMAT",
        "AGRI_AUDITOR_GEMINI_MODEL",
        "AGRI_AUDITOR_GEMINI_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_runtime_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    config = load_runtime_config()
    assert isinstance(config, RuntimeConfig)
    assert config.log_level == DEFAULT_LOG_LEVEL
    assert config.log_format == DEFAULT_LOG_FORMAT
    assert config.gemini_model == DEFAULT_GEMINI_MODEL
    assert config.gemini_timeout_sec == DEFAULT_GEMINI_TIMEOUT_SEC


def test_load_runtime_config_invalid_log_format(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("AGRI_AUDITOR_LOG_FORMAT", "invalid-format")
    with pytest.raises(ValueError, match="AGRI_AUDITOR_LOG_FORMAT"):
        load_runtime_config()


def test_load_runtime_config_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("AGRI_AUDITOR_GEMINI_TIMEOUT_SEC", "0")
    with pytest.raises(ValueError, match="AGRI_AUDITOR_GEMINI_TIMEOUT_SEC"):
        load_runtime_config()


def test_resolve_log_format_auto_non_tty_is_json() -> None:
    stream = io.StringIO()
    assert resolve_log_format("auto", stream=stream) == "json"


def test_resolve_log_format_auto_tty_is_console() -> None:
    class _TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert resolve_log_format("auto", stream=_TtyBuffer()) == "console"


def test_configure_logging_accepts_json_and_console() -> None:
    assert configure_logging("INFO", "json") == "json"
    assert configure_logging("INFO", "console") == "console"


def test_log_event_emits_without_error() -> None:
    configure_logging("INFO", "json")
    logger = get_logger("test-config-logging")
    log_event(logger, "info", "config_logging_test", key="value", count=1)
