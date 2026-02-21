from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_TIMEOUT_SEC = 20.0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "auto"

_VALID_LOG_FORMATS = {"auto", "json", "console"}
_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str = DEFAULT_LOG_LEVEL
    log_format: str = DEFAULT_LOG_FORMAT
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_timeout_sec: float = DEFAULT_GEMINI_TIMEOUT_SEC


def normalize_log_level(value: str) -> str:
    level = str(value).strip().upper()
    if level in _VALID_LOG_LEVELS:
        return level

    # Also allow numeric logging levels.
    try:
        numeric = int(level)
    except ValueError as exc:
        raise ValueError(
            f"Invalid AGRI_AUDITOR_LOG_LEVEL '{value}'. "
            f"Expected one of {sorted(_VALID_LOG_LEVELS)} or a numeric level."
        ) from exc

    if numeric < 0:
        raise ValueError(
            f"Invalid AGRI_AUDITOR_LOG_LEVEL '{value}'. Numeric levels must be >= 0."
        )
    return logging.getLevelName(numeric)


def normalize_log_format(value: str) -> str:
    fmt = str(value).strip().lower()
    if fmt not in _VALID_LOG_FORMATS:
        raise ValueError(
            f"Invalid AGRI_AUDITOR_LOG_FORMAT '{value}'. "
            f"Expected one of {sorted(_VALID_LOG_FORMATS)}."
        )
    return fmt


def parse_positive_float(value: str, *, env_var: str) -> float:
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {env_var} '{value}'. Expected a positive float.") from exc
    if parsed <= 0:
        raise ValueError(f"Invalid {env_var} '{value}'. Value must be > 0.")
    return parsed


def load_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    source = env if env is not None else os.environ

    log_level = normalize_log_level(source.get("AGRI_AUDITOR_LOG_LEVEL", DEFAULT_LOG_LEVEL))
    log_format = normalize_log_format(source.get("AGRI_AUDITOR_LOG_FORMAT", DEFAULT_LOG_FORMAT))
    gemini_model = source.get("AGRI_AUDITOR_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not gemini_model:
        raise ValueError("AGRI_AUDITOR_GEMINI_MODEL must not be empty.")
    gemini_timeout_sec = parse_positive_float(
        source.get("AGRI_AUDITOR_GEMINI_TIMEOUT_SEC", str(DEFAULT_GEMINI_TIMEOUT_SEC)),
        env_var="AGRI_AUDITOR_GEMINI_TIMEOUT_SEC",
    )

    return RuntimeConfig(
        log_level=log_level,
        log_format=log_format,
        gemini_model=gemini_model,
        gemini_timeout_sec=gemini_timeout_sec,
    )
