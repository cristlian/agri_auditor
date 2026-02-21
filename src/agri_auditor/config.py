from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_TIMEOUT_SEC = 20.0
DEFAULT_GEMINI_WORKERS = 4
DEFAULT_GEMINI_RETRIES = 2
DEFAULT_GEMINI_BACKOFF_MS = 250
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "auto"
DEFAULT_DEPTH_WORKERS = 4
DEFAULT_SCORE_NORMALIZATION = "robust"
DEFAULT_SCORE_ROBUST_QUANTILE_LOW = 0.05
DEFAULT_SCORE_ROBUST_QUANTILE_HIGH = 0.95
DEFAULT_PEAK_PROMINENCE = 0.05
DEFAULT_PEAK_WIDTH = 1
DEFAULT_PEAK_MIN_DISTANCE = 150
DEFAULT_REPORT_MODE = "single"
DEFAULT_REPORT_TELEMETRY_DOWNSAMPLE = 1

_VALID_LOG_FORMATS = {"auto", "json", "console"}
_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
_VALID_SCORE_NORMALIZATION = {"minmax", "robust"}
_VALID_REPORT_MODES = {"single", "split"}


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str = DEFAULT_LOG_LEVEL
    log_format: str = DEFAULT_LOG_FORMAT
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_timeout_sec: float = DEFAULT_GEMINI_TIMEOUT_SEC
    gemini_workers: int = DEFAULT_GEMINI_WORKERS
    gemini_retries: int = DEFAULT_GEMINI_RETRIES
    gemini_backoff_ms: int = DEFAULT_GEMINI_BACKOFF_MS
    gemini_cache_dir: str | None = None
    depth_workers: int = DEFAULT_DEPTH_WORKERS
    depth_cache_dir: str | None = None
    score_normalization: str = DEFAULT_SCORE_NORMALIZATION
    score_robust_quantile_low: float = DEFAULT_SCORE_ROBUST_QUANTILE_LOW
    score_robust_quantile_high: float = DEFAULT_SCORE_ROBUST_QUANTILE_HIGH
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE
    peak_width: int = DEFAULT_PEAK_WIDTH
    peak_min_distance: int = DEFAULT_PEAK_MIN_DISTANCE
    report_mode: str = DEFAULT_REPORT_MODE
    report_telemetry_downsample: int = DEFAULT_REPORT_TELEMETRY_DOWNSAMPLE
    report_feature_columns: tuple[str, ...] = ()


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
    return str(numeric)


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


def parse_non_negative_float(value: str, *, env_var: str) -> float:
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid {env_var} '{value}'. Expected a non-negative float."
        ) from exc
    if parsed < 0:
        raise ValueError(f"Invalid {env_var} '{value}'. Value must be >= 0.")
    return parsed


def parse_positive_int(value: str, *, env_var: str) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {env_var} '{value}'. Expected a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"Invalid {env_var} '{value}'. Value must be > 0.")
    return parsed


def parse_non_negative_int(value: str, *, env_var: str) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid {env_var} '{value}'. Expected a non-negative integer."
        ) from exc
    if parsed < 0:
        raise ValueError(f"Invalid {env_var} '{value}'. Value must be >= 0.")
    return parsed


def parse_optional_dir(value: str, *, env_var: str) -> str | None:
    stripped = str(value).strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())


def normalize_score_normalization(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _VALID_SCORE_NORMALIZATION:
        raise ValueError(
            f"Invalid AGRI_AUDITOR_SCORE_NORMALIZATION '{value}'. "
            f"Expected one of {sorted(_VALID_SCORE_NORMALIZATION)}."
        )
    return normalized


def normalize_report_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _VALID_REPORT_MODES:
        raise ValueError(
            f"Invalid AGRI_AUDITOR_REPORT_MODE '{value}'. "
            f"Expected one of {sorted(_VALID_REPORT_MODES)}."
        )
    return normalized


def parse_feature_columns(value: str) -> tuple[str, ...]:
    columns = [item.strip() for item in str(value).split(",")]
    cleaned = tuple(col for col in columns if col)
    return cleaned


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
    gemini_workers = parse_positive_int(
        source.get("AGRI_AUDITOR_GEMINI_WORKERS", str(DEFAULT_GEMINI_WORKERS)),
        env_var="AGRI_AUDITOR_GEMINI_WORKERS",
    )
    gemini_retries = parse_non_negative_int(
        source.get("AGRI_AUDITOR_GEMINI_RETRIES", str(DEFAULT_GEMINI_RETRIES)),
        env_var="AGRI_AUDITOR_GEMINI_RETRIES",
    )
    gemini_backoff_ms = parse_non_negative_int(
        source.get("AGRI_AUDITOR_GEMINI_BACKOFF_MS", str(DEFAULT_GEMINI_BACKOFF_MS)),
        env_var="AGRI_AUDITOR_GEMINI_BACKOFF_MS",
    )
    gemini_cache_dir = parse_optional_dir(
        source.get("AGRI_AUDITOR_GEMINI_CACHE_DIR", ""),
        env_var="AGRI_AUDITOR_GEMINI_CACHE_DIR",
    )
    depth_workers = parse_positive_int(
        source.get("AGRI_AUDITOR_DEPTH_WORKERS", str(DEFAULT_DEPTH_WORKERS)),
        env_var="AGRI_AUDITOR_DEPTH_WORKERS",
    )
    depth_cache_dir = parse_optional_dir(
        source.get("AGRI_AUDITOR_DEPTH_CACHE_DIR", ""),
        env_var="AGRI_AUDITOR_DEPTH_CACHE_DIR",
    )
    score_normalization = normalize_score_normalization(
        source.get("AGRI_AUDITOR_SCORE_NORMALIZATION", DEFAULT_SCORE_NORMALIZATION)
    )
    score_robust_quantile_low = parse_non_negative_float(
        source.get(
            "AGRI_AUDITOR_SCORE_Q_LOW",
            str(DEFAULT_SCORE_ROBUST_QUANTILE_LOW),
        ),
        env_var="AGRI_AUDITOR_SCORE_Q_LOW",
    )
    score_robust_quantile_high = parse_non_negative_float(
        source.get(
            "AGRI_AUDITOR_SCORE_Q_HIGH",
            str(DEFAULT_SCORE_ROBUST_QUANTILE_HIGH),
        ),
        env_var="AGRI_AUDITOR_SCORE_Q_HIGH",
    )
    if score_robust_quantile_low >= score_robust_quantile_high:
        raise ValueError(
            "Invalid robust quantiles: AGRI_AUDITOR_SCORE_Q_LOW must be < AGRI_AUDITOR_SCORE_Q_HIGH."
        )
    if score_robust_quantile_high > 1.0:
        raise ValueError("Invalid AGRI_AUDITOR_SCORE_Q_HIGH. Value must be <= 1.0.")
    peak_prominence = parse_non_negative_float(
        source.get("AGRI_AUDITOR_PEAK_PROMINENCE", str(DEFAULT_PEAK_PROMINENCE)),
        env_var="AGRI_AUDITOR_PEAK_PROMINENCE",
    )
    peak_width = parse_positive_int(
        source.get("AGRI_AUDITOR_PEAK_WIDTH", str(DEFAULT_PEAK_WIDTH)),
        env_var="AGRI_AUDITOR_PEAK_WIDTH",
    )
    peak_min_distance = parse_positive_int(
        source.get("AGRI_AUDITOR_PEAK_MIN_DISTANCE", str(DEFAULT_PEAK_MIN_DISTANCE)),
        env_var="AGRI_AUDITOR_PEAK_MIN_DISTANCE",
    )
    report_mode = normalize_report_mode(
        source.get("AGRI_AUDITOR_REPORT_MODE", DEFAULT_REPORT_MODE)
    )
    report_telemetry_downsample = parse_positive_int(
        source.get(
            "AGRI_AUDITOR_REPORT_TELEMETRY_DOWNSAMPLE",
            str(DEFAULT_REPORT_TELEMETRY_DOWNSAMPLE),
        ),
        env_var="AGRI_AUDITOR_REPORT_TELEMETRY_DOWNSAMPLE",
    )
    report_feature_columns = parse_feature_columns(
        source.get("AGRI_AUDITOR_REPORT_FEATURE_COLUMNS", "")
    )

    return RuntimeConfig(
        log_level=log_level,
        log_format=log_format,
        gemini_model=gemini_model,
        gemini_timeout_sec=gemini_timeout_sec,
        gemini_workers=gemini_workers,
        gemini_retries=gemini_retries,
        gemini_backoff_ms=gemini_backoff_ms,
        gemini_cache_dir=gemini_cache_dir,
        depth_workers=depth_workers,
        depth_cache_dir=depth_cache_dir,
        score_normalization=score_normalization,
        score_robust_quantile_low=score_robust_quantile_low,
        score_robust_quantile_high=score_robust_quantile_high,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
        peak_min_distance=peak_min_distance,
        report_mode=report_mode,
        report_telemetry_downsample=report_telemetry_downsample,
        report_feature_columns=report_feature_columns,
    )
