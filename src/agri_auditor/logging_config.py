from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover - depends on environment setup
    structlog = None  # type: ignore[assignment]


def resolve_log_format(requested: str, stream: TextIO | None = None) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "json", "console"}:
        raise ValueError(
            f"Invalid log format '{requested}'. Expected one of ['auto', 'console', 'json']."
        )
    if normalized != "auto":
        return normalized

    out = stream if stream is not None else sys.stdout
    return "console" if getattr(out, "isatty", lambda: False)() else "json"


def configure_logging(log_level: str, log_format: str, stream: TextIO | None = None) -> str:
    effective_format = resolve_log_format(log_format, stream=stream)
    raw_level = str(log_level).strip()
    if raw_level.isdigit():
        numeric_level = int(raw_level)
    else:
        numeric_level = logging.getLevelName(raw_level.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level '{log_level}'.")

    logging.basicConfig(level=numeric_level, format="%(message)s", force=True)

    if structlog is None:
        return effective_format

    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if effective_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.reset_defaults()
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return effective_format


def get_logger(name: str) -> Any:
    if structlog is not None:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def log_event(logger: Any, level: str, event: str, **fields: Any) -> None:
    level_name = level.lower()
    if structlog is not None:
        getattr(logger, level_name)(event, **fields)
        return

    suffix = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    message = f"{event} {suffix}".strip()
    getattr(logger, level_name)(message)
