"""Centralised logging configuration (stdlib only, JSON in production)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


class KeyValueFormatter(logging.Formatter):
    """Human readable ``time level logger message key=value`` formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} "
            f"{record.levelname:<7} {record.name:<28} {record.getMessage()}"
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED_ATTRS and not k.startswith("_") and not k.startswith("msg")
        }
        if extras:
            base += " " + " ".join(f"{k}={_short(v)}" for k, v in sorted(extras.items()))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for log shipping."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_RESERVED_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime", "taskName"}


def _short(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def setup_logging(level: str | int = "INFO", *, json_logs: bool = False, force: bool = False) -> None:
    """Idempotently configure the root logger.

    Parameters
    ----------
    level:
        Log level name or numeric level.
    json_logs:
        Emit single-line JSON records (recommended for containers).
    force:
        Re-configure even when logging was already set up (used by tests).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else KeyValueFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else logging.getLevelName(str(level).upper()))

    # Tame chatty third-party loggers.
    for noisy, lvl in (("uvicorn.access", logging.WARNING), ("httpx", logging.WARNING), ("matplotlib", logging.WARNING)):
        logging.getLogger(noisy).setLevel(lvl)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module level logger with the SIHPS namespace."""
    return logging.getLogger(name if name.startswith("app") else f"app.{name}")
