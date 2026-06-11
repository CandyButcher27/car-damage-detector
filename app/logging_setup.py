"""Structured JSON logging with request-id correlation.

Uses stdlib only so we don't add deps. Emits one JSON object per log record
with stable fields: ts, level, logger, msg, request_id (when set), and any
extras. A ContextVar holds the current request id so handlers can read it
without threading it through every call.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

from .settings import SETTINGS

_request_id_ctx: ContextVar[str | None] = ContextVar("upsure_request_id", default=None)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def set_request_id(value: str | None) -> None:
    _request_id_ctx.set(value)


_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "asctime", "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "service": SETTINGS.service_name,
            "version": SETTINGS.service_version,
            "env": SETTINGS.environment,
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        record.request_id = get_request_id() or "-"
        return super().format(record)


def configure_logging() -> None:
    """Idempotently configure root logging."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if SETTINGS.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter())

    root.addHandler(handler)
    root.setLevel(SETTINGS.log_level)

    # Quiet down noisy third-party libraries
    for noisy in ("uvicorn.access", "tensorflow", "absl", "paddle"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
