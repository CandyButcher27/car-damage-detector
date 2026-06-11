"""Unified response envelope.

Every endpoint returns the same shape:

    {
      "success": true,
      "data": <endpoint-specific payload>,
      "error": null,
      "meta": {
        "request_id": "...",
        "endpoint": "/predict/damage",
        "api_version": "v1",
        "service_version": "1.1.0",
        "latency_ms": 234,
        "timestamp": "2026-06-11T12:34:56Z"
      }
    }

On failure:

    {
      "success": false,
      "data": null,
      "error": {
        "code": "MODEL_UNAVAILABLE",
        "message": "...",
        "retryable": true,
        "details": { ... } | null
      },
      "meta": { ... }
    }

UI clients can therefore branch on a single boolean and surface
``error.message`` directly to the user without per-endpoint special casing.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from .errors import ApiError, sanitize_text
from .logging_setup import get_request_id
from .settings import SETTINGS


API_VERSION = "v1"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _meta(
    request: Request | None,
    *,
    start_perf: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency_ms: float | None = None
    if start_perf is not None:
        latency_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)

    meta: dict[str, Any] = {
        "request_id": get_request_id(),
        "endpoint": str(request.url.path) if request is not None else None,
        "api_version": API_VERSION,
        "service_version": SETTINGS.service_version,
        "latency_ms": latency_ms,
        "timestamp": _now_iso(),
    }
    if extra:
        meta.update(extra)
    return meta


def envelope_success(
    data: Any,
    *,
    request: Request | None = None,
    start_perf: float | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "error": None,
        "meta": _meta(request, start_perf=start_perf, extra=meta_extra),
    }


def envelope_error(
    error: ApiError,
    *,
    request: Request | None = None,
    start_perf: float | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": {
            "code": error.code,
            "message": sanitize_text(error.message),
            "retryable": error.retryable,
            "details": error.details,
        },
        "meta": _meta(request, start_perf=start_perf, extra=meta_extra),
    }


def json_success(
    data: Any,
    *,
    request: Request | None = None,
    start_perf: float | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        content=envelope_success(data, request=request, start_perf=start_perf),
        status_code=status_code,
        headers=headers,
    )


def json_error(
    error: ApiError,
    *,
    request: Request | None = None,
    start_perf: float | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        content=envelope_error(error, request=request, start_perf=start_perf),
        status_code=error.http_status or 500,
        headers=headers,
    )
