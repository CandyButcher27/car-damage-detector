"""Observability: request_id middleware, Prometheus metrics, payload guard.

``prometheus_client`` is optional. If not installed the metrics endpoint
just responds 503 with a structured envelope; everything else still works.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import ApiError, PayloadTooLargeError
from .logging_setup import get_logger, set_request_id
from .responses import json_error
from .settings import SETTINGS

_log = get_logger("upsure.http")

try:  # pragma: no cover - exercised at import time
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]


_REGISTRY = None
HTTP_REQUESTS_TOTAL = None
HTTP_REQUEST_LATENCY = None
MODEL_READINESS = None
CIRCUIT_STATE = None
PIPELINE_LATENCY = None


def init_metrics() -> None:
    """Idempotently create metrics collectors."""
    global _REGISTRY, HTTP_REQUESTS_TOTAL, HTTP_REQUEST_LATENCY, MODEL_READINESS
    global CIRCUIT_STATE, PIPELINE_LATENCY

    if not _METRICS_AVAILABLE or _REGISTRY is not None:
        return

    _REGISTRY = CollectorRegistry(auto_describe=True)
    HTTP_REQUESTS_TOTAL = Counter(
        "upsure_http_requests_total",
        "HTTP requests by route, method, status_class.",
        ["route", "method", "status_class"],
        registry=_REGISTRY,
    )
    HTTP_REQUEST_LATENCY = Histogram(
        "upsure_http_request_duration_seconds",
        "Request latency in seconds.",
        ["route", "method"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=_REGISTRY,
    )
    MODEL_READINESS = Gauge(
        "upsure_model_ready",
        "1 when a model is loaded and ready, 0 otherwise.",
        ["model"],
        registry=_REGISTRY,
    )
    CIRCUIT_STATE = Gauge(
        "upsure_circuit_state",
        "Circuit-breaker state. 0=closed, 1=half_open, 2=open.",
        ["circuit"],
        registry=_REGISTRY,
    )
    PIPELINE_LATENCY = Histogram(
        "upsure_pipeline_duration_seconds",
        "Latency of internal pipeline stages.",
        ["stage"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=_REGISTRY,
    )


# ── Helpers other modules can call ──────────────────────────────────────────
def set_model_readiness(model: str, ready: bool) -> None:
    if _METRICS_AVAILABLE and MODEL_READINESS is not None:
        MODEL_READINESS.labels(model=model).set(1.0 if ready else 0.0)


def set_circuit_state(name: str, state: str) -> None:
    if not (_METRICS_AVAILABLE and CIRCUIT_STATE is not None):
        return
    mapping = {"closed": 0, "half_open": 1, "open": 2}
    CIRCUIT_STATE.labels(circuit=name).set(mapping.get(state, 0))


def record_pipeline_latency(stage: str, seconds: float) -> None:
    if _METRICS_AVAILABLE and PIPELINE_LATENCY is not None:
        PIPELINE_LATENCY.labels(stage=stage).observe(max(0.0, seconds))


# ── Request-id + access-log middleware ──────────────────────────────────────
class RequestContextMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-ID"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        incoming = request.headers.get(self.HEADER)
        request_id = incoming if incoming and len(incoming) <= 128 else uuid.uuid4().hex
        set_request_id(request_id)
        # Stash on state for handlers that want it without importing the ctxvar.
        request.state.request_id = request_id
        request.state.start_perf = time.perf_counter()

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except ApiError as exc:
            response = json_error(exc, request=request, start_perf=request.state.start_perf)
            status_code = response.status_code
        finally:
            elapsed = time.perf_counter() - request.state.start_perf
            route = _route_template(request) or request.url.path
            method = request.method

            if _METRICS_AVAILABLE and HTTP_REQUESTS_TOTAL is not None:
                HTTP_REQUESTS_TOTAL.labels(
                    route=route, method=method, status_class=f"{status_code // 100}xx"
                ).inc()
                HTTP_REQUEST_LATENCY.labels(route=route, method=method).observe(elapsed)

            _log.info(
                "request handled",
                extra={
                    "event": "http.request",
                    "method": method,
                    "route": route,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": round(elapsed * 1000.0, 2),
                    "client": request.client.host if request.client else None,
                },
            )

        response.headers[self.HEADER] = request_id
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - request.state.start_perf) * 1000.0:.2f}"
        return response


def _route_template(request: Request) -> str | None:
    """Use the matched route template (e.g. ``/users/{id}``) for low-cardinality metrics."""
    route = request.scope.get("route")
    return getattr(route, "path", None) if route is not None else None


# ── Upload-size guard ───────────────────────────────────────────────────────
class MaxBodySizeMiddleware:
    """Reject requests whose ``Content-Length`` exceeds ``max_bytes``.

    Implemented as a raw ASGI middleware so the rejection happens before
    FastAPI starts buffering multipart parts. For streamed bodies without
    Content-Length we cap the cumulative bytes seen in ``receive``.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_bytes:
            await self._reject(scope, send, observed_bytes=content_length)
            return

        seen = 0
        capped = False

        async def receive_wrapped() -> dict:
            nonlocal seen, capped
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                seen += len(body)
                if seen > self.max_bytes:
                    capped = True
            return message

        if capped:  # pragma: no cover - guarded above
            await self._reject(scope, send, observed_bytes=seen)
            return

        await self.app(scope, receive_wrapped, send)

    async def _reject(self, scope: Scope, send: Send, *, observed_bytes: int) -> None:
        err = PayloadTooLargeError(
            f"Upload exceeds the {self.max_bytes // (1024 * 1024)} MB limit.",
            details={"limit_bytes": self.max_bytes, "observed_bytes": observed_bytes},
        )
        from .responses import envelope_error
        import json as _json

        body = _json.dumps(envelope_error(err)).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": err.http_status or 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return int(value.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                return None
    return None


# ── /metrics endpoint registration ──────────────────────────────────────────
def attach_metrics_endpoint(app: FastAPI) -> None:
    if not SETTINGS.metrics_enabled:
        return

    init_metrics()

    @app.get(SETTINGS.metrics_path, include_in_schema=False)
    def metrics_endpoint() -> Response:
        if not _METRICS_AVAILABLE or _REGISTRY is None:
            return PlainTextResponse(
                "# prometheus_client not installed; metrics unavailable.\n",
                status_code=503,
            )
        return Response(content=generate_latest(_REGISTRY), media_type=CONTENT_TYPE_LATEST)
