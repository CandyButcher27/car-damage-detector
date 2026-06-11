"""K8s health probes — split into liveness and readiness.

* ``/livez``  — cheap, never blocks. Returns 200 as long as the event loop
  runs. Kubernetes will restart the pod if this fails for too long.
* ``/readyz`` — checks every dependency (models, optional ANPR) and only
  returns 200 when the pod can actually serve traffic. Kubernetes will
  pull the pod out of Service endpoints (no traffic) while this is red.
* ``/health`` — legacy alias kept for the existing UI / scripts. Same
  envelope as everything else but includes detailed component data.

Each probe accepts a ``HealthRegistry`` of named checks so any module can
register a check at import or startup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import APIRouter, Request

from .logging_setup import get_logger
from .observability import set_model_readiness
from .responses import envelope_success, json_error, json_success
from .errors import ApiError, ErrorCode

_log = get_logger("upsure.health")


CheckFn = Callable[[], "Awaitable[ComponentStatus] | ComponentStatus"]


@dataclass(slots=True)
class ComponentStatus:
    name: str
    ready: bool
    critical: bool = True
    detail: str | None = None
    extra: dict | None = None

    def as_dict(self) -> dict:
        payload: dict = {
            "name": self.name,
            "ready": self.ready,
            "critical": self.critical,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.extra:
            payload["extra"] = self.extra
        return payload


class HealthRegistry:
    def __init__(self) -> None:
        self._checks: list[tuple[str, CheckFn]] = []

    def register(self, name: str, fn: CheckFn) -> None:
        self._checks.append((name, fn))

    async def run_all(self) -> list[ComponentStatus]:
        out: list[ComponentStatus] = []
        for name, fn in self._checks:
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as exc:  # one bad check shouldn't kill the probe
                _log.exception("health check failed", extra={"check": name})
                result = ComponentStatus(
                    name=name, ready=False, critical=True, detail=str(exc)
                )
            out.append(result)
            set_model_readiness(name, bool(result.ready))
        return out


registry = HealthRegistry()


def build_router() -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/livez", include_in_schema=False)
    async def livez(request: Request):
        return json_success(
            {"status": "alive", "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            request=request,
            start_perf=request.state.start_perf,
        )

    @router.get("/readyz", include_in_schema=False)
    async def readyz(request: Request):
        components = await registry.run_all()
        critical_failed = [c for c in components if c.critical and not c.ready]
        if critical_failed:
            error = ApiError(
                code=ErrorCode.MODEL_UNAVAILABLE,
                message="One or more required components are not ready.",
                retryable=True,
                details={"components": [c.as_dict() for c in components]},
            )
            return json_error(error, request=request, start_perf=request.state.start_perf)

        return json_success(
            {"status": "ready", "components": [c.as_dict() for c in components]},
            request=request,
            start_perf=request.state.start_perf,
        )

    @router.get("/health")
    async def health(request: Request):
        components = await registry.run_all()
        payload = {
            "status": "ok",
            "components": [c.as_dict() for c in components],
        }
        return json_success(payload, request=request, start_perf=request.state.start_perf)

    return router


__all__ = ["registry", "build_router", "ComponentStatus", "HealthRegistry"]
