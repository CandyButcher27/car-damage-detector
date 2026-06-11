"""Resilience patterns: circuit breaker, retry-with-backoff, timeout.

The goal is fault containment, not fault elimination. Each pattern is
intentionally small: no third-party deps, no fancy state machines, just
enough to keep one bad downstream from taking down the whole pod.

CircuitBreaker
    classic CLOSED → OPEN → HALF_OPEN cycle. Increments a failure counter on
    every exception that isn't whitelisted; trips OPEN after
    ``failure_threshold`` consecutive failures and stays open for
    ``recovery_seconds`` before allowing a single probe call.

retry
    decorator for transient errors at startup (model load, etc.). Capped
    exponential backoff with jitter. Not used inside request paths — request
    retries belong to the client.

run_with_timeout
    runs a blocking callable in the default loop's threadpool and raises
    ``DependencyTimeoutError`` if it exceeds the deadline.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from starlette.concurrency import run_in_threadpool

from .errors import CircuitOpenError, DependencyTimeoutError
from .logging_setup import get_logger

T = TypeVar("T")

_log = get_logger("upsure.resilience")


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreaker:
    """Thread-safe circuit breaker.

    ``name`` is used in logs and exception messages so SREs can grep.
    ``ignored_exceptions`` are user-error-class exceptions (e.g. validation)
    that should not be counted against the downstream's health.
    """

    name: str
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    half_open_max_calls: int = 1
    ignored_exceptions: tuple[type[BaseException], ...] = ()

    _state: str = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_in_flight: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ── public ──────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failures": self._failures,
                "opened_at": self._opened_at,
                "recovery_seconds": self.recovery_seconds,
            }

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Synchronous call. Raises ``CircuitOpenError`` when the breaker is open."""
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            self._after_call(success=False, exc=exc)
            raise
        else:
            self._after_call(success=True)
            return result

    async def acall(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Async call. Awaits coroutines; runs sync funcs in a thread pool."""
        self._before_call()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await run_in_threadpool(func, *args, **kwargs)
        except BaseException as exc:
            self._after_call(success=False, exc=exc)
            raise
        else:
            self._after_call(success=True)
            return result

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = 0.0
            self._half_open_in_flight = 0

    # ── internals ───────────────────────────────────────────────────────
    def _before_call(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                if now - self._opened_at >= self.recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_in_flight = 0
                else:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is open; downstream is unhealthy.",
                        details={
                            "name": self.name,
                            "retry_after_seconds": max(
                                0.0,
                                round(self.recovery_seconds - (now - self._opened_at), 2),
                            ),
                        },
                    )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is probing recovery; please retry shortly.",
                        details={"name": self.name},
                    )
                self._half_open_in_flight += 1

    def _after_call(self, *, success: bool, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None and isinstance(exc, self.ignored_exceptions):
                # Don't count user errors against the downstream.
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                return

            if success:
                if self._state == CircuitState.HALF_OPEN:
                    _log.info(
                        "circuit recovered",
                        extra={"circuit": self.name, "event": "circuit.close"},
                    )
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_in_flight = 0
                return

            self._failures += 1
            if self._state == CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    _log.warning(
                        "circuit opened",
                        extra={
                            "circuit": self.name,
                            "event": "circuit.open",
                            "failures": self._failures,
                            "exception": repr(exc) if exc is not None else None,
                        },
                    )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_in_flight = 0


# ── Retry-with-backoff (startup only) ───────────────────────────────────────
def retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    do_not_retry: tuple[type[BaseException], ...] = (
        KeyboardInterrupt,
        SystemExit,
        # Missing files are PERMANENT failures, not transient. Retrying just
        # blocks the caller (~7s on the default 3 attempts) for no benefit.
        # If a file appears later, a new request triggers a fresh load.
        FileNotFoundError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Synchronous retry decorator with capped exponential backoff + jitter."""

    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except do_not_retry:
                    raise
                except retry_on as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    sleep_for = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    sleep_for += random.uniform(0, base_delay)
                    _log.warning(
                        "retrying",
                        extra={
                            "func": func.__qualname__,
                            "attempt": attempt,
                            "sleep_seconds": round(sleep_for, 3),
                            "exception": repr(exc),
                        },
                    )
                    time.sleep(sleep_for)
            assert last_exc is not None
            raise last_exc

        _wrapped.__wrapped__ = func  # type: ignore[attr-defined]
        _wrapped.__name__ = func.__name__
        _wrapped.__doc__ = func.__doc__
        return _wrapped

    return _decorator


# ── Timeout for blocking callables ──────────────────────────────────────────
async def run_with_timeout(
    func: Callable[..., T],
    *args: Any,
    timeout_seconds: float,
    label: str = "operation",
    **kwargs: Any,
) -> T:
    """Run a blocking callable in the threadpool with a deadline.

    Note: Python cannot truly cancel a thread; the worker will keep running
    in the background after we raise, but our request returns promptly so
    upstream callers (k8s LB, browser) don't pile up. The threadpool is
    sized by uvicorn and bounded.
    """
    try:
        return await asyncio.wait_for(
            run_in_threadpool(func, *args, **kwargs),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise DependencyTimeoutError(
            f"{label} exceeded the {timeout_seconds:.1f}s timeout.",
            details={"label": label, "timeout_seconds": timeout_seconds},
        ) from exc


# ── Concurrency limiter (bulkhead) ─────────────────────────────────────────
class Bulkhead:
    """Bounded concurrency limiter using an asyncio semaphore.

    Use one Bulkhead per downstream so e.g. a flood of OCR requests can't
    exhaust workers needed for damage inference.
    """

    def __init__(self, name: str, max_concurrent: int) -> None:
        self.name = name
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = asyncio.Semaphore(self.max_concurrent)

    async def __aenter__(self) -> "Bulkhead":
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()


def safe_call(
    func: Callable[..., T],
    *args: Any,
    ignored: Iterable[type[BaseException]] = (),
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Run a callable; on whitelisted exception return ``default``.

    Tiny utility for ``best-effort`` paths (e.g. ANPR on /predict/damage)
    where the main response should succeed even if a sidecar pipeline
    fails.
    """
    try:
        return func(*args, **kwargs)
    except tuple(ignored) as exc:  # type: ignore[misc]
        _log.warning(
            "safe_call swallowed",
            extra={"func": getattr(func, "__qualname__", str(func)), "exception": repr(exc)},
        )
        return default
