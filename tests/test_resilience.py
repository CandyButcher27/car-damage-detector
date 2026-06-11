"""Tests for app.resilience — circuit breaker, retry, timeout."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.errors import ApiError, CircuitOpenError, DependencyTimeoutError
from app.resilience import CircuitBreaker, retry, run_with_timeout


# ── CircuitBreaker ──────────────────────────────────────────────────────────
def test_breaker_starts_closed():
    cb = CircuitBreaker(name="t1")
    assert cb.state == "closed"
    assert cb.failure_count == 0


def test_breaker_passes_through_success():
    cb = CircuitBreaker(name="t2")
    assert cb.call(lambda: 42) == 42
    assert cb.state == "closed"


def test_breaker_counts_failures():
    cb = CircuitBreaker(name="t3", failure_threshold=3)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cb.state == "closed"
    assert cb.failure_count == 2


def test_breaker_opens_at_threshold():
    cb = CircuitBreaker(name="t4", failure_threshold=3)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cb.state == "open"


def test_breaker_blocks_calls_while_open():
    cb = CircuitBreaker(name="t5", failure_threshold=1, recovery_seconds=60)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cb.state == "open"
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)


def test_breaker_recovers_after_timeout():
    cb = CircuitBreaker(name="t6", failure_threshold=1, recovery_seconds=0.05)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cb.state == "open"

    time.sleep(0.1)
    # In half-open, a successful call closes the circuit.
    assert cb.call(lambda: "ok") == "ok"
    assert cb.state == "closed"
    assert cb.failure_count == 0


def test_breaker_reopens_when_half_open_call_fails():
    cb = CircuitBreaker(name="t7", failure_threshold=1, recovery_seconds=0.05)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    time.sleep(0.1)

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still bad")))
    assert cb.state == "open"


def test_breaker_ignores_user_errors():
    class _UserErr(ApiError):
        pass

    cb = CircuitBreaker(
        name="t8", failure_threshold=2, ignored_exceptions=(_UserErr,),
    )
    for _ in range(5):
        with pytest.raises(_UserErr):
            cb.call(
                lambda: (_ for _ in ()).throw(
                    _UserErr(code="X", message="user error", retryable=False)
                )
            )
    assert cb.state == "closed"
    assert cb.failure_count == 0


def test_breaker_snapshot_contains_useful_fields():
    cb = CircuitBreaker(name="snapshot")
    snap = cb.snapshot()
    assert snap["name"] == "snapshot"
    assert snap["state"] == "closed"
    assert snap["recovery_seconds"] > 0


def test_breaker_reset_clears_state():
    cb = CircuitBreaker(name="reset", failure_threshold=1)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cb.state == "open"
    cb.reset()
    assert cb.state == "closed"
    assert cb.failure_count == 0


# ── retry decorator ─────────────────────────────────────────────────────────
def test_retry_succeeds_eventually():
    attempts = {"n": 0}

    @retry(attempts=3, base_delay=0.01, max_delay=0.02)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("flaky")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_retry_raises_after_attempts_exhausted():
    @retry(attempts=2, base_delay=0.01, max_delay=0.02)
    def always_fails():
        raise RuntimeError("forever")

    with pytest.raises(RuntimeError):
        always_fails()


def test_retry_does_not_swallow_keyboard_interrupt():
    @retry(attempts=3, base_delay=0.01)
    def quitter():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        quitter()


def test_retry_fails_fast_on_file_not_found():
    """A missing file is permanent — retrying just stalls the caller."""
    attempts = {"n": 0}

    @retry(attempts=5, base_delay=0.01)
    def missing_model():
        attempts["n"] += 1
        raise FileNotFoundError("model not found")

    with pytest.raises(FileNotFoundError):
        missing_model()
    assert attempts["n"] == 1, "expected single attempt, got {}".format(attempts["n"])


# ── run_with_timeout ────────────────────────────────────────────────────────
def test_run_with_timeout_returns_value():
    import asyncio

    async def _go():
        return await run_with_timeout(lambda: 7, timeout_seconds=1.0, label="quick")

    assert asyncio.run(_go()) == 7


def test_run_with_timeout_raises_dependency_timeout():
    import asyncio

    def _slow():
        time.sleep(0.5)
        return "too late"

    async def _go():
        return await run_with_timeout(_slow, timeout_seconds=0.05, label="slow")

    with pytest.raises(DependencyTimeoutError):
        asyncio.run(_go())
