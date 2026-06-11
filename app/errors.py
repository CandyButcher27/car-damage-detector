"""Custom exceptions and error sanitization.

A small taxonomy keeps error handling consistent across endpoints and lets
the middleware translate every failure into the same envelope.

Codes are stable and safe to expose to UI consumers; messages are
human-readable; ``retryable`` tells the client whether a retry has any chance
of succeeding without intervention.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .settings import SETTINGS


# ── Error codes ─────────────────────────────────────────────────────────────
class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    NOT_FOUND = "NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    DEPENDENCY_TIMEOUT = "DEPENDENCY_TIMEOUT"
    PIPELINE_FAILURE = "PIPELINE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_HTTP_BY_CODE: dict[str, int] = {
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.UNSUPPORTED_MEDIA: 415,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.CIRCUIT_OPEN: 503,
    ErrorCode.DEPENDENCY_TIMEOUT: 504,
    ErrorCode.PIPELINE_FAILURE: 502,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ApiError(Exception):
    """Base error type — every endpoint failure should be one of these.

    Plain Exception (not a dataclass) so subclasses can inherit cleanly. The
    field-by-field constructor signature is preserved for callers.
    """

    __slots__ = ("code", "message", "retryable", "details", "http_status")

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details
        self.http_status = http_status if http_status is not None else _HTTP_BY_CODE.get(code, 500)
        super().__init__(f"{code}: {message}")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"{type(self).__name__}(code={self.code!r}, message={self.message!r}, "
            f"retryable={self.retryable!r}, details={self.details!r}, "
            f"http_status={self.http_status!r})"
        )


class ValidationError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.VALIDATION_ERROR, message, retryable=False, details=details)


class UnsupportedMediaError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.UNSUPPORTED_MEDIA, message, retryable=False, details=details)


class PayloadTooLargeError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.PAYLOAD_TOO_LARGE, message, retryable=False, details=details)


class ModelUnavailableError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.MODEL_UNAVAILABLE, message, retryable=True, details=details)


class CircuitOpenError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CIRCUIT_OPEN, message, retryable=True, details=details)


class DependencyTimeoutError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.DEPENDENCY_TIMEOUT, message, retryable=True, details=details)


class PipelineFailureError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.PIPELINE_FAILURE, message, retryable=False, details=details)


# ── Sanitization ────────────────────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """Strip absolute file paths and home-directory hints from a message.

    Production responses must not leak server paths or usernames. Logs keep
    full detail; HTTP responses get the sanitized form.
    """
    if not text:
        return text
    if SETTINGS.log_include_paths:
        return text

    cleaned = text
    home = os.path.expanduser("~")
    if home and home != "~":
        cleaned = cleaned.replace(home, "~")

    try:
        cwd = str(Path.cwd())
        cleaned = cleaned.replace(cwd, ".")
    except Exception:
        pass

    return cleaned


def to_api_error(exc: BaseException) -> ApiError:
    """Translate an arbitrary exception into a safe ApiError."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, FileNotFoundError):
        return ModelUnavailableError(
            "Required model or resource was not found on the server.",
            details={"hint": "Check model artifacts on the pod's volume mount."},
        )
    if isinstance(exc, TimeoutError):
        return DependencyTimeoutError("A downstream operation timed out.")
    return ApiError(
        code=ErrorCode.INTERNAL_ERROR,
        message=sanitize_text(str(exc)) or "Unexpected internal error.",
        retryable=False,
    )
