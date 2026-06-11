"""Environment-driven settings.

Kept dependency-free — we read from `os.environ` with sensible defaults so the
container can be tuned via Kubernetes ConfigMap/Secret without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_list(name: str, default: Sequence[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True, frozen=True)
class Settings:
    # General
    service_name: str = os.getenv("UPSURE_SERVICE_NAME", "data-ingestion")
    service_version: str = os.getenv("UPSURE_SERVICE_VERSION", "1.1.0")
    environment: str = os.getenv("UPSURE_ENV", "development")
    api_prefix: str = os.getenv("UPSURE_API_PREFIX", "")

    # HTTP / CORS
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(_env_list("UPSURE_CORS_ORIGINS", ["*"]))
    )
    cors_allow_credentials: bool = _env_bool("UPSURE_CORS_ALLOW_CREDENTIALS", False)
    max_upload_bytes: int = _env_int("UPSURE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)  # 25 MB
    request_timeout_seconds: float = _env_float("UPSURE_REQUEST_TIMEOUT_SECONDS", 120.0)

    # Logging
    log_level: str = os.getenv("UPSURE_LOG_LEVEL", "INFO").upper()
    log_json: bool = _env_bool("UPSURE_LOG_JSON", True)
    log_include_paths: bool = _env_bool("UPSURE_LOG_INCLUDE_PATHS", False)

    # Metrics
    metrics_enabled: bool = _env_bool("UPSURE_METRICS_ENABLED", True)
    metrics_path: str = os.getenv("UPSURE_METRICS_PATH", "/metrics")

    # Model preload (turn off in dev for fast iteration)
    preload_models_on_startup: bool = _env_bool("UPSURE_PRELOAD_MODELS", False)
    require_models_for_ready: bool = _env_bool("UPSURE_REQUIRE_MODELS_FOR_READY", True)

    # Circuit-breaker defaults
    cb_failure_threshold: int = _env_int("UPSURE_CB_FAILURE_THRESHOLD", 5)
    cb_recovery_seconds: float = _env_float("UPSURE_CB_RECOVERY_SECONDS", 30.0)
    cb_half_open_max_calls: int = _env_int("UPSURE_CB_HALF_OPEN_MAX_CALLS", 1)

    # Subprocess (OCR) timeout
    ocr_subprocess_timeout_seconds: float = _env_float(
        "UPSURE_OCR_SUBPROCESS_TIMEOUT_SECONDS", 90.0
    )

    # Inference concurrency limits (bulkhead)
    damage_concurrency: int = _env_int("UPSURE_DAMAGE_CONCURRENCY", 4)
    ocr_concurrency: int = _env_int("UPSURE_OCR_CONCURRENCY", 2)


SETTINGS = Settings()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
