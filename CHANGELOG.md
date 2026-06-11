# Changelog

All notable changes to this project are tracked here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-11

This is the production-hardening release. Every endpoint now returns a
unified response envelope, observability primitives are in place, and the
service can be deployed to a single Kubernetes pod that fetches its model
artefacts at startup.

### Added
- Unified response envelope `{success, data, error, meta}` on every endpoint.
  `meta` carries `request_id`, `endpoint`, `api_version`, `service_version`,
  `latency_ms`, and `timestamp`.
- Stable error taxonomy (`VALIDATION_ERROR`, `UNSUPPORTED_MEDIA`,
  `PAYLOAD_TOO_LARGE`, `MODEL_UNAVAILABLE`, `CIRCUIT_OPEN`,
  `DEPENDENCY_TIMEOUT`, `PIPELINE_FAILURE`, `INTERNAL_ERROR`,
  `NOT_FOUND`) with `retryable` flag so clients can branch deterministically.
- `app/` package owns cross-cutting concerns:
  - `app.settings` — env-driven config
  - `app.logging_setup` — JSON-line structured logger with `request_id` ContextVar
  - `app.observability` — request-ID middleware, Prometheus metrics
    (`upsure_http_requests_total`, `upsure_http_request_duration_seconds`,
    `upsure_pipeline_duration_seconds`, `upsure_model_ready`,
    `upsure_circuit_state`), and an ASGI body-size guard
  - `app.resilience` — `CircuitBreaker`, `retry` decorator, async
    `run_with_timeout`, `Bulkhead`
  - `app.responses` / `app.errors` — envelope helpers + sanitization
  - `app.health` — k8s probes `/livez`, `/readyz`, plus legacy `/health`
- `onnx_inference.py` — generic `BinaryOnnxImageClassifier` and
  `YoloOnnxDetector` (pulled from the `intern` branch).
- Batched damage inference: `/predict/damage` runs one ONNX `session.run`
  for *all* submitted views, then per-view YOLO localisation.
- k8s manifests (`k8s/`): single-replica `Deployment` with
  liveness/readiness/startup probes, an `initContainer` that pulls model
  artefacts over HTTPS into an `emptyDir`, ConfigMap, ClusterIP `Service`.
- `Dockerfile` (multi-stage), `.dockerignore`, `docker-compose.yml`,
  `.env.example`, `CHANGELOG.md`.
- New env knobs:
  - `UPSURE_CAR_THRESHOLD` (default `0.65`)
  - `UPSURE_CAR_MODEL`, `UPSURE_CAR_MODEL_POSITIVE_HIGH`
  - `UPSURE_MODELS_BASE_URL`, `UPSURE_DAMAGE_MODEL_FILE`,
    `UPSURE_YOLO_MODEL_FILE`, `UPSURE_CAR_MODEL_FILE`
  - `UPSURE_LOG_JSON`, `UPSURE_LOG_LEVEL`, `UPSURE_LOG_INCLUDE_PATHS`
  - `UPSURE_PRELOAD_MODELS`, `UPSURE_REQUIRE_MODELS_FOR_READY`
  - `UPSURE_CB_FAILURE_THRESHOLD`, `UPSURE_CB_RECOVERY_SECONDS`,
    `UPSURE_CB_HALF_OPEN_MAX_CALLS`
  - `UPSURE_DAMAGE_CONCURRENCY`, `UPSURE_OCR_CONCURRENCY`
  - `UPSURE_OCR_SUBPROCESS_TIMEOUT_SECONDS`
  - `UPSURE_MAX_UPLOAD_BYTES`, `UPSURE_REQUEST_TIMEOUT_SECONDS`
- `tests/e2e_smoke.py` and `tests/e2e_full.py` — runtime smoke and full
  scenario suite that hits a live container and dumps the envelope of every
  case to `reports/*.json`.
- `tests/test_resilience.py` — 15 focused tests for the circuit breaker,
  retry decorator, and timeout helper.

### Changed
- **BREAKING**: every HTTP response now returns the envelope above. Clients
  that previously read `data["damage_detected"]` directly must now read
  `data["data"]["damage_detected"]`. See "Migration" below.
- `CAR_THRESHOLD` default raised from `0.35` → `0.65`. This eliminated the
  only false positive observed in the E2E benchmark
  (`Non_card_image_1.jpeg`); the resulting confusion matrix on the
  15-sample set is `TP=5, TN=10, FP=0, FN=0`.
- Car classifier prefers `.onnx` over `.keras`. Resolution order is the
  user override (`UPSURE_CAR_MODEL`), then `models/best_car_model_v2.onnx`,
  then `models/digiLifeDoc_best_car_model_v2.onnx`, then `.keras`.
- Damage model-load `FileNotFoundError` is no longer retried (was 3 attempts
  with backoff, ~7 s wasted per missing-file 503). Permanent failures fail
  fast; transient errors still retry.
- OCR subprocess now runs with a `timeout` and is wrapped by a circuit
  breaker.
- Damage / YOLO / OCR / ANPR each have a dedicated circuit breaker
  (`DAMAGE_CB`, `YOLO_CB`, `OCR_CB`, `ANPR_CB`).
- PIL `Image.MAX_IMAGE_PIXELS` set to 64 megapixels to defuse the
  decompression-bomb attack.
- Body-size guard rejects uploads >25 MB before FastAPI buffers anything.
- `requirements.txt` trimmed:
  - `tensorflow` → `tensorflow-cpu`
  - dropped `opencv-contrib-python` (kept `opencv-python-headless`)
  - dropped `tensorboard`, `google-genai`, `google-auth`, `google-pasta`,
    `modelscope`, `aistudio-sdk`, `pyreadline3`
  - added `arabic-reshaper` (was imported by `ocr_simple_test.py` but not
    pinned)
  - added `prometheus_client`, `gunicorn`
- Dockerfile pip install uses `--ignore-installed` so packages already in
  the base image (notably `packaging`) are materialised under `/install`.
- Dockerfile creates `/home/upsure` with PaddleOCR/PaddleX subdirs so the
  non-root runtime user can write its cache.
- PaddleOCR weights pre-warmed at image build time (default on, controlled
  by `--build-arg PREWARM_PADDLE`).
- k8s switched from PVC + HPA + PDB to a single replica with an init
  container that fetches models. `strategy: Recreate`. No HPA, no PDB,
  no PVC.

### Removed
- `car_classifier_api.py` (standalone service) — `/predict/` on the main
  API now serves the same workflow via the same ONNX classifier.
- `keras==3.12.2` from requirements — the ONNX path is primary; the
  `tf.keras` fallback in `tensorflow-cpu` still loads any legacy `.keras`
  artefact that surfaces.
- `k8s/hpa.yaml`, `k8s/pdb.yaml`, `k8s/pvc-models.yaml` — single-replica
  stateless deployment doesn't need them.

### Fixed
- Endpoint catches that masked model-load failures as `415
  UNSUPPORTED_MEDIA` (should be `503 MODEL_UNAVAILABLE`).
- `@dataclass(slots=True)` + inheritance bug in `app.errors.ApiError`;
  subclasses now inherit cleanly.
- Health checks no longer trigger the retrying loader, so probes return
  in milliseconds instead of seconds.

### Security
- CORS is now driven by `UPSURE_CORS_ORIGINS` (comma-separated) and
  `UPSURE_CORS_ALLOW_CREDENTIALS`. The `*` default is preserved for dev
  but credentials are off.
- Container runs as UID 10001 with `readOnlyRootFilesystem: true`.
- Error messages strip absolute paths and home directories before
  reaching the wire (`UPSURE_LOG_INCLUDE_PATHS=true` to opt out for debug).

### Migration

UI clients need a one-line change to unwrap the envelope:

```diff
- const data = await res.json();
- if (data.damage_detected) { ... }
+ const { success, data, error } = await res.json();
+ if (!success) { handleError(error); return; }
+ if (data.damage_detected) { ... }
```

All previous top-level fields are unchanged — they're now nested under
`data`. Error payloads carry `error.code` (stable, machine-readable),
`error.message` (human-readable), `error.retryable` (boolean), and
optional `error.details`.

Probes that previously checked `/health` should switch to:
- `/livez` for Kubernetes liveness (always 200 if the process is alive),
- `/readyz` for readiness gating (200 only when critical models loaded).

`/health` still works and is unchanged in path; the body shape now matches
the envelope.
