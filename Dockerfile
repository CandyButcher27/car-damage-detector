# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# UpSure data-ingestion API — multi-stage container build.
#
# Stage 1 ("builder") installs pip dependencies into a wheelhouse. Stage 2
# ("runtime") copies the prepared site-packages into a slim image, drops to
# a non-root user, and exposes uvicorn on port 8000.
#
# Build:   docker build -t upsure/data-ingestion:latest .
# Run:     docker run --rm -p 8000:8000 --env-file .env upsure/data-ingestion:latest
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.10

# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time deps for Pillow / numpy / opencv / onnxruntime wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./

# --ignore-installed so packages already present in the base image (notably
# packaging, which TensorFlow and PaddleX both import at runtime) are still
# materialised under /install. Without it pip silently skips them and the
# runtime stage crashes on `from packaging import version`.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --prefix=/install --ignore-installed -r requirements.txt \
 && python -m pip install --prefix=/install --ignore-installed \
        "prometheus_client>=0.17,<1" \
        "gunicorn>=21.2,<24"

# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_USER=upsure
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    UPSURE_LOG_JSON=1 \
    UPSURE_LOG_LEVEL=INFO \
    UPSURE_METRICS_ENABLED=1 \
    UPSURE_PRELOAD_MODELS=0 \
    PORT=8000

# Runtime libraries needed by Pillow, opencv-headless, onnxruntime, tf.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid ${APP_GID} ${APP_USER} \
    && useradd --system --uid ${APP_UID} --gid ${APP_GID} \
        --home-dir /home/${APP_USER} --create-home --shell /usr/sbin/nologin ${APP_USER}

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=${APP_UID}:${APP_GID} . /app
# Pre-create writable locations for the runtime user:
#   /app/models  — bind mount target for model artefacts
#   /tmp/upsure  — per-request temp dir prefix
#   /home/upsure/.paddleocr, /home/upsure/.paddlex
#                — PaddleOCR/PaddleX cache (auto-downloaded weights). The
#                  runtime user has no shell so HOME must already exist.
RUN mkdir -p /app/models /tmp/upsure /home/${APP_USER}/.rapidocr /app/.cache \
 && chown -R ${APP_UID}:${APP_GID} /app /tmp/upsure /home/${APP_USER}

ENV HOME=/home/${APP_USER}

# Pre-warm RapidOCR model weights (~20 MB) so the first OCR request doesn't
# pay download cost. Skip with --build-arg PREWARM_OCR=0 to keep the image lean.
ARG PREWARM_OCR=1
RUN if [ "${PREWARM_OCR}" = "1" ]; then \
        echo "Pre-warming RapidOCR weights..." && \
        chown -R ${APP_UID}:${APP_GID} /home/${APP_USER} && \
        su -s /bin/sh -c "HOME=/home/${APP_USER} python -c '\
from rapidocr import RapidOCR, LangRec; \
e=RapidOCR(); print(\"RapidOCR EN ready\"); \
e2=RapidOCR(params={\"Rec.lang_type\": LangRec.ARABIC}); print(\"RapidOCR AR ready\")'" ${APP_USER} || \
            echo "WARNING: RapidOCR pre-warm failed; first OCR request will be slow." ; \
    fi

USER ${APP_UID}:${APP_GID}

EXPOSE 8000

# tini reaps zombie processes (OCR subprocess) and forwards signals.
ENTRYPOINT ["/usr/bin/tini", "--"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --silent --fail "http://127.0.0.1:${PORT}/livez" || exit 1

# Gunicorn + uvicorn worker. One worker per CPU is the typical recommendation,
# but ML models keep a lot of state per process so we pin to one worker and
# scale horizontally via k8s HPA instead.
CMD ["sh", "-c", "exec gunicorn poc_api:app \
        --bind 0.0.0.0:${PORT} \
        --workers ${UPSURE_WORKERS:-1} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout ${UPSURE_WORKER_TIMEOUT:-180} \
        --graceful-timeout ${UPSURE_GRACEFUL_TIMEOUT:-30} \
        --keep-alive ${UPSURE_KEEPALIVE:-5} \
        --access-logfile - \
        --log-level ${UPSURE_LOG_LEVEL:-info}"]
