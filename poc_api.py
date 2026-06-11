"""UpSure data-ingestion API — production entrypoint.

Public endpoints (unchanged paths, new wrapped envelope):

* ``GET  /``                     — service banner
* ``GET  /health``               — model + dependency status (full detail)
* ``GET  /livez``                — k8s liveness probe (always 200 if alive)
* ``GET  /readyz``               — k8s readiness probe (200 only when models ready)
* ``GET  /metrics``              — Prometheus scrape endpoint
* ``POST /predict/``             — car classification
* ``POST /predict/damage``       — damage detection + parallel ANPR
* ``POST /api/v1/process``       — unified document/image processing

Response envelope (all routes):

    {"success": true|false,
     "data":    <payload> | null,
     "error":   {code, message, retryable, details} | null,
     "meta":    {request_id, endpoint, api_version, service_version,
                 latency_ms, timestamp}}
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

# Defuse the PIL decompression bomb hole. A 25 MB upload can decode into
# multi-GB of pixels for sparse images; cap at ~64 megapixels (≈8K × 8K)
# which still covers any realistic photo we'll receive. Beyond that PIL
# raises Image.DecompressionBombError, which the endpoint catches as a
# UnsupportedMediaError → 415.
Image.MAX_IMAGE_PIXELS = 64_000_000

from app.errors import (
    ApiError,
    DependencyTimeoutError,
    ErrorCode,
    ModelUnavailableError,
    PipelineFailureError,
    UnsupportedMediaError,
    ValidationError,
    to_api_error,
)
from app.health import ComponentStatus, build_router as build_health_router, registry as health_registry
from app.logging_setup import configure_logging, get_logger
from app.observability import (
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    attach_metrics_endpoint,
    init_metrics,
    record_pipeline_latency,
    set_circuit_state,
    set_model_readiness,
)
from app.resilience import Bulkhead, CircuitBreaker, retry, run_with_timeout, safe_call
from app.responses import envelope_success, json_error, json_success
from app.settings import SETTINGS, repo_root

from card_inference import CardNonCardModel, _resolve_model_path
from onnx_inference import BinaryOnnxImageClassifier
from rag_json_chunker import chunk_json_file

# ── Bootstrap logging early so import-time failures are visible ─────────────
configure_logging()
log = get_logger("upsure.api")

try:
    from plate_pipeline import (
        get_model as _get_anpr_model,
        get_reader as _get_anpr_reader,
        run_pipeline as _run_anpr_pipeline,
    )
    _ANPR_AVAILABLE = True
except Exception as _anpr_import_exc:
    _ANPR_AVAILABLE = False
    log.warning(
        "plate_pipeline import failed; ANPR disabled",
        extra={"event": "anpr.import_failed", "exception": repr(_anpr_import_exc)},
    )

    def _get_anpr_model():  # type: ignore[misc]
        raise RuntimeError("plate_pipeline not importable")

    def _get_anpr_reader():  # type: ignore[misc]
        raise RuntimeError("plate_pipeline not importable")

    def _run_anpr_pipeline(img_bytes: bytes, **kwargs) -> dict:  # type: ignore[misc]
        raise RuntimeError("plate_pipeline not importable")

try:
    from keras.models import load_model as load_keras_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model as load_keras_model  # type: ignore
    except ImportError:
        load_keras_model = None

# Patch Keras layers to accept legacy quantization_config kwarg from older saves.
if load_keras_model is not None:
    try:
        import keras
        for layer_cls in [keras.layers.Dense, keras.layers.Conv2D]:
            if hasattr(layer_cls, "__init__"):
                _orig_init = layer_cls.__init__

                def _make_patched_init(orig):
                    def patched_init(self, *args, **kwargs):
                        kwargs.pop("quantization_config", None)
                        orig(self, *args, **kwargs)
                    return patched_init

                layer_cls.__init__ = _make_patched_init(_orig_init)
    except Exception as _patch_exc:  # pragma: no cover
        log.warning(
            "keras monkeypatch failed",
            extra={"event": "keras.patch_failed", "exception": repr(_patch_exc)},
        )


# ── Paths / constants ──────────────────────────────────────────────────────
POC_DIR = repo_root()
OCR_SCRIPT = POC_DIR / "ocr_simple_test.py"

try:
    MODEL_PATH = _resolve_model_path(POC_DIR, None)
except FileNotFoundError as _mp_exc:
    MODEL_PATH = POC_DIR / "models" / "card_noncard_classifier_model.keras"
    log.warning(
        "card classifier model not found",
        extra={"event": "model.missing", "path": str(MODEL_PATH), "exception": repr(_mp_exc)},
    )

# Car classifier resolution: prefer ONNX (faster, no Keras dep), fall back
# to the legacy .keras model. UPSURE_CAR_MODEL env var overrides everything.
_CAR_MODEL_CANDIDATES = [
    "models/best_car_model_v2.onnx",
    "models/digiLifeDoc_best_car_model_v2.onnx",
    "models/best_car_model_v2.keras",
]


def _resolve_car_model_path() -> Path:
    override = os.getenv("UPSURE_CAR_MODEL")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = POC_DIR / path
        return path
    for candidate in _CAR_MODEL_CANDIDATES:
        path = POC_DIR / candidate
        if path.exists():
            return path
    # None present — return the canonical .keras so error messages match
    # what the legacy code expected.
    return POC_DIR / _CAR_MODEL_CANDIDATES[-1]


CAR_MODEL_PATH = _resolve_car_model_path()
DAMAGE_THRESHOLD = 0.25
DAMAGE_IMG_SIZE = 260
DAMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DAMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OCR_PYTHON_DEFAULT = Path("D:/UpSure/OCR_test/venv/Scripts/python.exe")
if not OCR_PYTHON_DEFAULT.exists():
    OCR_PYTHON_DEFAULT = POC_DIR.parent / "OCR_test" / "venv" / "Scripts" / "python.exe"

OCR_PYTHON = Path(os.getenv("UPSURE_OCR_PYTHON", str(OCR_PYTHON_DEFAULT)))

# Default raised from 0.35 → 0.65 after a false-positive at conf 0.636 on
# Non_card_image_1.jpeg. Tunable per-deployment without redeploy via
# UPSURE_CAR_THRESHOLD.
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


CAR_THRESHOLD = _env_float("UPSURE_CAR_THRESHOLD", 0.65)
CAR_FALLBACK_SIZE = 128
PROCESS_TYPES = ("car", "mulkiya", "pdf", "file")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_IMAGE_SUFFIX = ".jpg"
OCR_IMAGE_SUFFIX = ".png"
PDF_SUFFIX = ".pdf"


@dataclass(slots=True)
class NormalizedInput:
    path: Path
    data: bytes
    kind: Literal["image", "pdf", "file"]
    mime_type: str
    converted: bool
    details: dict[str, Any]


def _resolve_damage_model_path() -> Path:
    env_path = os.getenv("UPSURE_DAMAGE_MODEL")
    if env_path:
        return Path(env_path)
    candidates = [
        POC_DIR / "models" / "damage_model.onnx",
        POC_DIR / "models" / "digiLifeDoc_damage_model.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DAMAGE_MODEL_PATH = _resolve_damage_model_path()

YOLO_SIZE = 640
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_CLASSES = ["car-part-crack", "deformation", "flat-tire", "glass-crack", "lamp-crack", "scratches"]
SEVERITY_MINOR_MAX = 0.05
SEVERITY_MODERATE_MAX = 0.15
_CAR_BBOX = [0.5, 0.5, 1.0, 1.0]


def _resolve_yolo_model_path() -> Path:
    env_path = os.getenv("UPSURE_YOLO_MODEL")
    if env_path:
        return Path(env_path)
    candidates = [
        POC_DIR / "models" / "damage_detector_v2.onnx",
        POC_DIR / "models" / "damage_detector.onnx",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


YOLO_MODEL_PATH = _resolve_yolo_model_path()
ANPR_MODEL_PATH = POC_DIR / "models" / "anpr_plate_detector"
ANPR_VIEW_PRIORITY = ["front", "back", "left", "right"]


# Rule tables (unchanged business logic)
_PARTS_RULES: dict[tuple[str, str], list[str]] = {
    ("car-part-crack", "top_left"):     ["hood", "left_fender", "windshield_frame"],
    ("car-part-crack", "top_right"):    ["hood", "right_fender", "windshield_frame"],
    ("car-part-crack", "bottom_left"):  ["front_bumper", "radiator_grille", "left_rocker_panel"],
    ("car-part-crack", "bottom_right"): ["front_bumper", "radiator_grille", "right_rocker_panel"],
    ("car-part-crack", "center"):       ["door_panel", "body_frame", "sill"],
    ("deformation", "top_left"):        ["hood", "left_fender", "left_a_pillar"],
    ("deformation", "top_right"):       ["hood", "right_fender", "right_a_pillar"],
    ("deformation", "bottom_left"):     ["front_bumper", "radiator_support", "left_frame_rail"],
    ("deformation", "bottom_right"):    ["front_bumper", "radiator_support", "right_frame_rail"],
    ("deformation", "center"):          ["door_panel", "b_pillar", "body_frame"],
    ("flat-tire", "top_left"):          ["left_front_tire", "left_front_rim", "left_front_brake_caliper"],
    ("flat-tire", "top_right"):         ["right_front_tire", "right_front_rim", "right_front_brake_caliper"],
    ("flat-tire", "bottom_left"):       ["left_rear_tire", "left_rear_rim", "left_rear_suspension"],
    ("flat-tire", "bottom_right"):      ["right_rear_tire", "right_rear_rim", "right_rear_suspension"],
    ("flat-tire", "center"):            ["tire", "rim", "suspension"],
    ("glass-crack", "top_left"):        ["windshield", "left_a_pillar", "wiper_linkage"],
    ("glass-crack", "top_right"):       ["windshield", "right_a_pillar", "wiper_linkage"],
    ("glass-crack", "bottom_left"):     ["rear_windshield", "left_c_pillar", "rear_wiper"],
    ("glass-crack", "bottom_right"):    ["rear_windshield", "right_c_pillar", "rear_wiper"],
    ("glass-crack", "center"):          ["side_window", "door_seal", "window_regulator"],
    ("lamp-crack", "top_left"):         ["left_headlight_assembly", "left_indicator", "left_daytime_running_light"],
    ("lamp-crack", "top_right"):        ["right_headlight_assembly", "right_indicator", "right_daytime_running_light"],
    ("lamp-crack", "bottom_left"):      ["left_tail_light", "left_reverse_light", "left_brake_light"],
    ("lamp-crack", "bottom_right"):     ["right_tail_light", "right_reverse_light", "right_brake_light"],
    ("lamp-crack", "center"):           ["lamp_assembly", "indicator"],
    ("scratches", "top_left"):          ["hood", "left_fender"],
    ("scratches", "top_right"):         ["hood", "right_fender"],
    ("scratches", "bottom_left"):       ["front_bumper", "left_rocker_panel"],
    ("scratches", "bottom_right"):      ["rear_bumper", "right_rocker_panel"],
    ("scratches", "center"):            ["door_panel"],
}

_REPAIR_RULES: dict[tuple[str, str], dict] = {
    ("car-part-crack", "minor"):    {"action": "Repair — filler + repaint",                      "replace": False},
    ("car-part-crack", "moderate"): {"action": "Replace cracked part",                           "replace": True},
    ("car-part-crack", "severe"):   {"action": "Replace part + inspect structural frame",        "replace": True},
    ("deformation",    "minor"):    {"action": "Repair — paintless dent repair (PDR)",           "replace": False},
    ("deformation",    "moderate"): {"action": "Repair — PDR + repaint panel",                   "replace": False},
    ("deformation",    "severe"):   {"action": "Replace panel + inspect frame rails",            "replace": True},
    ("flat-tire",      "minor"):    {"action": "Repair — patch tire",                            "replace": False},
    ("flat-tire",      "moderate"): {"action": "Replace tire",                                   "replace": True},
    ("flat-tire",      "severe"):   {"action": "Replace tire + inspect rim and suspension",      "replace": True},
    ("glass-crack",    "minor"):    {"action": "Repair — resin injection (if single crack)",     "replace": False},
    ("glass-crack",    "moderate"): {"action": "Replace glass panel",                            "replace": True},
    ("glass-crack",    "severe"):   {"action": "Replace glass + inspect frame seals",            "replace": True},
    ("lamp-crack",     "minor"):    {"action": "Replace lamp lens",                              "replace": True},
    ("lamp-crack",     "moderate"): {"action": "Replace full lamp assembly",                     "replace": True},
    ("lamp-crack",     "severe"):   {"action": "Replace lamp assembly + inspect mount",          "replace": True},
    ("scratches",      "minor"):    {"action": "Repair — machine polish + touch-up paint",       "replace": False},
    ("scratches",      "moderate"): {"action": "Repair — repaint panel",                         "replace": False},
    ("scratches",      "severe"):   {"action": "Repair — filler + full panel repaint",           "replace": False},
}


# ── Circuit breakers (one per downstream) ──────────────────────────────────
OCR_CB = CircuitBreaker(
    name="ocr_subprocess",
    failure_threshold=SETTINGS.cb_failure_threshold,
    recovery_seconds=SETTINGS.cb_recovery_seconds,
    half_open_max_calls=SETTINGS.cb_half_open_max_calls,
    ignored_exceptions=(ValidationError, UnsupportedMediaError),
)
ANPR_CB = CircuitBreaker(
    name="anpr_pipeline",
    failure_threshold=SETTINGS.cb_failure_threshold,
    recovery_seconds=SETTINGS.cb_recovery_seconds,
    half_open_max_calls=SETTINGS.cb_half_open_max_calls,
)
YOLO_CB = CircuitBreaker(
    name="yolo_damage",
    failure_threshold=SETTINGS.cb_failure_threshold,
    recovery_seconds=SETTINGS.cb_recovery_seconds,
    half_open_max_calls=SETTINGS.cb_half_open_max_calls,
)
DAMAGE_CB = CircuitBreaker(
    name="damage_binary",
    failure_threshold=SETTINGS.cb_failure_threshold,
    recovery_seconds=SETTINGS.cb_recovery_seconds,
    half_open_max_calls=SETTINGS.cb_half_open_max_calls,
)

_CIRCUITS: tuple[CircuitBreaker, ...] = (OCR_CB, ANPR_CB, YOLO_CB, DAMAGE_CB)


# ── Bulkheads ──────────────────────────────────────────────────────────────
_damage_bulkhead = Bulkhead("damage", SETTINGS.damage_concurrency)
_ocr_bulkhead = Bulkhead("ocr", SETTINGS.ocr_concurrency)


# ── Lazy model singletons ──────────────────────────────────────────────────
_card_model: CardNonCardModel | None = None
_car_model: Any | None = None
_car_img_size = CAR_FALLBACK_SIZE
_damage_session: ort.InferenceSession | None = None
_yolo_session: ort.InferenceSession | None = None


@retry(attempts=3, base_delay=0.5, max_delay=4.0)
def _load_card_model() -> CardNonCardModel:
    return CardNonCardModel.load(MODEL_PATH)


@retry(attempts=3, base_delay=0.5, max_delay=4.0)
def _load_car_model() -> tuple[Any, int]:
    """Load the car classifier. Returns (model, input_size).

    Prefers ONNX (no Keras dependency) and falls back to Keras only when an
    ONNX artefact isn't present. Both backends are wrapped by
    ``_classify_car_image_from_image`` so callers don't need to know which
    is in use.
    """
    if not CAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Car model file not found at {CAR_MODEL_PATH}")

    if CAR_MODEL_PATH.suffix.lower() == ".onnx":
        model = BinaryOnnxImageClassifier(
            CAR_MODEL_PATH,
            positive_label="car",
            negative_label="non_car",
            # The digiLifeDoc_best_car_model_v2 ONNX emits high probabilities
            # for non-car (verified against intern branch). For other ONNX
            # exports set UPSURE_CAR_MODEL_POSITIVE_HIGH=true.
            positive_when_output_high=_env_bool("UPSURE_CAR_MODEL_POSITIVE_HIGH", False),
        )
        return model, model.input_size

    if load_keras_model is None:
        raise RuntimeError("TensorFlow or Keras is required for the Keras car classifier.")
    model = load_keras_model(str(CAR_MODEL_PATH))
    return model, _get_car_img_size(model)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


@retry(attempts=3, base_delay=0.5, max_delay=4.0)
def _load_damage_session() -> ort.InferenceSession:
    if not DAMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Damage model not found at {DAMAGE_MODEL_PATH}")
    try:
        return ort.InferenceSession(str(DAMAGE_MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as exc:
        message = str(exc)
        if ".onnx.data" in message:
            raise RuntimeError(
                "Damage model is incomplete. "
                f"ONNX Runtime expects a companion external-data file next to "
                f"{DAMAGE_MODEL_PATH.name}: {DAMAGE_MODEL_PATH.name}.data"
            ) from exc
        raise RuntimeError(
            f"Failed to initialize damage model at {DAMAGE_MODEL_PATH}: {message}"
        ) from exc


@retry(attempts=3, base_delay=0.5, max_delay=4.0)
def _load_yolo_session() -> ort.InferenceSession:
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
    return ort.InferenceSession(str(YOLO_MODEL_PATH), providers=["CPUExecutionProvider"])


def _get_card_model() -> CardNonCardModel:
    global _card_model
    if _card_model is None:
        _card_model = _load_card_model()
        set_model_readiness("card_noncard", True)
    return _card_model


def _get_car_img_size(loaded_model: Any) -> int:
    try:
        shape = loaded_model.input_shape
        for dimension in shape[1:]:
            if dimension and dimension > 3:
                return int(dimension)
    except Exception:
        pass
    return CAR_FALLBACK_SIZE


def _get_car_model() -> Any:
    global _car_model, _car_img_size
    if _car_model is None:
        _car_model, _car_img_size = _load_car_model()
        set_model_readiness("car_classifier", True)
    return _car_model


def _get_damage_session() -> ort.InferenceSession:
    global _damage_session
    if _damage_session is None:
        _damage_session = _load_damage_session()
        set_model_readiness("damage_binary", True)
    return _damage_session


def _get_yolo_session() -> ort.InferenceSession:
    global _yolo_session
    if _yolo_session is None:
        _yolo_session = _load_yolo_session()
        set_model_readiness("yolo_damage", True)
    return _yolo_session


# ── Preprocessing / inference helpers ──────────────────────────────────────
def _preprocess_for_damage(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((DAMAGE_IMG_SIZE, DAMAGE_IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - DAMAGE_MEAN) / DAMAGE_STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax that's numerically stable for batched inputs."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _damage_result_from_probs(probs: np.ndarray) -> dict[str, Any]:
    """Convert a single (2,) probability vector into the response dict."""
    label = 1 if probs[1] >= DAMAGE_THRESHOLD else 0
    return {
        "damage_detected":  bool(label == 1),
        "confidence_score": float(round(float(probs[label]), 4)),
        "prob_damaged":     float(round(float(probs[1]), 4)),
        "prob_clean":       float(round(float(probs[0]), 4)),
    }


def _run_damage_inference(arr: np.ndarray) -> dict[str, Any]:
    sess = _get_damage_session()
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: arr})[0][0]
    probs = _softmax_rows(logits.reshape(1, -1))[0]
    return _damage_result_from_probs(probs)


def _run_damage_inference_batch(batch: np.ndarray) -> list[dict[str, Any]]:
    """Run damage inference on N preprocessed views in a single session call.

    ``batch`` has shape (N, 3, H, W). Returns a list of per-view result dicts
    in the same order. Use this when you have multiple views in the same
    request — it amortises ORT kernel launches and NumPy allocations.
    """
    if batch.shape[0] == 0:
        return []
    sess = _get_damage_session()
    input_name = sess.get_inputs()[0].name
    logits = np.asarray(sess.run(None, {input_name: batch})[0])
    if logits.ndim != 2:
        logits = logits.reshape(batch.shape[0], -1)
    probs = _softmax_rows(logits)
    return [_damage_result_from_probs(probs[i]) for i in range(batch.shape[0])]


def _preprocess_yolo(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((YOLO_SIZE, YOLO_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]


def _iou_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1 = box_a[0] - box_a[2] / 2; ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2; ay2 = box_a[1] + box_a[3] / 2
    bx1 = box_b[0] - box_b[2] / 2; by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2; by2 = box_b[1] + box_b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    order = scores.argsort()[::-1].tolist()
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou_boxes(boxes[i], boxes[j]) < YOLO_IOU]
    return keep


def _get_severity(damage_w: float, damage_h: float) -> str:
    ratio = damage_w * damage_h
    if ratio < SEVERITY_MINOR_MAX:
        return "minor"
    if ratio < SEVERITY_MODERATE_MAX:
        return "moderate"
    return "severe"


def _get_region(bbox: list[float]) -> str:
    dx = bbox[0] - _CAR_BBOX[0]
    dy = bbox[1] - _CAR_BBOX[1]
    rel_x = dx / (_CAR_BBOX[2] / 2)
    rel_y = dy / (_CAR_BBOX[3] / 2)
    if abs(rel_x) < 0.3 and abs(rel_y) < 0.3:
        return "center"
    if rel_y <= 0:
        return "top_left" if rel_x <= 0 else "top_right"
    return "bottom_left" if rel_x <= 0 else "bottom_right"


def _run_yolo_pipeline(img_bytes: bytes) -> list[dict[str, Any]]:
    arr = _preprocess_yolo(img_bytes)
    sess = _get_yolo_session()
    inp = sess.get_inputs()[0].name
    output = sess.run(None, {inp: arr})[0]
    preds = output[0].T

    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    max_scores = np.max(class_scores, axis=1)

    mask = max_scores >= YOLO_CONF
    boxes, class_ids, max_scores = boxes[mask], class_ids[mask], max_scores[mask]

    if len(boxes) == 0:
        return []

    keep = _nms(boxes, max_scores)
    results: list[dict[str, Any]] = []
    for i in keep:
        bbox_px = boxes[i].tolist()
        bbox = [float(round(v / YOLO_SIZE, 4)) for v in bbox_px]
        cls_name = YOLO_CLASSES[int(class_ids[i])]
        severity = _get_severity(bbox[2], bbox[3])
        region = _get_region(bbox)
        parts = _PARTS_RULES.get((cls_name, region), [])
        repair = _REPAIR_RULES.get(
            (cls_name, severity),
            {"action": "Manual inspection required", "replace": False},
        )
        results.append({
            "type":          cls_name,
            "confidence":    float(round(float(max_scores[i]), 4)),
            "severity":      severity,
            "bbox":          bbox,
            "parts_at_risk": parts,
            "repair_action": repair["action"],
            "replace":       repair["replace"],
        })
    return results


def _classify_car_image(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        return _classify_car_image_from_image(image)


def _classify_car_image_from_image(image: Image.Image) -> dict[str, Any]:
    """Run the car classifier and normalise output regardless of backend."""
    model = _get_car_model()

    # ONNX path — uses the generic BinaryOnnxImageClassifier from
    # ``onnx_inference``. Probabilities are already softmax/sigmoid-normalised.
    if isinstance(model, BinaryOnnxImageClassifier):
        result = model.classify(image, threshold=CAR_THRESHOLD)
        car_prob = result["car_probability"]
        return {
            "is_car": result["label"] == "car",
            "confidence": round(car_prob if result["label"] == "car" else 1.0 - car_prob, 4),
            "raw_score": round(result["model_output"], 4),
            "threshold_used": CAR_THRESHOLD,
            "backend": "onnx",
        }

    # Legacy Keras path — preserved for backwards compatibility with sites
    # that haven't migrated their model artefacts yet.
    prepared_image = ImageOps.exif_transpose(image).convert("RGB")
    prepared_image = prepared_image.resize((_car_img_size, _car_img_size), Image.Resampling.LANCZOS)
    array = np.asarray(prepared_image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    prediction = model.predict(array, verbose=0)

    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        is_car = score >= CAR_THRESHOLD
        confidence = score if is_car else (1.0 - score)
    else:
        index = int(np.argmax(prediction[0]))
        score = float(prediction[0][index])
        is_car = index == 0
        confidence = score

    return {
        "is_car": is_car,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 4),
        "threshold_used": CAR_THRESHOLD,
        "backend": "keras",
    }


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


# ── Upload helpers ─────────────────────────────────────────────────────────
def _is_pdf(upload: UploadFile) -> bool:
    filename = Path(upload.filename or "").name.lower()
    content_type = (upload.content_type or "").lower()
    return filename.endswith(".pdf") or content_type == "application/pdf"


def _is_image(upload: UploadFile) -> bool:
    filename = Path(upload.filename or "").name.lower()
    content_type = (upload.content_type or "").lower()
    return content_type.startswith("image/") or any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS)


def _guess_mime_type(upload: UploadFile, source_name: str) -> str:
    content_type = (upload.content_type or "").strip().lower()
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(source_name)
    return guessed or "application/octet-stream"


def _safe_stem(source_name: str) -> str:
    stem = Path(source_name).stem.strip()
    return stem or "upload"


def _image_bytes_to_format(
    file_bytes: bytes,
    *,
    output_format: Literal["JPEG", "PNG"],
) -> tuple[bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(file_bytes)) as image:
        original_format = image.format
        original_mode = image.mode
        original_size = [image.width, image.height]
        prepared_image = ImageOps.exif_transpose(image)

        if output_format == "JPEG":
            prepared_image = prepared_image.convert("RGB")
            suffix = MODEL_IMAGE_SUFFIX
            mime_type = "image/jpeg"
            save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": 95, "optimize": True}
        else:
            if prepared_image.mode not in {"RGB", "RGBA", "L"}:
                prepared_image = prepared_image.convert("RGB")
            suffix = OCR_IMAGE_SUFFIX
            mime_type = "image/png"
            save_kwargs = {"format": "PNG"}

        buffer = io.BytesIO()
        prepared_image.save(buffer, **save_kwargs)

    return buffer.getvalue(), {
        "source_kind": "image",
        "target_kind": "image",
        "target_suffix": suffix,
        "target_mime_type": mime_type,
        "original_format": original_format,
        "original_mode": original_mode,
        "original_size": original_size,
        "converted": True,
    }


def _pdf_first_page_to_jpeg(file_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise PipelineFailureError(
            "PyMuPDF is required to convert PDFs into images for this process_type.",
        ) from exc

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise UnsupportedMediaError("PDF has no pages to convert.")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        png_bytes = pixmap.tobytes("png")

    jpeg_bytes, image_details = _image_bytes_to_format(png_bytes, output_format="JPEG")
    image_details.update({
        "source_kind": "pdf",
        "target_kind": "image",
        "target_suffix": MODEL_IMAGE_SUFFIX,
        "target_mime_type": "image/jpeg",
        "pdf_page_used": 1,
    })
    return jpeg_bytes, image_details


def _image_bytes_to_pdf(file_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(file_bytes)) as image:
        original_format = image.format
        original_mode = image.mode
        original_size = [image.width, image.height]
        prepared_image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        prepared_image.save(buffer, format="PDF", resolution=200.0)

    return buffer.getvalue(), {
        "source_kind": "image",
        "target_kind": "pdf",
        "target_suffix": PDF_SUFFIX,
        "target_mime_type": "application/pdf",
        "original_format": original_format,
        "original_mode": original_mode,
        "original_size": original_size,
        "converted": True,
    }


def _write_normalized(temp_dir: Path, source_name: str, suffix: str, data: bytes) -> Path:
    path = temp_dir / f"{_safe_stem(source_name)}_normalized{suffix}"
    path.write_bytes(data)
    return path


def _normalize_for_image_model(
    *,
    file_bytes: bytes,
    source_name: str,
    temp_dir: Path,
    is_pdf_file: bool,
) -> NormalizedInput:
    if is_pdf_file:
        data, details = _pdf_first_page_to_jpeg(file_bytes)
    else:
        data, details = _image_bytes_to_format(file_bytes, output_format="JPEG")
    path = _write_normalized(temp_dir, source_name, MODEL_IMAGE_SUFFIX, data)
    return NormalizedInput(
        path=path, data=data, kind="image", mime_type="image/jpeg",
        converted=True, details=details,
    )


def _normalize_for_ocr(
    *,
    file_bytes: bytes,
    source_name: str,
    temp_dir: Path,
    is_pdf_file: bool,
    target_pdf: bool,
) -> NormalizedInput:
    if is_pdf_file:
        converted = Path(source_name).suffix.lower() != PDF_SUFFIX
        path = _write_normalized(temp_dir, source_name, PDF_SUFFIX, file_bytes)
        return NormalizedInput(
            path=path, data=file_bytes, kind="pdf", mime_type="application/pdf",
            converted=converted,
            details={
                "source_kind": "pdf", "target_kind": "pdf",
                "target_suffix": PDF_SUFFIX, "target_mime_type": "application/pdf",
                "converted": converted,
            },
        )

    if target_pdf:
        data, details = _image_bytes_to_pdf(file_bytes)
        path = _write_normalized(temp_dir, source_name, PDF_SUFFIX, data)
        return NormalizedInput(
            path=path, data=data, kind="pdf", mime_type="application/pdf",
            converted=True, details=details,
        )

    data, details = _image_bytes_to_format(file_bytes, output_format="PNG")
    path = _write_normalized(temp_dir, source_name, OCR_IMAGE_SUFFIX, data)
    return NormalizedInput(
        path=path, data=data, kind="image", mime_type="image/png",
        converted=True, details=details,
    )


def _mean_confidence(payload: Any) -> float:
    confidences: list[float] = []
    if isinstance(payload, dict):
        pages = payload.get("pages")
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                lines = page.get("lines")
                if not isinstance(lines, list):
                    continue
                for line in lines:
                    if isinstance(line, dict):
                        value = line.get("confidence")
                        if isinstance(value, (int, float)):
                            confidences.append(float(value))
        lines = payload.get("lines")
        if isinstance(lines, list):
            for line in lines:
                if isinstance(line, dict):
                    value = line.get("confidence")
                    if isinstance(value, (int, float)):
                        confidences.append(float(value))
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _flatten_ocr_lines(payload: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return lines

    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_number = page.get("page")
            page_lines = page.get("lines")
            if not isinstance(page_lines, list):
                continue
            for line_index, line in enumerate(page_lines, start=1):
                if not isinstance(line, dict):
                    continue
                lines.append({
                    "page": page_number,
                    "line_index": line_index,
                    "text": line.get("text"),
                    "confidence": line.get("confidence"),
                })
        return lines

    image_lines = payload.get("lines")
    if isinstance(image_lines, list):
        for line_index, line in enumerate(image_lines, start=1):
            if not isinstance(line, dict):
                continue
            lines.append({
                "page": payload.get("page", 1),
                "line_index": line_index,
                "text": line.get("text"),
                "confidence": line.get("confidence"),
            })
    return lines


# Mulkiya translation dictionaries (unchanged)
MULKIYA_FIELD_LABELS_EN = {
    "plate_number": "Plate number",
    "plate_text": "Plate letters",
    "vehicle_type": "Vehicle type",
    "make": "Make",
    "model": "Model",
    "color": "Color",
    "year": "Model year",
    "vin_or_chassis": "VIN or chassis number",
    "engine_cc": "Engine capacity (cc)",
    "empty_weight_kg": "Empty weight (kg)",
    "max_load_kg": "Maximum load (kg)",
    "seats": "Seats",
    "issue_date": "Issue date",
    "expiry_date": "Expiry date",
    "owner_name": "Owner name",
    "notes": "Notes",
}

ARABIC_VALUE_TRANSLATIONS = {
    "خصوصي": "Private",
    "بب": "Ba Ba (Arabic plate letters)",
    "تويوتا": "Toyota",
    "كورولا": "Corolla",
    "صالون": "Sedan",
    "تويوتا صالون كورولا": "Toyota Corolla sedan",
    "ابيض": "White",
    "أبيض": "White",
    "الولايات": "United States",
    "المتحدة الامريكية": "United States of America",
    "الولايات المتحدة الامريكية": "United States of America",
    "عمان": "Oman",
    "سلطنة": "Sultanate",
    "سلطنة عمان": "Sultanate of Oman",
    "شرطة": "Police",
    "شرطة عمان": "Oman Police",
    "السلطائية": "Royal",
    "الادارة": "Directorate",
    "العامة": "General",
    "للمرور": "Traffic",
    "رخصة": "License",
    "مركبة": "Vehicle",
    "رخصة مركبة": "Vehicle license",
    "رقم": "Number",
    "اللوحة": "Plate",
    "الوحة": "Plate",
    "نوع": "Type",
    "نوع الوحة": "Plate type",
    "نوع المركبة": "Vehicle type",
    "اللون": "Color",
    "المنشاء": "Origin",
    "سنة الطرار": "Model year",
    "سنة الصلع": "Manufacture year",
    "سعة المحرك": "Engine capacity",
    "الوزن": "Weight",
    "فارغ": "Empty",
    "الحمولة": "Load",
    "القصوى": "Maximum",
    "كجم": "kg",
    "عدد الركاب": "Number of passengers",
    "عدد المحاور": "Number of axles",
    "رقم الاعدةالشاص": "Chassis number",
    "رقم المحرة": "Engine number",
    "الرخصة من": "License from",
    "صلاحية": "Validity",
}


def _translate_text_local(text: Any) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str):
        return str(text)
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if cleaned in ARABIC_VALUE_TRANSLATIONS:
        return ARABIC_VALUE_TRANSLATIONS[cleaned]
    translated = cleaned
    for source, target in sorted(ARABIC_VALUE_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(source, target)
    return translated


def _translate_extracted_data_local(extracted_data: Any) -> Any:
    if not isinstance(extracted_data, dict):
        return extracted_data
    translated: dict[str, Any] = {}
    for key, value in extracted_data.items():
        if key == "source":
            continue
        label = MULKIYA_FIELD_LABELS_EN.get(key, key.replace("_", " ").title())
        translated_value = _translate_text_local(value) if isinstance(value, str) else value
        translated[key] = {"label": label, "value": value, "translation": translated_value}
    return translated


def _translate_ocr_payload_local(raw_ocr: Any) -> Any:
    lines = _flatten_ocr_lines(raw_ocr)
    return {
        "line_count": len(lines),
        "lines": [
            {
                "page": line.get("page"),
                "line_index": line.get("line_index"),
                "text": line.get("text"),
                "translation": _translate_text_local(line.get("text")),
                "confidence": line.get("confidence"),
            }
            for line in lines
        ],
    }


def _build_translation_payload(extracted_data: Any, raw_ocr: Any) -> dict[str, Any]:
    return {
        "target_language": "en",
        "provider": "local_dictionary",
        "note": (
            "Local helper translation for Mulkiya review. It translates known OCR "
            "labels and common values, and leaves unknown text unchanged."
        ),
        "extracted_data": _translate_extracted_data_local(extracted_data),
        "raw_ocr": _translate_ocr_payload_local(raw_ocr) if raw_ocr is not None else None,
    }


# ── OCR subprocess with timeout + circuit breaker ──────────────────────────
def _ocr_subprocess(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(POC_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=SETTINGS.ocr_subprocess_timeout_seconds,
    )


def _run_ocr_script(
    input_path: Path,
    *,
    lang: str,
    extract_mulkya: bool,
    is_pdf: bool,
    prefer_pdf_text: bool,
) -> subprocess.CompletedProcess[str]:
    python_executable = OCR_PYTHON if OCR_PYTHON.exists() else Path(sys.executable)
    args = [
        str(python_executable),
        str(OCR_SCRIPT),
        str(input_path),
        "--write_text",
        "--no_images",
        "--lang",
        lang,
    ]
    if extract_mulkya:
        args.append("--extract_mulkya")
    if is_pdf and prefer_pdf_text:
        args.append("--prefer_pdf_text")

    pipeline_start = time.perf_counter()
    try:
        completed = OCR_CB.call(_ocr_subprocess, args)
    except subprocess.TimeoutExpired as exc:
        record_pipeline_latency("ocr", time.perf_counter() - pipeline_start)
        log.error(
            "ocr subprocess timed out",
            extra={
                "event": "ocr.timeout",
                "timeout_seconds": SETTINGS.ocr_subprocess_timeout_seconds,
            },
        )
        raise DependencyTimeoutError(
            f"OCR did not finish within {SETTINGS.ocr_subprocess_timeout_seconds:.0f}s.",
            details={"stage": "ocr"},
        ) from exc
    except FileNotFoundError as exc:
        raise ModelUnavailableError(
            "OCR helper Python is not available on this pod.",
            details={"hint": "Set UPSURE_OCR_PYTHON to a path inside the container."},
        ) from exc

    record_pipeline_latency("ocr", time.perf_counter() - pipeline_start)

    if completed.returncode != 0:
        log.error(
            "ocr subprocess failed",
            extra={
                "event": "ocr.failed",
                "returncode": completed.returncode,
                "stderr_tail": (completed.stderr or "")[-500:],
            },
        )
        raise PipelineFailureError(
            "OCR pipeline failed.",
            details={"returncode": completed.returncode},
        )
    return completed


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_text_preview(path: Path, max_chars: int = 4000) -> tuple[str | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
        return text[:max_chars], "utf-8"
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars], "utf-8-replace"
        except OSError as exc:
            return None, str(exc)
    except OSError as exc:
        return None, str(exc)


def _collect_general_file_details(path: Path, *, source_name: str, mime_type: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    details: dict[str, Any] = {
        "filename": source_name,
        "mime_type": mime_type,
        "extension": suffix or None,
        "size_bytes": path.stat().st_size,
        "category": "binary",
        "suggested_process_type": "file",
    }
    if mime_type == "application/pdf":
        details["category"] = "pdf"
        details["suggested_process_type"] = "pdf"
        return details

    if mime_type.startswith("image/"):
        details["category"] = "image"
        details["suggested_process_type"] = "mulkiya"
        try:
            with Image.open(path) as image:
                details["image"] = {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                }
        except Exception as exc:
            details["image_error"] = str(exc)
        return details

    if mime_type.startswith("text/") or suffix in {".md", ".txt", ".log", ".py", ".js", ".ts", ".css", ".html", ".xml", ".yaml", ".yml", ".ini", ".cfg"}:
        preview, encoding_used = _read_text_preview(path)
        details["category"] = "text"
        details["suggested_process_type"] = "file"
        details["text_preview"] = preview
        details["encoding_used"] = encoding_used
        if preview is not None:
            details["line_count_preview"] = len(preview.splitlines())
        return details

    if suffix == ".json":
        details["category"] = "json"
        try:
            payload = _load_json(path)
            details["top_level_type"] = type(payload).__name__
            if isinstance(payload, dict):
                details["top_level_keys"] = list(payload.keys())[:25]
            elif isinstance(payload, list):
                details["item_count"] = len(payload)
        except Exception as exc:
            details["json_error"] = str(exc)
        return details

    if suffix in {".csv", ".tsv"}:
        details["category"] = "tabular"
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                rows = []
                for _, row in zip(range(5), reader):
                    rows.append(row)
            details["preview_rows"] = rows
        except Exception as exc:
            details["tabular_error"] = str(exc)
        return details

    if suffix in {".zip", ".docx", ".xlsx", ".pptx"} or zipfile.is_zipfile(path):
        details["category"] = "archive"
        try:
            with zipfile.ZipFile(path) as archive:
                details["archive_entries"] = archive.namelist()[:30]
        except Exception as exc:
            details["archive_error"] = str(exc)
        return details

    if suffix == ".xml":
        details["category"] = "xml"
        try:
            root = ElementTree.parse(path).getroot()
            details["root_tag"] = root.tag
        except Exception as exc:
            details["xml_error"] = str(exc)
        return details

    return details


def _relativize_chunks(chunks: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        item["source_file"] = source_name
        output.append(item)
    return output


def _build_pipeline_response(
    *,
    source_name: str,
    input_kind: str,
    classification: dict[str, Any] | None,
    confidence_score: float,
    extracted_data: Any,
    raw_ocr: Any,
    chunk_source_path: Path | None,
    note: str,
    car_classification: dict[str, Any] | None = None,
    normalized_input: NormalizedInput | None = None,
    translation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if chunk_source_path and chunk_source_path.exists():
        chunks = [
            {
                "source_file": chunk.source_file,
                "document_type": chunk.document_type,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunk_json_file(chunk_source_path, max_chars=1200, overlap_lines=3)
        ]
        rag_chunks = _relativize_chunks(chunks, source_name)
        artifact_chunk_source = str(chunk_source_path)
    else:
        rag_chunks = []
        artifact_chunk_source = None

    response = {
        "input": {
            "filename": source_name,
            "kind": input_kind,
            "normalized": (
                {
                    "kind": normalized_input.kind,
                    "mime_type": normalized_input.mime_type,
                    "converted": normalized_input.converted,
                    "details": normalized_input.details,
                }
                if normalized_input
                else None
            ),
        },
        "classification": classification,
        "car_classification": car_classification,
        "confidence_score": confidence_score,
        "extracted_data": extracted_data,
        "raw_ocr": raw_ocr,
        "rag_chunks": rag_chunks,
        "artifacts": {"chunk_source": artifact_chunk_source},
        "note": note,
    }
    if translation is not None:
        response["translation"] = translation
    return response


# ── Lifespan: optional model preload + circuit-state metric refresh ────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_metrics()
    set_model_readiness("damage_binary", False)
    set_model_readiness("yolo_damage", False)
    set_model_readiness("car_classifier", False)
    set_model_readiness("card_noncard", False)
    set_model_readiness("anpr", False)

    log.info(
        "service starting",
        extra={
            "event": "service.start",
            "version": SETTINGS.service_version,
            "env": SETTINGS.environment,
            "preload_models": SETTINGS.preload_models_on_startup,
        },
    )

    if SETTINGS.preload_models_on_startup:
        # Best-effort preload: log failures but don't block startup. /readyz
        # will reflect the real state.
        for label, loader in (
            ("damage_binary", _get_damage_session),
            ("yolo_damage", _get_yolo_session),
            ("card_noncard", _get_card_model),
            ("car_classifier", _get_car_model),
        ):
            try:
                await run_in_threadpool(loader)
                log.info("model preloaded", extra={"event": "model.preload", "model": label})
            except Exception as exc:
                log.warning(
                    "model preload failed",
                    extra={"event": "model.preload_failed", "model": label, "exception": repr(exc)},
                )

    # Background circuit-state metric publisher (tiny, no asyncio.sleep loop —
    # we just publish once; subsequent state changes update via set_circuit_state
    # from CircuitBreaker callers).
    for cb in _CIRCUITS:
        set_circuit_state(cb.name, cb.state)

    try:
        yield
    finally:
        log.info("service stopping", extra={"event": "service.stop"})


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"UpSure {SETTINGS.service_name}",
    version=SETTINGS.service_version,
    description="Production data-ingestion API for vehicle, OCR, damage, and ANPR workflows.",
    lifespan=lifespan,
)

# Order matters: outermost first.
# 1. body-size guard (raw ASGI, runs before everything else)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=SETTINGS.max_upload_bytes)

# 2. CORS — env-driven allowlist. "*" still works for dev but credentials off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=SETTINGS.cors_allow_credentials,
    expose_headers=["X-Request-ID", "X-Process-Time-ms"],
)

# 3. request-id + access log
app.add_middleware(RequestContextMiddleware)

attach_metrics_endpoint(app)


# ── Health-check registrations ─────────────────────────────────────────────
# Health probes use the *unwrapped* loaders (no retry) so each /readyz hit is
# fast. Startup preload still uses the retrying version below, which is the
# right place to absorb transient PVC-mount or network races.

def _check_damage_model() -> ComponentStatus:
    global _damage_session
    if _damage_session is not None:
        return ComponentStatus("damage_binary", True, critical=True, extra={"path": str(DAMAGE_MODEL_PATH)})
    try:
        _damage_session = _load_damage_session.__wrapped__()
        set_model_readiness("damage_binary", True)
        return ComponentStatus("damage_binary", True, critical=True, extra={"path": str(DAMAGE_MODEL_PATH)})
    except Exception as exc:
        set_model_readiness("damage_binary", False)
        return ComponentStatus("damage_binary", False, critical=True, detail=str(exc), extra={"path": str(DAMAGE_MODEL_PATH)})


def _check_yolo_model() -> ComponentStatus:
    global _yolo_session
    if _yolo_session is not None:
        return ComponentStatus("yolo_damage", True, critical=False, extra={"path": str(YOLO_MODEL_PATH)})
    try:
        _yolo_session = _load_yolo_session.__wrapped__()
        set_model_readiness("yolo_damage", True)
        return ComponentStatus("yolo_damage", True, critical=False, extra={"path": str(YOLO_MODEL_PATH)})
    except Exception as exc:
        set_model_readiness("yolo_damage", False)
        return ComponentStatus("yolo_damage", False, critical=False, detail=str(exc), extra={"path": str(YOLO_MODEL_PATH)})


def _check_car_model() -> ComponentStatus:
    global _car_model, _car_img_size
    if _car_model is not None:
        return ComponentStatus("car_classifier", True, critical=False, extra={"path": str(CAR_MODEL_PATH)})
    try:
        _car_model, _car_img_size = _load_car_model.__wrapped__()
        set_model_readiness("car_classifier", True)
        return ComponentStatus("car_classifier", True, critical=False, extra={"path": str(CAR_MODEL_PATH)})
    except Exception as exc:
        set_model_readiness("car_classifier", False)
        return ComponentStatus("car_classifier", False, critical=False, detail=str(exc), extra={"path": str(CAR_MODEL_PATH)})


def _check_anpr_model() -> ComponentStatus:
    if not _ANPR_AVAILABLE:
        return ComponentStatus(
            "anpr", False, critical=False,
            detail="plate_pipeline module not importable",
            extra={"path": str(ANPR_MODEL_PATH)},
        )
    if not ANPR_MODEL_PATH.exists():
        return ComponentStatus(
            "anpr", False, critical=False,
            detail=f"ANPR model directory not found",
            extra={"path": str(ANPR_MODEL_PATH)},
        )
    try:
        _get_anpr_model()
        _get_anpr_reader()
        set_model_readiness("anpr", True)
        return ComponentStatus("anpr", True, critical=False, extra={"path": str(ANPR_MODEL_PATH)})
    except Exception as exc:
        set_model_readiness("anpr", False)
        return ComponentStatus("anpr", False, critical=False, detail=str(exc), extra={"path": str(ANPR_MODEL_PATH)})


def _check_card_model() -> ComponentStatus:
    global _card_model
    if _card_model is not None:
        return ComponentStatus("card_noncard", True, critical=False, extra={"path": str(MODEL_PATH)})
    if not MODEL_PATH.exists():
        return ComponentStatus(
            "card_noncard", False, critical=False,
            detail="card/noncard model file not found",
            extra={"path": str(MODEL_PATH)},
        )
    try:
        _card_model = _load_card_model.__wrapped__()
        set_model_readiness("card_noncard", True)
        return ComponentStatus("card_noncard", True, critical=False, extra={"path": str(MODEL_PATH)})
    except Exception as exc:
        set_model_readiness("card_noncard", False)
        return ComponentStatus("card_noncard", False, critical=False, detail=str(exc), extra={"path": str(MODEL_PATH)})


def _check_circuits() -> ComponentStatus:
    snapshots = [cb.snapshot() for cb in _CIRCUITS]
    # Surface circuit state into Prometheus on every probe — cheap and
    # ensures dashboards stay close to real-time without a background task.
    for cb in _CIRCUITS:
        set_circuit_state(cb.name, cb.state)
    any_open = any(snap["state"] == "open" for snap in snapshots)
    return ComponentStatus(
        "circuits", not any_open, critical=False,
        detail="one or more circuits open" if any_open else None,
        extra={"circuits": snapshots},
    )


health_registry.register("damage_binary", _check_damage_model)
health_registry.register("yolo_damage", _check_yolo_model)
health_registry.register("car_classifier", _check_car_model)
health_registry.register("card_noncard", _check_card_model)
health_registry.register("anpr", _check_anpr_model)
health_registry.register("circuits", _check_circuits)


# Mount health router (provides /livez, /readyz, /health)
app.include_router(build_health_router())


# ── Global error handlers ──────────────────────────────────────────────────
@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    start_perf = getattr(request.state, "start_perf", None)
    log.warning(
        "api error",
        extra={
            "event": "api.error",
            "code": exc.code,
            "retryable": exc.retryable,
            "http_status": exc.http_status,
            "path": request.url.path,
        },
    )
    return json_error(exc, request=request, start_perf=start_perf)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    start_perf = getattr(request.state, "start_perf", None)
    # Map fastapi HTTPException to our envelope without losing the status code.
    code_map = {
        400: ErrorCode.VALIDATION_ERROR,
        404: ErrorCode.NOT_FOUND,
        413: ErrorCode.PAYLOAD_TOO_LARGE,
        415: ErrorCode.UNSUPPORTED_MEDIA,
        503: ErrorCode.MODEL_UNAVAILABLE,
        504: ErrorCode.DEPENDENCY_TIMEOUT,
    }
    error = ApiError(
        code=code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR),
        message=str(exc.detail),
        retryable=exc.status_code in (503, 504),
        http_status=exc.status_code,
    )
    return json_error(error, request=request, start_perf=start_perf)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    start_perf = getattr(request.state, "start_perf", None)
    err = ValidationError(
        "Request validation failed.",
        details={"errors": _sanitize_for_json(exc.errors())},
    )
    return json_error(err, request=request, start_perf=start_perf)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    start_perf = getattr(request.state, "start_perf", None)
    log.exception(
        "unhandled exception",
        extra={"event": "api.unhandled", "path": request.url.path},
    )
    return json_error(to_api_error(exc), request=request, start_perf=start_perf)


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
async def root(request: Request):
    return json_success(
        {
            "service": SETTINGS.service_name,
            "version": SETTINGS.service_version,
            "message": "UpSure data-ingestion API. Send POST requests to /api/v1/process.",
            "endpoints": ["/livez", "/readyz", "/health", "/metrics",
                          "/predict/", "/predict/damage", "/api/v1/process"],
        },
        request=request,
        start_perf=request.state.start_perf,
    )


@app.post("/predict/")
async def predict_car(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise ValidationError("Uploaded file must have a filename.")

    source_name = Path(file.filename).name
    is_pdf_file = _is_pdf(file)

    pipeline_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="upsure-car-") as temp_dir_name:
        try:
            file_bytes = await file.read()
            normalized = _normalize_for_image_model(
                file_bytes=file_bytes,
                source_name=source_name,
                temp_dir=Path(temp_dir_name),
                is_pdf_file=is_pdf_file,
            )
            image = _load_image_bytes(normalized.data)
            classification = await run_with_timeout(
                _classify_car_image_from_image, image,
                timeout_seconds=SETTINGS.request_timeout_seconds,
                label="car_classifier",
            )
        except ApiError:
            raise
        except (FileNotFoundError, RuntimeError) as exc:
            # Model-load failure surfaces here when the loader can't open
            # the .keras artefact. Route to 503 so clients know to retry.
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise UnsupportedMediaError(
                f"Could not convert uploaded file to a model image: {exc}",
            ) from exc

    record_pipeline_latency("predict_car", time.perf_counter() - pipeline_start)

    return json_success(
        {
            "filename": source_name,
            "normalized": {
                "kind": normalized.kind,
                "mime_type": normalized.mime_type,
                "converted": normalized.converted,
                "details": normalized.details,
            },
            **classification,
        },
        request=request,
        start_perf=request.state.start_perf,
    )


@app.post("/api/v1/process")
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    process_type: Literal["car", "mulkiya", "pdf", "file"] = Form(...),
    card_threshold: float = Form(0.5),
    ocr_lang: str = Form("ar"),
    prefer_pdf_text: bool = Form(False),
    skip_ocr: bool = Form(False),
    translate_to_en: bool = Form(False),
):
    if not file.filename:
        raise ValidationError("Uploaded file must have a filename.")
    if process_type not in PROCESS_TYPES:
        raise ValidationError(
            f"process_type must be one of: {', '.join(PROCESS_TYPES)}",
            details={"received": process_type},
        )

    source_name = Path(file.filename).name
    is_pdf_file = _is_pdf(file)
    mime_type = _guess_mime_type(file, source_name)

    pipeline_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="upsure-poc-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / source_name
        file_bytes = await file.read()

        if process_type == "file":
            input_path.write_bytes(file_bytes)
            details = _collect_general_file_details(
                input_path, source_name=source_name, mime_type=mime_type,
            )
            details_path = input_path.with_name(f"{input_path.stem}_file_summary.json")
            _write_json(details_path, details)
            payload = _build_pipeline_response(
                source_name=source_name,
                input_kind="file",
                classification={
                    "label": details.get("category", "binary"),
                    "mime_type": mime_type,
                    "suggested_process_type": details.get("suggested_process_type", "file"),
                },
                confidence_score=1.0,
                extracted_data=details,
                raw_ocr=None,
                chunk_source_path=details_path,
                note="General file inspection executed without OCR.",
                car_classification=None,
                translation=_build_translation_payload(details, None) if translate_to_en else None,
            )
            record_pipeline_latency("process_file", time.perf_counter() - pipeline_start)
            return json_success(payload, request=request, start_perf=request.state.start_perf)

        if process_type == "car":
            try:
                normalized = _normalize_for_image_model(
                    file_bytes=file_bytes, source_name=source_name,
                    temp_dir=temp_dir, is_pdf_file=is_pdf_file,
                )
                image = _load_image_bytes(normalized.data)
                car_classification = await run_with_timeout(
                    _classify_car_image_from_image, image,
                    timeout_seconds=SETTINGS.request_timeout_seconds,
                    label="car_classifier",
                )
            except ApiError:
                raise
            except (FileNotFoundError, RuntimeError) as exc:
                raise ModelUnavailableError(str(exc)) from exc
            except Exception as exc:
                raise UnsupportedMediaError(
                    f"Could not convert uploaded file to a car model image: {exc}",
                ) from exc

            payload = _build_pipeline_response(
                source_name=source_name,
                input_kind="image",
                classification=None,
                confidence_score=car_classification.get("confidence", 0.0),
                extracted_data=None,
                raw_ocr=None,
                chunk_source_path=None,
                note="Car classification model executed.",
                car_classification=car_classification,
                normalized_input=normalized,
            )
            record_pipeline_latency("process_car", time.perf_counter() - pipeline_start)
            return json_success(payload, request=request, start_perf=request.state.start_perf)

        if process_type == "mulkiya":
            if not is_pdf_file:
                try:
                    model_input = _normalize_for_image_model(
                        file_bytes=file_bytes, source_name=source_name,
                        temp_dir=temp_dir, is_pdf_file=False,
                    )
                    image = _load_image_bytes(model_input.data)
                except Exception as exc:
                    raise UnsupportedMediaError(
                        f"Could not convert uploaded file to a Mulkiya model image: {exc}",
                    ) from exc
                probability = _get_card_model().predict_probability(image, normalize=True)
                label = "card" if probability >= card_threshold else "not card"
                classification = {
                    "label": label,
                    "probability": probability,
                    "threshold": card_threshold,
                }
                confidence_score = probability
            else:
                model_input = None
                classification = {
                    "label": "unknown",
                    "reason": "PDF Mulkiya classification requires OCR.",
                    "threshold": card_threshold,
                }
                confidence_score = 0.0

            if skip_ocr:
                payload = _build_pipeline_response(
                    source_name=source_name,
                    input_kind="pdf" if is_pdf_file else "image",
                    classification=classification,
                    confidence_score=confidence_score,
                    extracted_data=None,
                    raw_ocr=None,
                    chunk_source_path=None,
                    note="Mulkiya classification executed without OCR.",
                    car_classification=None,
                    normalized_input=model_input,
                )
                record_pipeline_latency("process_mulkiya_skip_ocr", time.perf_counter() - pipeline_start)
                return json_success(payload, request=request, start_perf=request.state.start_perf)

            try:
                ocr_input = _normalize_for_ocr(
                    file_bytes=file_bytes, source_name=source_name,
                    temp_dir=temp_dir, is_pdf_file=is_pdf_file, target_pdf=False,
                )
            except Exception as exc:
                raise UnsupportedMediaError(
                    f"Could not convert uploaded file to an OCR-supported format: {exc}",
                ) from exc

            async with _ocr_bulkhead:
                _run_ocr_script(
                    ocr_input.path,
                    lang=ocr_lang,
                    extract_mulkya=True,
                    is_pdf=ocr_input.kind == "pdf",
                    prefer_pdf_text=prefer_pdf_text if ocr_input.kind == "pdf" else False,
                )

            ocr_json_path = ocr_input.path.with_name(f"{ocr_input.path.stem}_ocr.json")
            if not ocr_json_path.exists():
                raise PipelineFailureError("OCR JSON output was not created.")

            raw_ocr = _load_json(ocr_json_path)
            extracted_data_path = ocr_input.path.with_name(f"{ocr_input.path.stem}_mulkya.json")

            if extracted_data_path.exists():
                extracted_data = _load_json(extracted_data_path)
                chunk_source_path = extracted_data_path
                note = "Mulkiya pipeline executed with card inference and OCR extraction."
            else:
                extracted_data = {"lines": _flatten_ocr_lines(raw_ocr)}
                chunk_source_path = ocr_json_path
                confidence_score = _mean_confidence(raw_ocr)
                note = "Mulkiya OCR ran but structured Mulkiya JSON was not created; returned OCR lines."

            payload = _build_pipeline_response(
                source_name=source_name,
                input_kind="pdf" if is_pdf_file else "image",
                classification=classification,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                raw_ocr=raw_ocr,
                chunk_source_path=chunk_source_path,
                note=note,
                car_classification=None,
                normalized_input=ocr_input,
                translation=_build_translation_payload(extracted_data, raw_ocr) if translate_to_en else None,
            )
            record_pipeline_latency("process_mulkiya", time.perf_counter() - pipeline_start)
            return json_success(payload, request=request, start_perf=request.state.start_perf)

        if process_type == "pdf":
            try:
                normalized = _normalize_for_ocr(
                    file_bytes=file_bytes, source_name=source_name,
                    temp_dir=temp_dir, is_pdf_file=is_pdf_file, target_pdf=True,
                )
            except Exception as exc:
                raise UnsupportedMediaError(
                    f"Could not convert uploaded file to PDF for OCR: {exc}",
                ) from exc

            async with _ocr_bulkhead:
                _run_ocr_script(
                    normalized.path, lang=ocr_lang, extract_mulkya=False,
                    is_pdf=True, prefer_pdf_text=prefer_pdf_text,
                )

            ocr_json_path = normalized.path.with_name(f"{normalized.path.stem}_ocr.json")
            if not ocr_json_path.exists():
                raise PipelineFailureError("OCR JSON output was not created.")

            raw_ocr = _load_json(ocr_json_path)
            extracted_data = {"lines": _flatten_ocr_lines(raw_ocr)}
            confidence_score = _mean_confidence(raw_ocr)

            payload = _build_pipeline_response(
                source_name=source_name,
                input_kind="pdf",
                classification=None,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                raw_ocr=raw_ocr,
                chunk_source_path=ocr_json_path,
                note="PDF OCR and text chunking executed.",
                car_classification=None,
                normalized_input=normalized,
                translation=_build_translation_payload(extracted_data, raw_ocr) if translate_to_en else None,
            )
            record_pipeline_latency("process_pdf", time.perf_counter() - pipeline_start)
            return json_success(payload, request=request, start_perf=request.state.start_perf)

    raise PipelineFailureError("Unhandled process_type.")


# ── Damage pipeline helpers ────────────────────────────────────────────────
def _damage_for_view(img_bytes: bytes, yolo_available: bool) -> dict[str, Any]:
    """Single-view damage classification + YOLO localisation.

    Kept for the 1-view fast path and as a fallback if batched inference
    isn't available. For multi-view requests, the endpoint uses
    ``_run_damage_inference_batch`` directly.
    """
    pred = DAMAGE_CB.call(lambda: _run_damage_inference(_preprocess_for_damage(img_bytes)))
    return _finish_damage_view(pred, img_bytes, yolo_available)


def _finish_damage_view(
    pred: dict[str, Any],
    img_bytes: bytes,
    yolo_available: bool,
) -> dict[str, Any]:
    """Run YOLO and the general-damage fallback once a binary prediction is in."""
    if pred["damage_detected"] and yolo_available:
        try:
            pred["damages"] = YOLO_CB.call(_run_yolo_pipeline, img_bytes)
        except Exception as exc:
            log.warning(
                "yolo failed for view; falling back",
                extra={"event": "yolo.failed", "exception": repr(exc)},
            )
            pred["damages"] = []
    else:
        pred["damages"] = []

    if pred["damage_detected"] and not pred["damages"]:
        p = pred["prob_damaged"]
        sev = "severe" if p >= 0.80 else "moderate" if p >= 0.65 else "minor"
        pred["damages"] = [{
            "type": "general-damage",
            "severity": sev,
            "confidence": round(p, 4),
            "bbox": [0.5, 0.5, 1.0, 1.0],
            "region": "unknown",
            "parts_at_risk": ["undetermined"],
            "replace": False,
            "repair_action": "Visual inspection required — damage detected but type could not be localized automatically",
        }]
    return pred


def _damage_batch_for_views(
    view_img_bytes: dict[str, bytes],
    yolo_available: bool,
) -> dict[str, dict[str, Any]]:
    """Batched damage path: one binary inference for N views, then per-view YOLO.

    Runs entirely on the threadpool side — call from ``run_in_threadpool``
    so the GIL doesn't bottleneck the event loop.
    """
    names = list(view_img_bytes.keys())
    if not names:
        return {}

    preprocessed = np.concatenate(
        [_preprocess_for_damage(view_img_bytes[name]) for name in names], axis=0
    )
    batch_preds = DAMAGE_CB.call(_run_damage_inference_batch, preprocessed)

    out: dict[str, dict[str, Any]] = {}
    for name, pred in zip(names, batch_preds):
        out[name] = _finish_damage_view(pred, view_img_bytes[name], yolo_available)
    return out


def _anpr_for_view(img_bytes: bytes) -> dict[str, Any]:
    try:
        result = ANPR_CB.call(_run_anpr_pipeline, img_bytes, is_oman_plate=True)
        return {
            "detected":   result.get("detected", False),
            "plate_text": result.get("plate_text", ""),
            "confidence": result.get("confidence", 0.0),
            "num_plates": result.get("num_plates", 0),
        }
    except Exception as exc:
        log.warning(
            "anpr failed; returning empty plate",
            extra={"event": "anpr.failed", "exception": repr(exc)},
        )
        return {
            "detected": False, "plate_text": "", "confidence": 0.0,
            "num_plates": 0, "error": "plate detection unavailable",
        }


@app.post("/predict/damage")
async def predict_damage(
    request: Request,
    front: UploadFile | None = File(default=None),
    back:  UploadFile | None = File(default=None),
    left:  UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
):
    views: dict[str, UploadFile] = {
        k: v for k, v in [("front", front), ("back", back), ("left", left), ("right", right)] if v
    }
    if not views:
        raise ValidationError("Provide at least one image (front/back/left/right).")

    try:
        _get_damage_session()
    except (FileNotFoundError, RuntimeError) as exc:
        raise ModelUnavailableError(str(exc)) from exc

    yolo_available = YOLO_MODEL_PATH.exists()
    pipeline_start = time.perf_counter()

    # Phase 1: read + decode all uploads (cannot await in threads).
    view_img_bytes: dict[str, bytes] = {}
    for view_name, upload in views.items():
        if not upload.filename:
            raise ValidationError(f"{view_name} upload must have a filename.")
        try:
            raw = await upload.read()
            if _is_pdf(upload):
                norm_bytes, _ = _pdf_first_page_to_jpeg(raw)
            else:
                norm_bytes, _ = _image_bytes_to_format(raw, output_format="JPEG")
            view_img_bytes[view_name] = norm_bytes
        except ApiError:
            raise
        except Exception as exc:
            raise UnsupportedMediaError(
                f"Could not process {view_name}: {exc}",
                details={"view": view_name},
            ) from exc

    # Phase 2: batched damage inference (single ORT call for N views)
    # in parallel with ANPR on the priority view.
    anpr_source_view = next((v for v in ANPR_VIEW_PRIORITY if v in view_img_bytes), None)
    run_anpr = _ANPR_AVAILABLE and anpr_source_view is not None

    async with _damage_bulkhead:
        damage_coro = run_in_threadpool(_damage_batch_for_views, view_img_bytes, yolo_available)
        if run_anpr:
            damage_batch, anpr_raw = await asyncio.gather(
                damage_coro,
                run_in_threadpool(_anpr_for_view, view_img_bytes[anpr_source_view]),
                return_exceptions=True,
            )
        else:
            damage_batch = await asyncio.gather(damage_coro, return_exceptions=True)
            damage_batch = damage_batch[0]
            anpr_raw = None

    # Aggregate damage results
    per_view: dict[str, Any] = {}
    overall_damaged = False
    max_confidence = 0.0
    any_per_view_error = False

    if isinstance(damage_batch, Exception):
        # One batch failure = mark every view as errored (consistent envelope).
        any_per_view_error = True
        api_err = to_api_error(damage_batch)
        for view_name in view_img_bytes:
            per_view[view_name] = {
                "error": {
                    "code": api_err.code,
                    "message": api_err.message,
                    "retryable": api_err.retryable,
                },
                "damage_detected": False,
                "damages": [],
            }
    else:
        for view_name, result in damage_batch.items():
            per_view[view_name] = result
            if result.get("damage_detected"):
                overall_damaged = True
                score = result.get("confidence_score", 0.0)
                if score > max_confidence:
                    max_confidence = score

    # Build plate result
    plate: dict[str, Any] = {"detected": False, "plate_text": "", "confidence": 0.0, "source_view": None}
    if anpr_raw is not None and not isinstance(anpr_raw, Exception):
        plate = {**anpr_raw, "source_view": anpr_source_view}
    elif isinstance(anpr_raw, Exception):
        plate["error"] = "plate detection unavailable"
    elif not _ANPR_AVAILABLE:
        plate["error"] = "ANPR module not available"

    record_pipeline_latency("predict_damage", time.perf_counter() - pipeline_start)

    payload = {
        "damage_detected":      overall_damaged,
        "total_views_analyzed": len(per_view),
        "overall_confidence":   round(max_confidence, 4),
        "per_view":             per_view,
        "plate":                plate,
        "any_view_error":       any_per_view_error,
    }
    return json_success(payload, request=request, start_perf=request.state.start_perf)


def _load_image(path: Path):
    with Image.open(path) as image:
        return image.copy()


def _load_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.copy()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    log_level = SETTINGS.log_level.lower()
    uvicorn.run(
        "poc_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=log_level,
        access_log=False,  # our middleware emits structured logs already
    )
