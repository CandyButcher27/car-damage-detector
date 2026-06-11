from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from onnx_inference import BinaryOnnxImageClassifier, YoloOnnxDetector, resolve_model_path


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("upsure.damage")

POC_DIR = Path(__file__).resolve().parent

CAR_MODEL_PATH = resolve_model_path(
    POC_DIR,
    "UPSURE_CAR_MODEL",
    [
        "models/digiLifeDoc_best_car_model_v2.onnx",
        "models/best_car_model_v2.onnx",
    ],
)
DAMAGE_MODEL_PATH = resolve_model_path(
    POC_DIR,
    "UPSURE_DAMAGE_MODEL",
    [
        "models/digiLifeDoc_damage_model.onnx",
        "models/damage_model.onnx",
    ],
)
DAMAGE_DETECTOR_PATH = resolve_model_path(
    POC_DIR,
    "UPSURE_DAMAGE_DETECTOR_MODEL",
    [
        "models/digiLifeDoc_damage_detector_v2.onnx",
        "models/damage_detector_v2.onnx",
    ],
)

CAR_THRESHOLD = 0.35
DAMAGE_THRESHOLD = 0.25
DAMAGE_IMG_SIZE = 260
DAMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DAMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_IMAGE_SUFFIX = ".jpg"


@dataclass(slots=True)
class NormalizedInput:
    path: Path
    data: bytes
    kind: Literal["image"]
    mime_type: str
    converted: bool
    details: dict[str, Any]


app = FastAPI(title="UpSure Car Damage Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_car_model: BinaryOnnxImageClassifier | None = None
_damage_session: ort.InferenceSession | None = None
_damage_detector: YoloOnnxDetector | None = None


def _get_car_model() -> BinaryOnnxImageClassifier:
    global _car_model
    if _car_model is None:
        logger.info("Loading car model from %s", CAR_MODEL_PATH)
        _car_model = BinaryOnnxImageClassifier(
            CAR_MODEL_PATH,
            positive_label="car",
            negative_label="non_car",
            positive_when_output_high=False,
        )
    return _car_model


def _get_damage_session() -> ort.InferenceSession:
    global _damage_session
    if _damage_session is None:
        if not DAMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Damage model not found at {DAMAGE_MODEL_PATH}")
        logger.info("Loading damage classifier from %s", DAMAGE_MODEL_PATH)
        try:
            _damage_session = ort.InferenceSession(
                str(DAMAGE_MODEL_PATH), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize damage model at {DAMAGE_MODEL_PATH}: {exc}"
            ) from exc
    return _damage_session


def _get_damage_detector() -> YoloOnnxDetector:
    global _damage_detector
    if _damage_detector is None:
        logger.info("Loading damage detector from %s", DAMAGE_DETECTOR_PATH)
        _damage_detector = YoloOnnxDetector(DAMAGE_DETECTOR_PATH)
    return _damage_detector


def _load_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.copy()


def _is_pdf(upload: UploadFile) -> bool:
    filename = Path(upload.filename or "").name.lower()
    content_type = (upload.content_type or "").lower()
    return filename.endswith(".pdf") or content_type == "application/pdf"


def _safe_stem(source_name: str) -> str:
    stem = Path(source_name).stem.strip()
    return stem or "upload"


def _image_bytes_to_jpeg(file_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(file_bytes)) as image:
        original_format = image.format
        original_mode = image.mode
        original_size = [image.width, image.height]
        prepared_image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        prepared_image.save(buffer, format="JPEG", quality=95, optimize=True)

    return buffer.getvalue(), {
        "source_kind": "image",
        "target_kind": "image",
        "target_suffix": MODEL_IMAGE_SUFFIX,
        "target_mime_type": "image/jpeg",
        "original_format": original_format,
        "original_mode": original_mode,
        "original_size": original_size,
        "converted": True,
    }


def _pdf_first_page_to_jpeg(file_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to convert PDFs into images.") from exc

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise RuntimeError("PDF has no pages to convert.")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        png_bytes = pixmap.tobytes("png")

    jpeg_bytes, details = _image_bytes_to_jpeg(png_bytes)
    details.update({"source_kind": "pdf", "pdf_page_used": 1})
    return jpeg_bytes, details


def _write_normalized(temp_dir: Path, source_name: str, data: bytes) -> Path:
    path = temp_dir / f"{_safe_stem(source_name)}_normalized{MODEL_IMAGE_SUFFIX}"
    path.write_bytes(data)
    return path


def _normalize_for_image_model(
    *, file_bytes: bytes, source_name: str, temp_dir: Path, is_pdf_file: bool
) -> NormalizedInput:
    if is_pdf_file:
        data, details = _pdf_first_page_to_jpeg(file_bytes)
    else:
        data, details = _image_bytes_to_jpeg(file_bytes)

    path = _write_normalized(temp_dir, source_name, data)
    return NormalizedInput(
        path=path,
        data=data,
        kind="image",
        mime_type="image/jpeg",
        converted=True,
        details=details,
    )


def _classify_car_image_from_image(image: Image.Image) -> dict[str, Any]:
    classification = _get_car_model().classify(image, CAR_THRESHOLD)
    is_car = classification["label"] == "car"
    return {
        "is_car": is_car,
        "confidence": classification["confidence"],
        "car_probability": classification["car_probability"],
        "non_car_probability": classification["non_car_probability"],
        "threshold_used": CAR_THRESHOLD,
    }


def _preprocess_for_damage(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((DAMAGE_IMG_SIZE, DAMAGE_IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - DAMAGE_MEAN) / DAMAGE_STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def _run_damage_inference(arr: np.ndarray) -> dict[str, Any]:
    sess = _get_damage_session()
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: arr})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    label = 1 if probs[1] >= DAMAGE_THRESHOLD else 0
    return {
        "damage_detected": bool(label == 1),
        "confidence_score": float(round(float(probs[label]), 4)),
        "prob_damaged": float(round(float(probs[1]), 4)),
        "prob_clean": float(round(float(probs[0]), 4)),
    }


def _run_damage_detector(image_bytes: bytes) -> dict[str, Any]:
    image = _load_image_bytes(image_bytes)
    return _get_damage_detector().detect(image)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "UpSure Car Damage Detection API. POST images to /predict/damage"}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    def model_status(loader) -> dict[str, Any]:
        try:
            loader()
            return {"ready": True, "error": None}
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    car_status = model_status(_get_car_model)
    damage_status = model_status(_get_damage_session)
    detector_status = model_status(_get_damage_detector)

    return {
        "status": "ok",
        "models": {
            "car": {**car_status, "path": str(CAR_MODEL_PATH)},
            "damage_classifier": {**damage_status, "path": str(DAMAGE_MODEL_PATH)},
            "damage_detector": {**detector_status, "path": str(DAMAGE_DETECTOR_PATH)},
        },
    }


@app.post("/predict/damage")
async def predict_damage(
    request: Request,
    front: UploadFile | None = File(default=None),
    back: UploadFile | None = File(default=None),
    left: UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """
    Car damage detection.

    Accepts 1-4 vehicle views (front/back/left/right), or a single image via
    file/image/upload. Per view, in order:
      1. best_car_model  -> if not a car, the view is skipped (no damage steps).
      2. damage_model     -> classifies damaged / clean.
      3. damage_detector  -> only runs when the classifier flags damage, to locate it.

    Each view is independent: a non-car view is skipped without failing the request.
    """
    # Ignore empty form fields (a Postman key with no file selected has no filename).
    views: dict[str, UploadFile] = {
        k: v
        for k, v in [("front", front), ("back", back), ("left", left), ("right", right)]
        if v and v.filename
    }

    if not views:
        # Fallback: accept any uploaded files regardless of field name, mapping
        # them to front/back/left/right in order. Keeps Postman uploads working
        # even when the form key does not match the named views.
        form = await request.form()
        uploaded = [
            value
            for _, value in form.multi_items()
            if hasattr(value, "filename") and hasattr(value, "read") and value.filename
        ]
        for slot, upload in zip(("front", "back", "left", "right"), uploaded):
            views[slot] = upload

    if not views:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Send 1-4 images as form-data files.",
        )

    try:
        _get_car_model()
        _get_damage_session()
        _get_damage_detector()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Model load failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    per_view: dict[str, Any] = {}
    overall_damaged = False
    max_confidence = 0.0

    with tempfile.TemporaryDirectory(prefix="upsure-damage-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for view_name, upload in views.items():
            logger.info("[%s] processing upload '%s'", view_name, upload.filename)
            try:
                normalized = _normalize_for_image_model(
                    file_bytes=await upload.read(),
                    source_name=Path(upload.filename).name,
                    temp_dir=temp_dir,
                    is_pdf_file=_is_pdf(upload),
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=415,
                    detail=f"Could not convert {view_name} upload to a model image: {exc}",
                )

            # Stage 1: is it a car?
            car = _classify_car_image_from_image(_load_image_bytes(normalized.data))
            logger.info(
                "[%s] car check: is_car=%s car_prob=%.4f",
                view_name, car["is_car"], car["car_probability"],
            )
            if not car["is_car"]:
                logger.info("[%s] not a car -> skipping damage analysis", view_name)
                per_view[view_name] = {
                    "status": "skipped",
                    "reason": "not_a_car",
                    "message": f"The '{view_name}' image is not a car. Please upload a valid car image for this view.",
                    "car": car,
                }
                continue

            # Stage 2: damage classifier.
            classifier = _run_damage_inference(_preprocess_for_damage(normalized.data))
            logger.info(
                "[%s] damage classifier: damaged=%s prob_damaged=%.4f",
                view_name, classifier["damage_detected"], classifier["prob_damaged"],
            )

            # Stage 3: detector only when classifier flags damage.
            detector: dict[str, Any] | None = None
            if classifier["damage_detected"]:
                detector = _run_damage_detector(normalized.data)
                logger.info(
                    "[%s] damage detector: detections=%d max_conf=%.4f",
                    view_name, len(detector["detections"]), detector["max_confidence"],
                )
            else:
                logger.info("[%s] classifier clean -> skipping detector", view_name)

            view_damaged = classifier["damage_detected"]
            damage_confidence = classifier["prob_damaged"]
            if detector is not None and detector["damage_detected"]:
                damage_confidence = max(damage_confidence, detector["max_confidence"])

            per_view[view_name] = {
                "status": "analyzed",
                "car": car,
                "damage_detected": view_damaged,
                "damage_confidence": round(damage_confidence, 4),
                "classifier": classifier,
                "detector": detector,
            }

            if view_damaged:
                overall_damaged = True
            max_confidence = max(max_confidence, damage_confidence)

    analyzed = [v for v in per_view.values() if v.get("status") == "analyzed"]
    skipped = [name for name, v in per_view.items() if v.get("status") == "skipped"]
    logger.info(
        "request done: views=%d analyzed=%d damaged=%s",
        len(views), len(analyzed), overall_damaged,
    )

    if not analyzed:
        message = "No valid car image found. Please upload a car image."
    elif skipped:
        message = f"Processed {len(analyzed)} car image(s); not a car (skipped): {', '.join(skipped)}."
    else:
        message = f"Processed {len(analyzed)} car image(s)."

    return {
        "damage_detected": overall_damaged,
        "message": message,
        "total_views_received": len(views),
        "views_analyzed": len(analyzed),
        "views_skipped": len(views) - len(analyzed),
        "overall_confidence": round(max_confidence, 4),
        "per_view": per_view,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
