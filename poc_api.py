from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps

from card_inference import CardNonCardModel, _resolve_model_path
from rag_json_chunker import chunk_json_file

try:
    from keras.models import load_model as load_keras_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model as load_keras_model
    except ImportError:
        load_keras_model = None

# Monkeypatch Keras layers to support loading legacy models in newer Keras/TensorFlow versions
if load_keras_model is not None:
    try:
        import keras
        for layer_cls in [keras.layers.Dense, keras.layers.Conv2D]:
            if hasattr(layer_cls, "__init__"):
                orig_init = layer_cls.__init__

                def _make_patched_init(orig):
                    def patched_init(self, *args, **kwargs):
                        kwargs.pop("quantization_config", None)
                        orig(self, *args, **kwargs)

                    return patched_init

                layer_cls.__init__ = _make_patched_init(orig_init)
    except Exception as e:
        print(f"Warning: Failed to monkeypatch Keras layers: {e}")


POC_DIR = Path(__file__).resolve().parent
OCR_SCRIPT = POC_DIR / "ocr_simple_test.py"
MODEL_PATH = _resolve_model_path(POC_DIR, None)
CAR_MODEL_PATH    = POC_DIR / "models" / "best_car_model.keras"
DAMAGE_THRESHOLD  = 0.25
DAMAGE_IMG_SIZE   = 260
DAMAGE_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DAMAGE_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OCR_PYTHON_DEFAULT = Path("D:/UpSure/OCR_test/venv/Scripts/python.exe")
if not OCR_PYTHON_DEFAULT.exists():
    OCR_PYTHON_DEFAULT = POC_DIR.parent / "OCR_test" / "venv" / "Scripts" / "python.exe"

OCR_PYTHON = Path(
    os.getenv(
        "UPSURE_OCR_PYTHON",
        str(OCR_PYTHON_DEFAULT),
    )
)
CAR_THRESHOLD = 0.35
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


app = FastAPI(title="UpSure PoC Document and Car Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_card_model: CardNonCardModel | None = None
_car_model: Any | None = None
_car_img_size = CAR_FALLBACK_SIZE
_damage_session: ort.InferenceSession | None = None


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started_at) * 1000.0:.2f}"
    return response


def _get_card_model() -> CardNonCardModel:
    global _card_model

    if _card_model is None:
        _card_model = CardNonCardModel.load(MODEL_PATH)
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
        if load_keras_model is None:
            raise RuntimeError("TensorFlow or Keras is required for car classification.")
        if not CAR_MODEL_PATH.exists():
            raise FileNotFoundError(f"Car model file not found at {CAR_MODEL_PATH}")

        _car_model = load_keras_model(str(CAR_MODEL_PATH))
        _car_img_size = _get_car_img_size(_car_model)
    return _car_model


def _get_damage_session() -> ort.InferenceSession:
    global _damage_session
    if _damage_session is None:
        if not DAMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Damage model not found at {DAMAGE_MODEL_PATH}")
        try:
            _damage_session = ort.InferenceSession(
                str(DAMAGE_MODEL_PATH), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            message = str(exc)
            if ".onnx.data" in message:
                raise RuntimeError(
                    "Damage model is incomplete. "
                    f"ONNX Runtime expects a companion external-data file next to {DAMAGE_MODEL_PATH.name}: "
                    f"{DAMAGE_MODEL_PATH.name}.data"
                ) from exc
            raise RuntimeError(f"Failed to initialize damage model at {DAMAGE_MODEL_PATH}: {message}") from exc
    return _damage_session


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
        "damage_detected":  bool(label == 1),
        "confidence_score": float(round(float(probs[label]), 4)),
        "prob_damaged":     float(round(float(probs[1]), 4)),
        "prob_clean":       float(round(float(probs[0]), 4)),
    }


def _classify_car_image(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        return _classify_car_image_from_image(image)


def _classify_car_image_from_image(image: Image.Image) -> dict[str, Any]:
    model = _get_car_model()
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _sanitize_for_json(exc.errors())})


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
        raise RuntimeError("PyMuPDF is required to convert PDFs into images for this process_type.") from exc

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise RuntimeError("PDF has no pages to convert.")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        png_bytes = pixmap.tobytes("png")

    jpeg_bytes, image_details = _image_bytes_to_format(png_bytes, output_format="JPEG")
    image_details.update(
        {
            "source_kind": "pdf",
            "target_kind": "image",
            "target_suffix": MODEL_IMAGE_SUFFIX,
            "target_mime_type": "image/jpeg",
            "pdf_page_used": 1,
        }
    )
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
        path=path,
        data=data,
        kind="image",
        mime_type="image/jpeg",
        converted=True,
        details=details,
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
            path=path,
            data=file_bytes,
            kind="pdf",
            mime_type="application/pdf",
            converted=converted,
            details={
                "source_kind": "pdf",
                "target_kind": "pdf",
                "target_suffix": PDF_SUFFIX,
                "target_mime_type": "application/pdf",
                "converted": converted,
            },
        )

    if target_pdf:
        data, details = _image_bytes_to_pdf(file_bytes)
        path = _write_normalized(temp_dir, source_name, PDF_SUFFIX, data)
        return NormalizedInput(
            path=path,
            data=data,
            kind="pdf",
            mime_type="application/pdf",
            converted=True,
            details=details,
        )

    data, details = _image_bytes_to_format(file_bytes, output_format="PNG")
    path = _write_normalized(temp_dir, source_name, OCR_IMAGE_SUFFIX, data)
    return NormalizedInput(
        path=path,
        data=data,
        kind="image",
        mime_type="image/png",
        converted=True,
        details=details,
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
                lines.append(
                    {
                        "page": page_number,
                        "line_index": line_index,
                        "text": line.get("text"),
                        "confidence": line.get("confidence"),
                    }
                )
        return lines

    image_lines = payload.get("lines")
    if isinstance(image_lines, list):
        for line_index, line in enumerate(image_lines, start=1):
            if not isinstance(line, dict):
                continue
            lines.append(
                {
                    "page": payload.get("page", 1),
                    "line_index": line_index,
                    "text": line.get("text"),
                    "confidence": line.get("confidence"),
                }
            )

    return lines


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
        translated[key] = {
            "label": label,
            "value": value,
            "translation": translated_value,
        }
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

    completed = subprocess.run(
        args,
        cwd=str(POC_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        message = ["OCR pipeline failed."]
        if completed.stdout:
            message.append(f"STDOUT:\n{completed.stdout}")
        if completed.stderr:
            message.append(f"STDERR:\n{completed.stderr}")
        raise RuntimeError("\n".join(message))

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
        "artifacts": {
            "chunk_source": artifact_chunk_source,
        },
        "note": note,
    }
    if translation is not None:
        response["translation"] = translation
    return response


@app.get("/")
def root():
    return {
        "message": "UpSure PoC Document and Car Pipeline API is running. Send POST requests to /api/v1/process"
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    damage_model_ready = True
    damage_model_error = None
    try:
        _get_damage_session()
    except Exception as exc:
        damage_model_ready = False
        damage_model_error = str(exc)

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "car_model_path": str(CAR_MODEL_PATH),
        "damage_model_path": str(DAMAGE_MODEL_PATH),
        "damage_model_ready": damage_model_ready,
        "damage_model_error": damage_model_error,
        "ocr_script": str(OCR_SCRIPT),
    }


@app.post("/predict/")
async def predict_car(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    source_name = Path(file.filename).name
    is_pdf_file = _is_pdf(file)
    with tempfile.TemporaryDirectory(prefix="upsure-car-") as temp_dir_name:
        try:
            normalized = _normalize_for_image_model(
                file_bytes=await file.read(),
                source_name=source_name,
                temp_dir=Path(temp_dir_name),
                is_pdf_file=is_pdf_file,
            )
            image = _load_image_bytes(normalized.data)
            classification = _classify_car_image_from_image(image)
        except Exception as exc:
            raise HTTPException(status_code=415, detail=f"Could not convert uploaded file to a model image: {str(exc)}")

    return {
        "filename": source_name,
        "normalized": {
            "kind": normalized.kind,
            "mime_type": normalized.mime_type,
            "converted": normalized.converted,
            "details": normalized.details,
        },
        **classification,
    }


@app.post("/api/v1/process")
async def process_document(
    file: UploadFile = File(...),
    process_type: Literal["car", "mulkiya", "pdf", "file"] = Form(...),
    card_threshold: float = Form(0.5),
    ocr_lang: str = Form("ar"),
    prefer_pdf_text: bool = Form(False),
    skip_ocr: bool = Form(False),
    translate_to_en: bool = Form(False),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    if process_type not in PROCESS_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"process_type must be one of: {', '.join(PROCESS_TYPES)}",
        )

    source_name = Path(file.filename).name
    is_pdf_file = _is_pdf(file)
    mime_type = _guess_mime_type(file, source_name)

    with tempfile.TemporaryDirectory(prefix="upsure-poc-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / source_name

        file_bytes = await file.read()

        if process_type == "file":
            input_path.write_bytes(file_bytes)
            details = _collect_general_file_details(
                input_path,
                source_name=source_name,
                mime_type=mime_type,
            )
            details_path = input_path.with_name(f"{input_path.stem}_file_summary.json")
            _write_json(details_path, details)
            note = "General file inspection executed without OCR."

            return _build_pipeline_response(
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
                note=note,
                car_classification=None,
                translation=_build_translation_payload(details, None) if translate_to_en else None,
            )

        if process_type == "car":
            try:
                normalized = _normalize_for_image_model(
                    file_bytes=file_bytes,
                    source_name=source_name,
                    temp_dir=temp_dir,
                    is_pdf_file=is_pdf_file,
                )
                image = _load_image_bytes(normalized.data)
                car_classification = _classify_car_image_from_image(image)
                confidence_score = car_classification.get("confidence", 0.0)
                note = "Car classification model executed."
            except Exception as exc:
                raise HTTPException(
                    status_code=415,
                    detail=f"Could not convert uploaded file to a car model image: {exc}",
                )

            return _build_pipeline_response(
                source_name=source_name,
                input_kind="image",
                classification=None,
                confidence_score=confidence_score,
                extracted_data=None,
                raw_ocr=None,
                chunk_source_path=None,
                note=note,
                car_classification=car_classification,
                normalized_input=normalized,
            )

        if process_type == "mulkiya":
            if not is_pdf_file:
                try:
                    model_input = _normalize_for_image_model(
                        file_bytes=file_bytes,
                        source_name=source_name,
                        temp_dir=temp_dir,
                        is_pdf_file=False,
                    )
                    image = _load_image_bytes(model_input.data)
                except Exception as exc:
                    raise HTTPException(
                        status_code=415,
                        detail=f"Could not convert uploaded file to a Mulkiya model image: {exc}",
                    )
                probability = _get_card_model().predict_probability(
                    image,
                    normalize=True,
                )
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
                note = "Mulkiya classification executed without OCR."
                return _build_pipeline_response(
                    source_name=source_name,
                    input_kind="pdf" if is_pdf_file else "image",
                    classification=classification,
                    confidence_score=confidence_score,
                    extracted_data=None,
                    raw_ocr=None,
                    chunk_source_path=None,
                    note=note,
                    car_classification=None,
                    normalized_input=model_input,
                )

            try:
                ocr_input = _normalize_for_ocr(
                    file_bytes=file_bytes,
                    source_name=source_name,
                    temp_dir=temp_dir,
                    is_pdf_file=is_pdf_file,
                    target_pdf=False,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=415,
                    detail=f"Could not convert uploaded file to an OCR-supported format: {exc}",
                )
            _run_ocr_script(
                ocr_input.path,
                lang=ocr_lang,
                extract_mulkya=True,
                is_pdf=ocr_input.kind == "pdf",
                prefer_pdf_text=prefer_pdf_text if ocr_input.kind == "pdf" else False,
            )

            ocr_json_path = ocr_input.path.with_name(f"{ocr_input.path.stem}_ocr.json")
            if not ocr_json_path.exists():
                raise HTTPException(status_code=500, detail="OCR JSON output was not created.")

            raw_ocr = _load_json(ocr_json_path)
            extracted_data_path = ocr_input.path.with_name(f"{ocr_input.path.stem}_mulkya.json")

            if extracted_data_path.exists():
                extracted_data = _load_json(extracted_data_path)
                chunk_source_path = extracted_data_path
                note = "Mulkiya pipeline executed with card inference and OCR extraction."
            else:
                extracted_data = {
                    "lines": _flatten_ocr_lines(raw_ocr),
                }
                chunk_source_path = ocr_json_path
                confidence_score = _mean_confidence(raw_ocr)
                note = "Mulkiya OCR ran but structured Mulkiya JSON was not created; returned OCR lines."

            return _build_pipeline_response(
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

        if process_type == "pdf":
            try:
                normalized = _normalize_for_ocr(
                    file_bytes=file_bytes,
                    source_name=source_name,
                    temp_dir=temp_dir,
                    is_pdf_file=is_pdf_file,
                    target_pdf=True,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=415,
                    detail=f"Could not convert uploaded file to PDF for OCR: {exc}",
                )
            _run_ocr_script(
                normalized.path,
                lang=ocr_lang,
                extract_mulkya=False,
                is_pdf=True,
                prefer_pdf_text=prefer_pdf_text,
            )

            ocr_json_path = normalized.path.with_name(f"{normalized.path.stem}_ocr.json")
            if not ocr_json_path.exists():
                raise HTTPException(status_code=500, detail="OCR JSON output was not created.")

            raw_ocr = _load_json(ocr_json_path)
            extracted_data = {
                "lines": _flatten_ocr_lines(raw_ocr),
            }
            confidence_score = _mean_confidence(raw_ocr)
            note = "PDF OCR and text chunking executed."

            return _build_pipeline_response(
                source_name=source_name,
                input_kind="pdf",
                classification=None,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                raw_ocr=raw_ocr,
                chunk_source_path=ocr_json_path,
                note=note,
                car_classification=None,
                normalized_input=normalized,
                translation=_build_translation_payload(extracted_data, raw_ocr) if translate_to_en else None,
            )

    raise HTTPException(status_code=500, detail="Unhandled process_type.")


@app.post("/predict/damage")
async def predict_damage(
    front: UploadFile | None = File(default=None),
    back:  UploadFile | None = File(default=None),
    left:  UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """
    Car damage detection endpoint.
    Accepts 1–4 vehicle view images (front/back/left/right).
    All views that are confirmed cars are run through the damage model.
    damage_detected=true if ANY view is damaged.

    Call this AFTER /api/v1/process?process_type=car confirms is_car=true for each view.
    """
    views: dict[str, UploadFile] = {
        k: v for k, v in [("front", front), ("back", back), ("left", left), ("right", right)] if v
    }

    if not views:
        raise HTTPException(status_code=400, detail="Provide at least one image (front/back/left/right).")

    try:
        _get_damage_session()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    per_view: dict[str, Any] = {}
    overall_damaged = False
    max_confidence = 0.0

    with tempfile.TemporaryDirectory(prefix="upsure-damage-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for view_name, upload in views.items():
            if not upload.filename:
                raise HTTPException(status_code=400, detail=f"{view_name} upload must have a filename.")
            try:
                normalized = _normalize_for_image_model(
                    file_bytes=await upload.read(),
                    source_name=Path(upload.filename).name,
                    temp_dir=temp_dir,
                    is_pdf_file=_is_pdf(upload),
                )
                arr = _preprocess_for_damage(normalized.data)
            except Exception as exc:
                raise HTTPException(
                    status_code=415,
                    detail=f"Could not convert {view_name} upload to a damage model image: {exc}",
                )
            pred = _run_damage_inference(arr)
            pred["normalized"] = {
                "kind": normalized.kind,
                "mime_type": normalized.mime_type,
                "converted": normalized.converted,
                "details": normalized.details,
            }
            per_view[view_name] = pred
            if pred["damage_detected"]:
                overall_damaged = True
            if pred["confidence_score"] > max_confidence:
                max_confidence = pred["confidence_score"]

    return {
        "damage_detected":      overall_damaged,
        "total_views_analyzed": len(views),
        "overall_confidence":   round(max_confidence, 4),
        "per_view":             per_view,
    }


def _load_image(path: Path):
    with Image.open(path) as image:
        return image.copy()


def _load_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.copy()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
