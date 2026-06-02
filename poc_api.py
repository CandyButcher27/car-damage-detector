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
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
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
CAR_MODEL_PATH = POC_DIR / "models" / "best_car_model.keras"
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


from fastapi.middleware.cors import CORSMiddleware

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
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return content_type.startswith("image/") or any(filename.endswith(ext) for ext in image_exts)


def _guess_mime_type(upload: UploadFile, source_name: str) -> str:
    content_type = (upload.content_type or "").strip().lower()
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(source_name)
    return guessed or "application/octet-stream"


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

    return {
        "input": {
            "filename": source_name,
            "kind": input_kind,
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


@app.get("/")
def root():
    return {
        "message": "UpSure PoC Document and Car Pipeline API is running. Send POST requests to /api/v1/process"
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "car_model_path": str(CAR_MODEL_PATH),
        "ocr_script": str(OCR_SCRIPT),
    }


@app.post("/predict/")
async def predict_car(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image = _load_image_bytes(await file.read())
        classification = _classify_car_image_from_image(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(exc)}")

    return {
        "filename": Path(file.filename).name,
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
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    if process_type not in PROCESS_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"process_type must be one of: {', '.join(PROCESS_TYPES)}",
        )

    if process_type != "file" and not _is_pdf(file) and not _is_image(file):
        raise HTTPException(status_code=415, detail="Please upload a PDF or image file.")

    source_name = Path(file.filename).name
    is_pdf_file = _is_pdf(file)
    is_image_file = _is_image(file)
    mime_type = _guess_mime_type(file, source_name)

    if process_type == "pdf" and not is_pdf_file:
        raise HTTPException(status_code=400, detail="process_type='pdf' requires a PDF file.")
    if process_type in {"car", "mulkiya"} and not is_image_file and not is_pdf_file:
        raise HTTPException(status_code=400, detail="process_type='car' or 'mulkiya' requires an image or PDF file.")
    if process_type == "car" and is_pdf_file:
        raise HTTPException(status_code=400, detail="process_type='car' requires an image file, not PDF.")

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
            )

        if process_type == "car":
            try:
                image = _load_image_bytes(file_bytes)
                car_classification = _classify_car_image_from_image(image)
                confidence_score = car_classification.get("confidence", 0.0)
                note = "Car classification model executed."
            except Exception as exc:
                car_classification = {
                    "error": str(exc),
                }
                confidence_score = 0.0
                note = f"Car classification model failed: {exc}"

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
            )

        if process_type == "mulkiya":
            if not is_pdf_file:
                image = _load_image_bytes(file_bytes)
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
                )

            input_path.write_bytes(file_bytes)
            _run_ocr_script(
                input_path,
                lang=ocr_lang,
                extract_mulkya=True,
                is_pdf=is_pdf_file,
                prefer_pdf_text=prefer_pdf_text if is_pdf_file else False,
            )

            ocr_json_path = input_path.with_name(f"{input_path.stem}_ocr.json")
            if not ocr_json_path.exists():
                raise HTTPException(status_code=500, detail="OCR JSON output was not created.")

            raw_ocr = _load_json(ocr_json_path)
            extracted_data_path = input_path.with_name(f"{input_path.stem}_mulkya.json")

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
            )

        if process_type == "pdf":
            input_path.write_bytes(file_bytes)
            _run_ocr_script(
                input_path,
                lang=ocr_lang,
                extract_mulkya=False,
                is_pdf=is_pdf_file,
                prefer_pdf_text=prefer_pdf_text if is_pdf_file else False,
            )

            ocr_json_path = input_path.with_name(f"{input_path.stem}_ocr.json")
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
            )

    raise HTTPException(status_code=500, detail="Unhandled process_type.")


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
