from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

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


class FileClassification(BaseModel):
    file_type: Literal["car", "card", "document", "other"] = Field(
        description="The classified type of the file."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        description="Explanation for the classification decision."
    )


def _classify_file_with_llm(
    file_path: Path,
    file_bytes: bytes,
    mime_type: str,
    llm_backend: str,
    gemini_api_key: str | None,
    ollama_host: str,
    ollama_model: str,
    card_threshold: float,
) -> dict[str, Any]:
    backend = (llm_backend or "gemini").lower()

    if backend == "gemini":
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Gemini API key not found. Falling back to heuristic/other methods.")
            backend = "heuristic"
        else:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                system_instruction = (
                    "You are a multimodal file routing classifier. Your job is to classify the uploaded file (image or PDF) "
                    "into one of the following categories:\n"
                    "- 'car': The file is an image of a car/vehicle (exterior, interior, engine, wheel, etc.).\n"
                    "- 'card': The file is a vehicle registration card (Mulkiya), identity card, driver's license, credit card, "
                    "or any card-like layout.\n"
                    "- 'document': The file is a scan or PDF of a text document (insurance policy, letter, invoice, contract, certificate).\n"
                    "- 'other': Any other image or file content.\n\n"
                    "Respond with a JSON object matching the FileClassification schema."
                )

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FileClassification,
                    temperature=0.0,
                    system_instruction=system_instruction,
                )

                part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[part, "Classify this file."],
                    config=config,
                )

                if response.text:
                    res_json = json.loads(response.text)
                    return {
                        "file_type": res_json.get("file_type", "document"),
                        "confidence": float(res_json.get("confidence", 1.0)),
                        "explanation": res_json.get("explanation", ""),
                        "backend_used": "gemini",
                    }
            except Exception as e:
                print(f"Gemini classification failed: {e}. Falling back to heuristic.")
                backend = "heuristic"

    if backend == "ollama":
        if mime_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
            return {
                "file_type": "document",
                "confidence": 1.0,
                "explanation": "PDF file automatically classified as document (Ollama backend vision bypass).",
                "backend_used": "ollama (bypass)",
            }

        try:
            b64_image = base64.b64encode(file_bytes).decode("utf-8")
            prompt = (
                "Analyze this image and classify it into one of the following categories:\n"
                "- 'car': The image is a picture of a car or vehicle component.\n"
                "- 'card': The image is a vehicle registration card, identity card, driver's license, or credit card.\n"
                "- 'document': The image is a text document page, scan, invoice, insurance form, or letter.\n"
                "- 'other': Any other image.\n\n"
                "Respond ONLY with a JSON object matching this schema:\n"
                "{\n"
                '  "file_type": "car" | "card" | "document" | "other",\n'
                '  "confidence": float,\n'
                '  "explanation": string\n'
                "}"
            )

            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [b64_image]}
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            }

            body = json.dumps(payload).encode("utf-8")
            url = ollama_host.rstrip("/") + "/api/chat"
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", "replace")

            res_payload = json.loads(raw)
            content = (res_payload.get("message") or {}).get("content")
            if content:
                res_json = json.loads(content)
                return {
                    "file_type": res_json.get("file_type", "document"),
                    "confidence": float(res_json.get("confidence", 1.0)),
                    "explanation": res_json.get("explanation", ""),
                    "backend_used": f"ollama ({ollama_model})",
                }
        except Exception as e:
            print(f"Ollama classification failed: {e}. Falling back to heuristic.")
            backend = "heuristic"

    # Heuristic fallback (using local Keras models if it's an image)
    if mime_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
        return {
            "file_type": "document",
            "confidence": 1.0,
            "explanation": "PDF file heuristically classified as document.",
            "backend_used": "heuristic",
        }

    try:
        card_model = _get_card_model()
        with Image.open(file_path) as img:
            prob = card_model.predict_probability(img, normalize=True)

        if prob >= card_threshold:
            return {
                "file_type": "card",
                "confidence": round(prob, 4),
                "explanation": f"Local card classifier predicted card probability of {prob:.4f} >= threshold {card_threshold}.",
                "backend_used": "heuristic (local card model)",
            }

        car_classification = _classify_car_image(file_path)
        if car_classification.get("is_car"):
            return {
                "file_type": "car",
                "confidence": car_classification.get("confidence", 1.0),
                "explanation": f"Local car classifier predicted car with confidence {car_classification.get('confidence')}.",
                "backend_used": "heuristic (local car model)",
            }
    except Exception as e:
        print(f"Heuristic image classification failed: {e}")

    return {
        "file_type": "document",
        "confidence": 0.5,
        "explanation": "Defaulted to document after all classification attempts and heuristics failed.",
        "backend_used": "heuristic (default)",
    }


app = FastAPI(title="UpSure PoC Document and Car Pipeline")

_card_model: CardNonCardModel | None = None
_car_model: Any | None = None
_car_img_size = CAR_FALLBACK_SIZE


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
    model = _get_car_model()
    with Image.open(image_path) as image:
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
        args.extend(["--extract_mulkya", "--llm_backend", "none"])

    if is_pdf and prefer_pdf_text:
        args.append("--prefer_pdf_text")

    completed = subprocess.run(
        args,
        cwd=str(POC_DIR),
        capture_output=True,
        text=True,
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
    llm_classification: dict[str, Any] | None = None,
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
        "llm_classification": llm_classification,
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

    with tempfile.TemporaryDirectory(prefix="upsure-car-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / Path(file.filename).name
        input_path.write_bytes(await file.read())

        try:
            classification = _classify_car_image(input_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(exc)}")

    return {
        "filename": Path(file.filename).name,
        **classification,
    }


@app.post("/api/v1/process")
async def process_document(
    file: UploadFile = File(...),
    card_threshold: float = 0.5,
    ocr_lang: str = "ar",
    prefer_pdf_text: bool = False,
    llm_backend: str = "gemini",
    gemini_api_key: str | None = None,
    x_gemini_api_key: str | None = Header(None, alias="X-Gemini-API-Key"),
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.2-vision",
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    if not _is_pdf(file) and not _is_image(file):
        raise HTTPException(status_code=415, detail="Please upload a PDF or image file.")

    source_name = Path(file.filename).name

    # Determine mime type
    mime_type = file.content_type
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(source_name)
    if not mime_type:
        mime_type = "application/octet-stream"

    with tempfile.TemporaryDirectory(prefix="upsure-poc-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / source_name
        
        file_bytes = await file.read()
        input_path.write_bytes(file_bytes)

        # 1. LLM-based file classification
        api_key = gemini_api_key or x_gemini_api_key or os.getenv("GEMINI_API_KEY")
        classification_result = _classify_file_with_llm(
            file_path=input_path,
            file_bytes=file_bytes,
            mime_type=mime_type,
            llm_backend=llm_backend,
            gemini_api_key=api_key,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            card_threshold=card_threshold,
        )
        file_type = classification_result["file_type"]

        # 2. Route based on classification result
        if file_type == "car":
            try:
                car_classification = _classify_car_image(input_path)
                confidence_score = car_classification.get("confidence", 0.0)
                note = f"File classified as 'car' by LLM ({classification_result.get('backend_used')}). Car classification model executed."
            except Exception as exc:
                car_classification = {
                    "error": str(exc),
                }
                confidence_score = 0.0
                note = f"File classified as 'car' by LLM ({classification_result.get('backend_used')}), but car classification model failed: {exc}"

            return _build_pipeline_response(
                source_name=source_name,
                input_kind="pdf" if _is_pdf(file) else "image",
                classification=None,
                confidence_score=confidence_score,
                extracted_data=None,
                raw_ocr=None,
                chunk_source_path=None,
                note=note,
                car_classification=car_classification,
                llm_classification=classification_result,
            )

        elif file_type == "card":
            is_pdf_file = _is_pdf(file)
            if not is_pdf_file:
                probability = _get_card_model().predict_probability(
                    _load_image(input_path),
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
                classification = None
                confidence_score = 1.0

            # Run OCR with Mulkiya extraction
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
                note = f"File classified as 'card' by LLM ({classification_result.get('backend_used')}). Card classifier and Mulkiya extractor executed."
            else:
                extracted_data = {
                    "lines": _flatten_ocr_lines(raw_ocr),
                }
                chunk_source_path = ocr_json_path
                confidence_score = _mean_confidence(raw_ocr)
                note = f"File classified as 'card' by LLM ({classification_result.get('backend_used')}), but Mulkiya JSON output was not created. Returned OCR text lines."

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
                llm_classification=classification_result,
            )

        else:  # document or other
            is_pdf_file = _is_pdf(file)
            # Run OCR without Mulkiya extraction
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
            note = f"File classified as '{file_type}' by LLM ({classification_result.get('backend_used')}). General OCR and text chunking executed."

            return _build_pipeline_response(
                source_name=source_name,
                input_kind="pdf" if is_pdf_file else "image",
                classification=None,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                raw_ocr=raw_ocr,
                chunk_source_path=ocr_json_path,
                note=note,
                car_classification=None,
                llm_classification=classification_result,
            )


def _load_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)