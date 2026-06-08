# UpSure PoC Documentation

This repository contains a proof-of-concept FastAPI service for document, Mulkiya, car, OCR, and damage-analysis workflows. The main goal is to take a single upload, route it through the right pipeline, and return a structured response that is easy to consume in downstream systems.

The project is centered around `poc_api.py`, which exposes one unified API for most workflows, plus a standalone car-classifier service for backward compatibility.

## What This Project Does

The service can:

1. Detect whether an uploaded image contains a car.
2. Classify Mulkiya-style documents as card or not-card.
3. Run OCR on images and PDFs.
4. Extract Mulkiya data when OCR is enabled.
5. Inspect general files and return useful metadata and previews.
6. Detect damage on one to four vehicle-view images (binary: damaged vs clean).
7. When damage is detected, run a five-stage severity pipeline: damage type, severity (minor/moderate/severe), parts at risk, and repair or replace recommendation.
8. Produce RAG-friendly JSON chunks from OCR or structured JSON outputs.
9. Benchmark latency across the available routes.

## Main Files

| File | Purpose |
|---|---|
| `poc_api.py` | Main FastAPI app. Handles unified routing, OCR orchestration, file normalization, and damage detection. |
| `car_classifier_api.py` | Standalone car-classifier API kept for compatibility and quick testing. |
| `card_inference.py` | Lightweight card vs not-card model loader plus CLI and GUI helpers. |
| `ocr_simple_test.py` | OCR pipeline script used by the API to produce OCR JSON and Mulkiya extraction artifacts. |
| `rag_json_chunker.py` | Turns OCR or structured JSON into overlapping chunks that are suitable for retrieval-augmented workflows. |
| `latency_analyzer.py` | Benchmarks the unified API and reports latency-focused metrics. |
| `benchmark_everything.py` | Broader benchmark runner that collects latency plus response-quality metrics and can save results. |
| `requirements.txt` | Python dependencies for the project. |
| `Samples/` | Example files for smoke testing, manual validation, and benchmarking. |

## Folder Layout

```text
.
|-- poc_api.py
|-- car_classifier_api.py
|-- card_inference.py
|-- ocr_simple_test.py
|-- rag_json_chunker.py
|-- latency_analyzer.py
|-- benchmark_everything.py
|-- requirements.txt
|-- README.md
|-- models/
`-- Samples/
```

The `models/` directory is expected to contain downloaded model artifacts. It is intentionally not meant for checked-in model binaries.

## How The Unified API Works

The main service accepts an uploaded file and a `process_type` value. That value determines the pipeline:

| `process_type` | What happens |
|---|---|
| `car` | Converts the input into a model-ready image and runs the car classifier. |
| `mulkiya` | Runs card vs not-card classification, then OCR and Mulkiya extraction unless `skip_ocr=true`. |
| `pdf` | Ensures the file is available as PDF, then runs OCR and returns extracted line data. |
| `file` | Does not run OCR or classification. Returns file metadata, previews, and lightweight inspection data. |

The API is designed to normalize input before inference:

1. Images can be converted to JPEG or PNG depending on the target pipeline.
2. PDFs can be rendered to the first page as an image when a model needs image input.
3. Images can also be wrapped into a one-page PDF when the OCR pipeline expects PDF input.
4. The response includes a `normalized` block so you can see what happened to the original file.

## Model Artifacts

Download the model files and place them in `models/` before starting the server.

| Model file | Used by |
|---|---|
| `best_car_model_v2.keras` | Car detection in `process_type=car`, `/predict/`, and `/predict/damage`. |
| `card_noncard_classifier_model.keras` | Mulkiya card vs not-card classification. The loader also accepts `card_noncard_model.keras` if you use that legacy filename. |
| `damage_model.onnx` | Stage 1 binary damage detection on `/predict/damage` (EfficientNet-B2, ONNX). |
| `damage_detector_v2.onnx` | Stage 2 damage-type detection on `/predict/damage` (YOLOv8n, 6 classes, ONNX). Only runs when Stage 1 detects damage. |

There is also support for `digiLifeDoc_damage_model.onnx` as a fallback binary damage-model filename, and `damage_detector.onnx` as a fallback YOLO filename.
You may also see `mulkiya_classifier_model.keras` in `models/`; it is a legacy artifact and is not the default filename used by the current loader.

Override the YOLO model path at runtime with `UPSURE_YOLO_MODEL`.

### Expected `models/` contents

```text
models/
|-- best_car_model_v2.keras
|-- card_noncard_classifier_model.keras
|-- mulkiya_classifier_model.keras
|-- damage_model.onnx
`-- damage_detector_v2.onnx
```

If your damage model was exported with external data, the companion `.data` file must live next to the `.onnx` file.

## Environment And Prerequisites

You need:

1. Python 3.10+.
2. The packages listed in `requirements.txt`.
3. A working OCR Python environment for `ocr_simple_test.py`.
4. PyMuPDF for PDF-to-image conversion on the car and damage routes.

### OCR Python override

`poc_api.py` looks for the OCR interpreter in `UPSURE_OCR_PYTHON`.

If that variable is not set, the service tries these defaults:

1. `D:/UpSure/OCR_test/venv/Scripts/python.exe`
2. `../OCR_test/venv/Scripts/python.exe` relative to this repository
3. The current Python interpreter as a last fallback

If OCR fails with an environment error, check that `UPSURE_OCR_PYTHON` points to the correct virtual environment.

## Setup

### 1. Create or activate a virtual environment

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

If you already have a working environment, reuse it.

### 2. Install dependencies

```bat
pip install -r requirements.txt
```

### 3. Download the model files

Place the model artifacts in `models/` before starting the API.

### 4. Set the OCR interpreter if needed

```bat
set UPSURE_OCR_PYTHON=D:\path\to\OCR_test\venv\Scripts\python.exe
```

## Running The Services

### Unified API

This is the recommended entry point for most work:

```bat
.\.venv\Scripts\python.exe -m uvicorn poc_api:app --host 0.0.0.0 --port 8000
```

### Standalone car classifier

Use this for a quick car-only smoke test or backward-compatible workflows:

```bat
.\.venv\Scripts\python.exe -m uvicorn car_classifier_api:app --host 0.0.0.0 --port 8001
```

## API Reference

### `GET /`

Returns a small status message showing that the service is running.

### `GET /health`

Returns:

1. The resolved model paths.
2. Whether the binary damage model is ready (`damage_model_ready`).
3. Any binary damage model error message.
4. Whether the YOLO damage-type model is ready (`yolo_model_ready`).
5. Any YOLO model error message.
6. The OCR script path.

### `POST /predict/`

Direct car classification endpoint.

Input:

1. `file` as `multipart/form-data`.

Behavior:

1. Images are normalized to JPEG before inference.
2. PDFs are rendered to the first page and then classified.
3. The response includes the normalized-artifact metadata.

Example:

```bat
curl.exe -X POST "http://localhost:8000/predict/" ^
  -F "file=@Samples\car_1001.jpg"
```

### `POST /api/v1/process`

Unified processing endpoint.

Required form fields:

1. `file`
2. `process_type` with one of `car`, `mulkiya`, `pdf`, or `file`

Optional form fields:

1. `card_threshold` default `0.5`
2. `ocr_lang` default `ar`
3. `prefer_pdf_text` default `false`
4. `skip_ocr` default `false`
5. `translate_to_en` default `false`

#### `process_type=car`

1. Converts the input to a model-ready image.
2. Runs the car classifier.
3. Returns the car classification in `car_classification`.

#### `process_type=mulkiya`

1. Runs card vs not-card classification on the image form of the input.
2. Runs OCR unless `skip_ocr=true`.
3. If OCR succeeds, returns `raw_ocr`, `extracted_data`, and `rag_chunks`.
4. If the structured Mulkiya JSON is created by the OCR pipeline, that structured artifact is used as the chunk source.

For PDF uploads:

1. The classifier portion is skipped because the code expects OCR for PDF Mulkiya inputs.
2. The response sets the classification label to `unknown` and explains why.

#### `process_type=pdf`

1. Ensures the input is available as PDF.
2. Runs OCR.
3. Returns the OCR lines and chunked JSON output.

If you pass an image, the API wraps it into a one-page PDF before OCR.

#### `process_type=file`

1. Does not run OCR.
2. Returns metadata and a category-specific preview.
3. Useful for inspecting arbitrary files before deciding which pipeline to use.

Supported inspection categories include:

1. Image
2. PDF
3. Text
4. JSON
5. Tabular files like CSV and TSV
6. ZIP-based office formats like DOCX, XLSX, and PPTX
7. XML
8. Generic binary files

#### Example requests

Car:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\car_10.jpg" ^
  -F "process_type=car"
```

Mulkiya:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya"
```

Mulkiya without OCR:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya" ^
  -F "skip_ocr=true"
```

PDF OCR:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\sample_pdf.pdf" ^
  -F "process_type=pdf" ^
  -F "prefer_pdf_text=true"
```

General file inspection:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@README.md" ^
  -F "process_type=file"
```

### `POST /predict/damage`

Damage detection endpoint for vehicle views.

Accepted fields:

1. `front`
2. `back`
3. `left`
4. `right`

You must provide at least one view. The endpoint accepts up to four.

Behavior:

1. Each view is normalized to a JPEG model input.
2. PDFs are rendered to the first page before classification.
3. Stage 1 (EfficientNet-B2 binary model) runs on CPU through ONNX Runtime.
4. `damage_detected=true` if any view is classified as damaged.
5. `overall_confidence` is the highest confidence score among all submitted views.
6. For each view where damage is detected, Stage 2 (YOLOv8n) runs automatically and populates a `damages` list with per-detection detail.
7. If the YOLO model is not present, Stage 2 is skipped silently and `damages` is an empty list.

Each entry in `damages` contains:

1. `type` — one of `car-part-crack`, `deformation`, `flat-tire`, `glass-crack`, `lamp-crack`, `scratches`
2. `confidence` — YOLO detection confidence
3. `severity` — `minor`, `moderate`, or `severe` (derived from bounding box area)
4. `bbox` — normalized bounding box `[cx, cy, w, h]` in `[0, 1]`
5. `parts_at_risk` — list of affected parts based on damage type and image region
6. `repair_action` — recommended action string
7. `replace` — boolean indicating whether replacement is recommended over repair

Example:

```bat
curl.exe -X POST "http://localhost:8000/predict/damage" ^
  -F "front=@Samples\car_1001.jpg" ^
  -F "back=@Samples\car_1003.jpg" ^
  -F "left=@Samples\car_1005.jpg" ^
  -F "right=@Samples\car_1007.jpg"
```

## Response Shape

The unified endpoint returns a structured response with these common top-level fields:

1. `input`
2. `classification`
3. `car_classification`
4. `confidence_score`
5. `extracted_data`
6. `raw_ocr`
7. `rag_chunks`
8. `artifacts`
9. `note`
10. `translation` when `translate_to_en=true`

The `input.normalized` block tells you:

1. Which normalized file kind was used.
2. The normalized MIME type.
3. Whether conversion happened.
4. Extra conversion details such as source format and PDF page selection.

## Internal Data Flow

Here is the simplest way to understand the pipeline:

1. The upload enters `poc_api.py`.
2. The API checks the `process_type`.
3. The file is normalized into the format needed by the next step.
4. The model or OCR script runs.
5. The API loads any generated JSON artifacts.
6. The response is assembled with the original file context, inference result, and helpful metadata.

For OCR-based paths, the API calls `ocr_simple_test.py` in a subprocess and then reads the JSON artifacts it writes next to the normalized input.

## Translation Support

`translate_to_en=true` does not call a remote translation service.

Instead, the API uses a small local dictionary-based helper that:

1. Translates known Mulkiya field names into English.
2. Translates a set of common Arabic values.
3. Leaves unknown text unchanged.

This is useful for quick review, but it is not a full translation engine.

## RAG Chunking

If a pipeline produces structured JSON, the API feeds that JSON into `rag_json_chunker.py`.

The chunker:

1. Flattens structured JSON into readable text lines.
2. Splits OCR page text into overlapping chunks.
3. Returns chunk metadata such as page number and chunk index.

Default chunk settings used by the API:

1. `max_chars=1200`
2. `overlap_lines=3`

## Supporting Scripts

### `card_inference.py`

This is a compact helper around the card vs not-card model.

It supports:

1. CLI inference on a single image path.
2. A Tkinter GUI for manual testing.
3. Model path overrides with `--model-path`.
4. Threshold tuning with `--threshold`.
5. Turning normalization off with `--no-normalize`.

Example:

```bat
.\.venv\Scripts\python.exe card_inference.py --image Samples\Mulkiya_front.jpg
```

### `ocr_simple_test.py`

This script is the OCR workhorse behind the unified API.

The API uses it with:

1. `--write_text`
2. `--no_images`
3. `--lang`
4. `--extract_mulkya` when the Mulkiya pipeline needs structured extraction
5. `--prefer_pdf_text` when PDF text should be preferred

It writes OCR outputs as JSON files next to the input artifact so the API can load them afterward.

### `rag_json_chunker.py`

This can be run on its own to turn OCR or structured JSON files into JSONL chunks.

Example:

```bat
.\.venv\Scripts\python.exe rag_json_chunker.py Samples\ --output rag_chunks.jsonl
```

### `latency_analyzer.py`

This script benchmarks the unified API with latency-focused metrics.

It supports scenarios like:

1. `health`
2. `predict-car`
3. `predict-damage`
4. `process-car`
5. `process-mulkiya`
6. `process-pdf`
7. `standalone-car`

Useful flags:

1. `--sample-format`
2. `--random-sample-format`
3. `--front-file`, `--back-file`, `--left-file`, `--right-file`
4. `--mulkiya-skip-ocr`
5. `--prefer-pdf-text`

Example:

```bat
.\.venv\Scripts\python.exe latency_analyzer.py --scenario all --runs 5
```

### `benchmark_everything.py`

This is a broader benchmark runner that reports more than latency alone.

It can:

1. Benchmark the unified routes.
2. Optionally include the standalone car API.
3. Save JSON and CSV output.
4. Collect response-quality metrics such as confidence and OCR line counts.

Example:

```bat
.\.venv\Scripts\python.exe benchmark_everything.py --save-results --include-standalone
```

## Sample Data

The `Samples/` folder contains files that are useful for testing each pipeline:

1. `Samples/car_10.jpg`, `Samples/car_1001.jpg`, `Samples/car_1003.jpg`, `Samples/car_1005.jpg`, `Samples/car_1007.jpg`
2. `Samples/Mulkiya_front.jpg`, `Samples/Mulkiya_back.jpg`
3. `Samples/Non_card_image_1.jpeg`, `Samples/Non_card_image_2.jpeg`
4. `Samples/sample_pdf.pdf`
5. `Samples/converted/` for alternate image formats used in benchmarking

## Common Troubleshooting

### `415 Unsupported Media Type` or conversion errors

This usually means the uploaded file could not be converted to the format required by the chosen pipeline.

Common causes:

1. Unsupported binary files sent to `car`, `mulkiya`, `pdf`, or `damage` routes.
2. Corrupt image files.
3. Missing PDF conversion dependencies.

### OCR failure

If OCR fails, check:

1. `UPSURE_OCR_PYTHON`
2. The OCR environment itself
3. Whether PaddleOCR and its dependencies are installed in that environment

### Damage model not ready

If `/health` reports `damage_model_ready=false`, check:

1. That `models/damage_model.onnx` exists.
2. That any external data file required by the ONNX export exists next to it.
3. That `UPSURE_DAMAGE_MODEL` is not pointing to the wrong file.

### YOLO damage-type model not ready

If `/health` reports `yolo_model_ready=false`, check:

1. That `models/damage_detector_v2.onnx` exists.
2. That `UPSURE_YOLO_MODEL` is not pointing to the wrong file.

Note: the YOLO model missing is non-fatal. The `/predict/damage` endpoint still runs Stage 1 and returns binary results. The `damages` field will be an empty list for all views.

### Car model load failure

If car inference fails, confirm:

1. `models/best_car_model_v2.keras` exists.
2. TensorFlow or Keras is installed.
3. The file is not corrupted.

## Important Notes

1. The unified API uses `process_type` to choose the workflow.
2. `process_type=mulkiya` supports `skip_ocr=true`.
3. `process_type=file` is intended for inspection, not inference.
4. The API adds an `X-Process-Time-ms` response header on every request.
5. Model artifacts are expected to be downloaded locally and are not meant to be committed.

## Quick Start

If you just want the shortest path to a working setup:

1. Create and activate `.venv`.
2. Run `pip install -r requirements.txt`.
3. Download the model files into `models/`.
4. Set `UPSURE_OCR_PYTHON` if your OCR interpreter lives elsewhere.
5. Start the unified API on port `8000`.
6. Test with one of the sample files in `Samples/`.
