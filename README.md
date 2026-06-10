# UpSure Combined PoC

FastAPI proof of concept for vehicle, Mulkiya, OCR, file-inspection, damage-analysis, and ANPR workflows.

The main service is `poc_api.py`. It accepts uploads, normalizes each file into the format required by the selected pipeline, runs the appropriate model or OCR process, and returns a structured JSON response. `car_classifier_api.py` is a smaller standalone car-classifier service kept for compatibility and quick smoke tests.

## What The Project Does

1. Classifies whether an image or first PDF page contains a car.
2. Classifies Mulkiya-style images as card or not-card.
3. Runs OCR over images and PDFs.
4. Extracts structured Mulkiya fields from OCR text.
5. Inspects general files without OCR or model inference.
6. Detects vehicle damage across one to four uploaded views.
7. Runs damage localization/type detection only when binary damage is detected.
8. Runs ANPR on the best available vehicle view and reads plate text.
9. Produces RAG-friendly chunks from OCR JSON or structured JSON artifacts.
10. Provides latency and benchmark utilities for the available routes.

## Main Entry Points

| File | Purpose |
|---|---|
| `poc_api.py` | Main FastAPI app. Owns routing, normalization, OCR orchestration, model calls, damage flow, ANPR integration, and response assembly. |
| `car_classifier_api.py` | Standalone car-classifier API using `best_car_model_v2.keras`. |
| `card_inference.py` | Lightweight NumPy/HDF5 loader for the card vs not-card Keras model. |
| `ocr_simple_test.py` | OCR worker script called by `poc_api.py` in a subprocess. Produces OCR JSON and optional Mulkiya JSON. |
| `plate_pipeline.py` | ANPR pipeline: plate detection with YOLOv4 TensorFlow SavedModel and plate OCR with PaddleOCR. |
| `rag_json_chunker.py` | Converts OCR or structured JSON into chunk objects suitable for RAG workflows. |
| `latency_analyzer.py` | Latency-focused benchmark runner. |
| `benchmark_everything.py` | Broader benchmark runner that can save JSON/CSV results. |
| `Samples/` | Example images, PDFs, and converted formats for smoke testing. |
| `models/` | Local model artifacts. Model binaries are expected to be downloaded locally. |

## Project Layout

```text
.
|-- poc_api.py
|-- car_classifier_api.py
|-- card_inference.py
|-- ocr_simple_test.py
|-- plate_pipeline.py
|-- rag_json_chunker.py
|-- latency_analyzer.py
|-- benchmark_everything.py
|-- requirements.txt
|-- README.md
|-- models/
`-- Samples/
```

## Pipeline Diagram

![UpSure pipeline diagram](<Pipeline diagram.png>)

## Model Call Order

| Route or workflow | Model sequence |
|---|---|
| `/predict/` | `best_car_model_v2.keras` |
| `/api/v1/process`, `process_type=car` | `best_car_model_v2.keras` |
| `/api/v1/process`, `process_type=mulkiya`, image input | `card_noncard_classifier_model.keras` -> if `skip_ocr=false`: `PaddleOCR` -> rule-based Mulkiya extraction -> RAG chunking |
| `/api/v1/process`, `process_type=mulkiya`, PDF input | Card classifier is skipped -> if `skip_ocr=false`: PDF text layer or `PaddleOCR` -> rule-based extraction -> RAG chunking |
| `/api/v1/process`, `process_type=pdf` | PDF text layer when `prefer_pdf_text=true` and text exists, otherwise `PaddleOCR` -> RAG chunking |
| `/api/v1/process`, `process_type=file` | No ML model. File inspection only, then JSON chunking. |
| `/predict/damage` | Per submitted view: `damage_model.onnx` -> if damaged and YOLO model exists: `damage_detector_v2.onnx`. In parallel, one selected view runs ANPR: YOLOv4 plate detector -> PaddleOCR plate OCR. |
| `car_classifier_api.py` standalone service | Loads `best_car_model_v2.keras` at startup, then reuses it for `/predict/`. |

Models in `poc_api.py` are lazy-loaded. The first request that needs a model loads it into memory; later requests reuse the cached model/session.

## Model Artifacts

Place model files in `models/` before starting the services.

| Model artifact | Used by |
|---|---|
| `best_car_model_v2.keras` | Car classification in `/predict/`, `/api/v1/process` with `process_type=car`, and the standalone car service. |
| `card_noncard_classifier_model.keras` | Mulkiya card vs not-card classification for image uploads. The loader also accepts the legacy filename `card_noncard_model.keras`. |
| `damage_model.onnx` | Stage 1 binary damage classification for `/predict/damage`. Fallback filename: `digiLifeDoc_damage_model.onnx`. |
| `damage_detector_v2.onnx` | Stage 2 YOLO damage-type/localization model for `/predict/damage`. Fallback filename: `damage_detector.onnx`. |
| `models/anpr_plate_detector/` | TensorFlow SavedModel used by `plate_pipeline.py` for license-plate detection. |
| `mulkiya_classifier_model.keras` | Legacy artifact. It is present in some setups but is not the default model used by the current loader. |

Expected local contents:

```text
models/
|-- best_car_model_v2.keras
|-- card_noncard_classifier_model.keras
|-- damage_model.onnx
|-- damage_detector_v2.onnx
|-- mulkiya_classifier_model.keras
`-- anpr_plate_detector/
```

Useful model environment overrides:

```bat
set UPSURE_DAMAGE_MODEL=D:\path\to\damage_model.onnx
set UPSURE_YOLO_MODEL=D:\path\to\damage_detector_v2.onnx
```

If an ONNX model was exported with external data, keep its companion `.data` file next to the `.onnx` file.

## Environment

Requirements:

1. Python 3.10+.
2. Packages from `requirements.txt`.
3. TensorFlow or Keras for the car model and ANPR SavedModel.
4. ONNX Runtime for damage models.
5. PaddleOCR for OCR and plate reading.
6. PyMuPDF for PDF conversion and optional PDF text extraction.

### OCR Python Override

`poc_api.py` runs `ocr_simple_test.py` as a subprocess. It picks the OCR interpreter in this order:

1. `UPSURE_OCR_PYTHON`, if set.
2. `D:/UpSure/OCR_test/venv/Scripts/python.exe`, if it exists.
3. `../OCR_test/venv/Scripts/python.exe` relative to this repository, if it exists.
4. The current Python interpreter.

Set it explicitly when your OCR dependencies live in a separate environment:

```bat
set UPSURE_OCR_PYTHON=D:\path\to\OCR_test\venv\Scripts\python.exe
```

## Setup

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Then download or copy the required model artifacts into `models/`.

## Run The Services

Unified API, recommended for most testing:

```bat
.\.venv\Scripts\python.exe -m uvicorn poc_api:app --host 0.0.0.0 --port 8000
```

Standalone car classifier:

```bat
.\.venv\Scripts\python.exe -m uvicorn car_classifier_api:app --host 0.0.0.0 --port 8001
```

## API Reference

### `GET /`

Returns a simple service status message.

### `GET /health`

Checks and reports:

1. Resolved model paths.
2. Binary damage-model readiness and errors.
3. YOLO damage-model readiness and errors.
4. ANPR model/readiness and errors.
5. OCR script path.

### `POST /predict/`

Direct car classification endpoint.

Input:

1. Multipart field `file`.

Behavior:

1. Image uploads are normalized to JPEG.
2. PDF uploads are rendered to the first page and then normalized to JPEG.
3. `best_car_model_v2.keras` returns `is_car`, `confidence`, `raw_score`, and `threshold_used`.

Example:

```bat
curl.exe -X POST "http://localhost:8000/predict/" ^
  -F "file=@Samples\car_1001.jpg"
```

### `POST /api/v1/process`

Unified processing endpoint.

Required form fields:

1. `file`
2. `process_type`: one of `car`, `mulkiya`, `pdf`, or `file`

Optional form fields:

| Field | Default | Meaning |
|---|---:|---|
| `card_threshold` | `0.5` | Threshold for Mulkiya image card/not-card classification. |
| `ocr_lang` | `ar` | PaddleOCR language. |
| `prefer_pdf_text` | `false` | For PDFs, use embedded text when available before OCR. |
| `skip_ocr` | `false` | For Mulkiya, stop after classification. |
| `translate_to_en` | `false` | Add local dictionary-based English helper translation. |

#### `process_type=car`

Normalizes the upload for image inference, calls `best_car_model_v2.keras`, and returns the result in `car_classification`.

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\car_10.jpg" ^
  -F "process_type=car"
```

#### `process_type=mulkiya`

For image uploads:

1. Normalize to JPEG.
2. Run `card_noncard_classifier_model.keras`.
3. If `skip_ocr=false`, normalize for OCR and run `ocr_simple_test.py`.
4. Load `<stem>_ocr.json`.
5. Load `<stem>_mulkya.json` when the extractor creates it.
6. Build RAG chunks from the structured Mulkiya JSON, or from OCR lines as fallback.

For PDF uploads, the card classifier is skipped because the current code expects OCR for PDF Mulkiya inputs. The classification label is set to `unknown`.

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya"
```

Skip OCR and return only the card/not-card result:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya" ^
  -F "skip_ocr=true"
```

#### `process_type=pdf`

Ensures the input is a PDF, runs OCR or PDF text extraction, returns flattened lines, and builds RAG chunks.

If an image is uploaded, the API wraps it into a one-page PDF before OCR.

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\sample_pdf.pdf" ^
  -F "process_type=pdf" ^
  -F "prefer_pdf_text=true"
```

#### `process_type=file`

Runs lightweight file inspection only. No OCR and no classifier are called.

Supported inspection categories include image, PDF, text, JSON, CSV/TSV, ZIP-based Office files, XML, and generic binary files.

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@README.md" ^
  -F "process_type=file"
```

### `POST /predict/damage`

Vehicle damage and plate endpoint.

Accepted multipart fields:

1. `front`
2. `back`
3. `left`
4. `right`

At least one view is required. Up to four views can be submitted.

Behavior:

1. Each submitted view is normalized to JPEG. PDFs are rendered to their first page.
2. `damage_model.onnx` runs on every submitted view.
3. If any view is damaged, the top-level `damage_detected` is `true`.
4. For each damaged view, `damage_detector_v2.onnx` runs when available.
5. YOLO outputs are enriched with severity, parts-at-risk, repair action, and replacement recommendation.
6. If the YOLO model is missing or fails, the route still returns a binary damage result and falls back to a generic damage entry when needed.
7. In parallel with damage inference, ANPR runs on the best available view by priority: `front`, `back`, `left`, then `right`.

Damage classes:

1. `car-part-crack`
2. `deformation`
3. `flat-tire`
4. `glass-crack`
5. `lamp-crack`
6. `scratches`

Example:

```bat
curl.exe -X POST "http://localhost:8000/predict/damage" ^
  -F "front=@Samples\car_1001.jpg" ^
  -F "back=@Samples\car_1003.jpg" ^
  -F "left=@Samples\car_1005.jpg" ^
  -F "right=@Samples\car_1007.jpg"
```

## Unified Response Shape

`/api/v1/process` returns these common top-level fields:

| Field | Meaning |
|---|---|
| `input` | Original filename, input kind, and normalization details. |
| `classification` | Card/file/PDF classification data when applicable. |
| `car_classification` | Car model result for car workflows. |
| `confidence_score` | Main confidence value for the selected workflow. |
| `extracted_data` | Structured extracted fields or file/OCR line data. |
| `raw_ocr` | Raw OCR JSON payload when OCR ran. |
| `rag_chunks` | Chunked text built from OCR or structured JSON. |
| `artifacts` | Internal artifact paths such as the chunk source. |
| `note` | Human-readable summary of what ran. |
| `translation` | Present only when `translate_to_en=true`. |

## OCR And Mulkiya Extraction

`poc_api.py` calls `ocr_simple_test.py` with:

1. `--write_text`
2. `--no_images`
3. `--lang <ocr_lang>`
4. `--extract_mulkya` for Mulkiya OCR
5. `--prefer_pdf_text` for PDF OCR when requested

For images, the script runs PaddleOCR directly and writes `<stem>_ocr.json`. For Mulkiya images, it also tries to write `<stem>_mulkya.json` using rule-based extraction. If Arabic OCR misses important fields, the script can run an auxiliary English OCR pass to improve extraction.

For PDFs, `prefer_pdf_text=true` lets the script use the embedded PDF text layer when available. Otherwise, each page is rendered and passed through PaddleOCR.

## Translation Support

`translate_to_en=true` uses a local dictionary helper in `poc_api.py`. It translates known Mulkiya labels and common Arabic values, and leaves unknown text unchanged. It does not call a remote translation API.

## RAG Chunking

Whenever a pipeline produces JSON suitable for chunking, `poc_api.py` calls `chunk_json_file()` from `rag_json_chunker.py`.

Default API chunk settings:

1. `max_chars=1200`
2. `overlap_lines=3`

OCR documents are chunked page-by-page with overlap. Structured JSON is flattened into readable key/value lines and chunked without line overlap.

## Supporting Scripts

### `card_inference.py`

Run the card/non-card model from the CLI:

```bat
.\.venv\Scripts\python.exe card_inference.py --image Samples\Mulkiya_front.jpg
```

Useful options:

1. `--model-path`
2. `--threshold`
3. `--no-normalize`

### `rag_json_chunker.py`

Create JSONL chunks from OCR or structured JSON files:

```bat
.\.venv\Scripts\python.exe rag_json_chunker.py Samples\ --output rag_chunks.jsonl
```

### `latency_analyzer.py`

Benchmark common routes:

```bat
.\.venv\Scripts\python.exe latency_analyzer.py --scenario all --runs 5
```

Supported scenarios include `health`, `predict-car`, `predict-damage`, `process-car`, `process-mulkiya`, `process-pdf`, and `standalone-car`.

### `benchmark_everything.py`

Run broader benchmarks and save output:

```bat
.\.venv\Scripts\python.exe benchmark_everything.py --save-results --include-standalone
```

## Sample Data

Useful sample files:

1. Car images: `Samples/car_10.jpg`, `Samples/car_1001.jpg`, `Samples/car_1003.jpg`, `Samples/car_1005.jpg`, `Samples/car_1007.jpg`
2. Mulkiya images: `Samples/Mulkiya_front.jpg`, `Samples/Mulkiya_back.jpg`
3. Non-card examples: `Samples/Non_card_image_1.jpeg`, `Samples/Non_card_image_2.jpeg`
4. PDF: `Samples/sample_pdf.pdf`
5. Format-conversion samples: `Samples/converted/`

## Troubleshooting

### `415 Unsupported Media Type` or conversion errors

The upload could not be converted into the format needed by the selected pipeline. Check that the file is a valid image/PDF for model or OCR workflows.

### OCR failures

Check:

1. `UPSURE_OCR_PYTHON`
2. PaddleOCR installation in the selected interpreter
3. PyMuPDF installation for PDF inputs
4. Whether the OCR subprocess error in the API response points to a missing package or model

### Damage model not ready

Check:

1. `models/damage_model.onnx` exists, or `UPSURE_DAMAGE_MODEL` points to a valid ONNX file.
2. Any required `.onnx.data` companion file is next to the model.
3. ONNX Runtime is installed.

### YOLO damage model not ready

Check:

1. `models/damage_detector_v2.onnx` exists, or `UPSURE_YOLO_MODEL` points to a valid ONNX file.
2. ONNX Runtime can load the file.

The damage route can still return Stage 1 binary damage results if the YOLO damage model is missing.

### ANPR not available

Check:

1. `plate_pipeline.py` imports successfully.
2. `models/anpr_plate_detector/` exists.
3. TensorFlow and PaddleOCR are installed in the API environment.

### Car model load failure

Check:

1. `models/best_car_model_v2.keras` exists.
2. TensorFlow or Keras is installed.
3. The model file is not corrupted.

## Quick Start

1. Create and activate `.venv`.
2. Run `pip install -r requirements.txt`.
3. Put model artifacts in `models/`.
4. Set `UPSURE_OCR_PYTHON` if OCR uses a separate environment.
5. Start `poc_api.py` on port `8000`.
6. Test with a file from `Samples/`.
