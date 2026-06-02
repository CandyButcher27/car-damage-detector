# Data Ingestion Pipeline

This repository packages the document, card, car, OCR, Mulkiya extraction, and general file inspection flows behind a single FastAPI service. The unified API routes each upload by the explicit `process_type` passed in the API call.

## Included Files

* `poc_api.py` - main unified FastAPI entry point for explicit document, card, and car dispatch.
* `car_classifier_api.py` - standalone car classifier API kept for backward compatibility.
* `card_inference.py` - local card/non-card model loader and inference helpers.
* `rag_json_chunker.py` - JSON parsing and overlapping chunk creation for downstream RAG usage.
* `ocr_simple_test.py` - OCR and Mulkiya extraction helper script.
* `requirements.txt` - Python dependency list.
* `Samples/` - local test assets for API validation.

## Model Artifacts

The model files are stored in the bucket below.

* [best_car_model.keras](https://storage.googleapis.com/owmdev/digiLifeDoc/best_car_model.keras)
* [mulkiya_classifier_model.keras](https://storage.googleapis.com/owmdev/digiLifeDoc/mulkiya_classifier_model.keras)
* [card_noncard_classifier_model.keras](https://storage.googleapis.com/owmdev/digiLifeDoc/card_noncard_classifier_model.keras)

## Sample Files

The repository includes sample files for quick API testing:

* `Samples/car_10.jpg`, `Samples/car_1001.jpg`, `Samples/car_1003.jpg`, `Samples/car_1005.jpg`, `Samples/car_1007.jpg`
* `Samples/Mulkiya_front.jpg`, `Samples/Mulkiya_back.jpg`
* `Samples/Non_card_image_1.jpeg`, `Samples/Non_card_image_2.jpeg`
* `Samples/sample_pdf.pdf`

## Prerequisites

* Python environment with the project dependencies installed.
* PaddleOCR and its runtime dependencies for OCR and Mulkiya extraction.


## Setup

### 1. Create or reuse a virtual environment

You can reuse the existing environment if it is already configured:

`D:\UpSure\PoC\.venv`

If you need to create a new one:

```bash
python -m venv .venv
```

Activate it with one of the following:

* Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
* Windows CMD: `.\.venv\Scripts\activate.bat`
* Linux/macOS: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the APIs

### Unified API

Recommended for end-to-end testing:

```bash
D:\UpSure\PoC\.venv\Scripts\python.exe -m uvicorn poc_api:app --host 0.0.0.0 --port 8000
```

### Standalone car classifier

Use this if you want to test the car model by itself:

```bash
D:\UpSure\PoC\.venv\Scripts\python.exe -m uvicorn car_classifier_api:app --host 0.0.0.0 --port 8001
```

## API Routes

Unified API on port 8000:

* `GET /` - basic service check.
* `GET /health` - model and script path status.
* `POST /predict/` - direct car inference.
* `POST /api/v1/process` - unified endpoint. Pass `process_type=car`, `process_type=mulkiya`, `process_type=pdf`, or `process_type=file`.

Standalone car API on port 8001:

* `GET /`
* `POST /predict/`

## End-to-End curl Tests

Run these from the repository root after starting the unified API.

### 1. Check the service is up

```bat
curl.exe http://localhost:8000/
curl.exe http://localhost:8000/health
```

### 2. Test car image dispatch

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\car_10.jpg" ^
  -F "process_type=car"
```

### 3. Test Mulkiya card dispatch

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya"
```

To classify Mulkiya without OCR:

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "process_type=mulkiya" ^
  -F "skip_ocr=true"
```

### 4. Test PDF/document dispatch

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\sample_pdf.pdf" ^
  -F "process_type=pdf" ^
  -F "prefer_pdf_text=true"
```

### 5. Test direct car classification

```bat
curl.exe -X POST "http://localhost:8000/predict/" ^
  -F "file=@Samples\car_1001.jpg"
```

### 6. Test general file inspection

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@README.md" ^
  -F "process_type=file"
```

### 7. Test the standalone car API on port 8001

```bat
curl.exe http://localhost:8001/
curl.exe -X POST "http://localhost:8001/predict/" ^
  -F "file=@Samples\car_1003.jpg"
```

## Latency Benchmarking

Use `latency_analyzer.py` to benchmark the explicit API dispatch paths:

```bat
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-car --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-mulkiya --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-mulkiya --mulkiya-skip-ocr --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-pdf --prefer-pdf-text --runs 10
```

You can pass `--scenario all` or omit `--scenario` to run every endpoint, and override samples with `--car-file`, `--mulkiya-file`, and `--pdf-file`.

## Notes

* The unified pipeline uses the `process_type` API field to select the corresponding inference path.
* `process_type=mulkiya` now supports `skip_ocr=true` for classifier-only checks.
* `process_type=file` accepts general file types and returns metadata plus lightweight previews where possible.
* OCR and Mulkiya extraction depend on the external OCR Python environment.
