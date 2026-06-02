# Data Ingestion Pipeline

This repository packages the document, card, car, OCR, Mulkiya extraction, general file inspection, and **car damage detection** flows behind a single FastAPI service. The unified API routes each upload by the explicit `process_type` passed in the API call.

## Included Files

* `poc_api.py` - main unified FastAPI entry point for document, card, car, and damage dispatch.
* `car_classifier_api.py` - standalone car classifier API kept for backward compatibility.
* `card_inference.py` - local card/non-card model loader and inference helpers.
* `rag_json_chunker.py` - JSON parsing and overlapping chunk creation for downstream RAG usage.
* `ocr_simple_test.py` - OCR and Mulkiya extraction helper script.
* `latency_analyzer.py` - latency benchmarking across all API endpoints.
* `requirements.txt` - Python dependency list.
* `Samples/` - local test assets for API validation.

## Model Artifacts

Download all model files and place them in the `models/` directory before starting the server.

| Model | Bucket URL | Used by |
|---|---|---|
| `best_car_model.keras` | [download](https://storage.googleapis.com/owmdev/digiLifeDoc/best_car_model.keras) | Car detection (`is_car`) |
| `mulkiya_classifier_model.keras` | [download](https://storage.googleapis.com/owmdev/digiLifeDoc/mulkiya_classifier_model.keras) | Mulkiya classification |
| `card_noncard_classifier_model.keras` | [download](https://storage.googleapis.com/owmdev/digiLifeDoc/card_noncard_classifier_model.keras) | Card/non-card classification |
| `damage_model.onnx` | [download](https://storage.googleapis.com/owmdev/digiLifeDoc/damage_model.onnx) | Car damage detection |

After downloading:
```
models/
├── best_car_model.keras
├── mulkiya_classifier_model.keras
├── card_noncard_classifier_model.keras
└── damage_model.onnx
```

## Sample Files

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

### 2. Download models

Download all four model files from the bucket (see Model Artifacts above) and place them in `models/`.

## Run the APIs

### Unified API

Recommended for end-to-end testing:

```bash
D:\UpSure\PoC\.venv\Scripts\python.exe -m uvicorn poc_api:app --host 0.0.0.0 --port 8000
```

### Standalone car classifier

Use this if you want to test the car detection model by itself:

```bash
D:\UpSure\PoC\.venv\Scripts\python.exe -m uvicorn car_classifier_api:app --host 0.0.0.0 --port 8001
```

## API Routes

Unified API on port 8000:

* `GET /` - basic service check.
* `GET /health` - model and script path status.
* `POST /predict/` - direct car detection inference (single image).
* `POST /predict/damage` - **car damage detection** (1–4 view images). See below.
* `POST /api/v1/process` - unified endpoint. Pass `process_type=car`, `process_type=mulkiya`, `process_type=pdf`, or `process_type=file`.

Standalone car API on port 8001:

* `GET /`
* `POST /predict/`

## Car Damage Detection — `/predict/damage`

This endpoint runs the EfficientNet-B2 damage detection model (ONNX, CPU-only) on 1–4 vehicle view images.

**Intended workflow:**
1. Each vehicle image is first sent to `/api/v1/process?process_type=car` to confirm `is_car=true`.
2. All confirmed-car images are then sent together to `/predict/damage`.
3. `damage_detected=true` if ANY view shows damage.

**Request:** `multipart/form-data` with optional fields `front`, `back`, `left`, `right` (min 1 required).

**Response:**
```json
{
  "damage_detected": true,
  "total_views_analyzed": 4,
  "overall_confidence": 0.94,
  "per_view": {
    "front": {"damage_detected": true,  "confidence_score": 0.94, "prob_damaged": 0.94, "prob_clean": 0.06},
    "back":  {"damage_detected": false, "confidence_score": 0.81, "prob_damaged": 0.19, "prob_clean": 0.81},
    "left":  {"damage_detected": true,  "confidence_score": 0.88, "prob_damaged": 0.88, "prob_clean": 0.12},
    "right": {"damage_detected": false, "confidence_score": 0.76, "prob_damaged": 0.24, "prob_clean": 0.76}
  }
}
```

**Model details:**
- Architecture: EfficientNet-B2 (9M params), fine-tuned via knowledge distillation from a 235B-param VLM
- Input: 260×260 RGB, ImageNet normalization
- Output: binary (damaged / clean), threshold 0.25
- Evaluated at car level: 95% accuracy, 100% recall, 90.9% precision (100 cars, 4-view test)

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

### 8. Test car damage detection (all 4 views)

```bat
curl.exe -X POST "http://localhost:8000/predict/damage" ^
  -F "front=@Samples\car_1001.jpg" ^
  -F "back=@Samples\car_1003.jpg" ^
  -F "left=@Samples\car_1005.jpg" ^
  -F "right=@Samples\car_1007.jpg"
```

Single view also valid:

```bat
curl.exe -X POST "http://localhost:8000/predict/damage" ^
  -F "front=@Samples\car_1001.jpg"
```

## Latency Benchmarking

Use `latency_analyzer.py` to benchmark all API dispatch paths:

```bat
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-car --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-mulkiya --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-mulkiya --mulkiya-skip-ocr --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario process-pdf --prefer-pdf-text --runs 10
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario predict-damage --runs 10
```

With real view images for damage benchmark:

```bat
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario predict-damage --runs 10 ^
  --front-file Samples\car_1001.jpg ^
  --back-file  Samples\car_1003.jpg ^
  --left-file  Samples\car_1005.jpg ^
  --right-file Samples\car_1007.jpg
```

Run all scenarios:

```bat
D:\UpSure\PoC\.venv\Scripts\python.exe latency_analyzer.py --scenario all --runs 5
```

You can pass `--scenario all` or omit `--scenario` to run every endpoint. Override sample files with `--car-file`, `--mulkiya-file`, `--pdf-file`, `--front-file`, `--back-file`, `--left-file`, `--right-file`.

## Notes

* The unified pipeline uses the `process_type` API field to select the corresponding inference path.
* `process_type=mulkiya` supports `skip_ocr=true` for classifier-only checks.
* `process_type=file` accepts general file types and returns metadata plus lightweight previews where possible.
* OCR and Mulkiya extraction depend on the external OCR Python environment.
* `/predict/damage` requires `models/damage_model.onnx` — download from bucket before starting the server.
* Model files are gitignored (`models/*`) — always download from the bucket, never commit them.
