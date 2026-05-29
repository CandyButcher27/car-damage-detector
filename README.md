# Data Ingestion Pipeline

This repository packages the document, card, car, OCR, and Mulkiya extraction flows behind a single FastAPI service. The unified API routes each upload to the appropriate pipeline automatically.

## Included Files

* `poc_api.py` - main unified FastAPI entry point for document, card, and car routing.
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
* Optional: `GEMINI_API_KEY` if you want to use Gemini for file routing.


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

### 2. Configure optional API credentials

If you want Gemini-based routing, set `GEMINI_API_KEY`:

* Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

* Windows CMD:

```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

* Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

If no key is provided, the API falls back to local heuristics and the bundled models.

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
* `POST /api/v1/process` - unified routing endpoint for PDFs and images.

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

### 2. Test car image routing

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\car_10.jpg" ^
  -F "llm_backend=heuristic"
```

### 3. Test Mulkiya card routing

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\Mulkiya_front.jpg" ^
  -F "llm_backend=heuristic"
```

### 4. Test PDF/document routing

```bat
curl.exe -X POST "http://localhost:8000/api/v1/process" ^
  -F "file=@Samples\sample_pdf.pdf" ^
  -F "llm_backend=heuristic" ^
  -F "prefer_pdf_text=true"
```

### 5. Test direct car classification

```bat
curl.exe -X POST "http://localhost:8000/predict/" ^
  -F "file=@Samples\car_1001.jpg"
```

### 6. Test the standalone car API on port 8001

```bat
curl.exe http://localhost:8001/
curl.exe -X POST "http://localhost:8001/predict/" ^
  -F "file=@Samples\car_1003.jpg"
```

## Notes

* The unified pipeline can use Gemini, Ollama, or local heuristics for routing.
* OCR and Mulkiya extraction depend on the external OCR Python environment.
