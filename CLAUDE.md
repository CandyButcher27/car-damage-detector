# data-ingestion — Claude Instructions

## Repo Purpose
FastAPI backend for UpsureAI's AI document/image processing pipeline.
GitLab: `https://gitlab-v2.upsure.io/ai-cohort/data-ingestion` (main branch)

## How to Run
```
cd C:\Users\sriva\Documents\Desktop\data-ingestion
.venv\Scripts\activate   # or create: python -m venv .venv
uvicorn poc_api:app --reload --port 8000
```
Health check: `GET http://localhost:8000/health`

## Key Files
- `poc_api.py` — single-file FastAPI app, all endpoints
- `card_inference.py` — card/non-card classifier (used by poc_api.py)
- `rag_json_chunker.py` — JSON chunking for RAG (used by poc_api.py)
- `models/` — all model files (see below)
- `tests/test_backend.py` — pytest unit + integration tests (19 unit, 1 integration)

## Models Directory (`models/`)
| File/Dir | Purpose | Notes |
|----------|---------|-------|
| `damage_model.onnx` | binary car damage detector (EfficientNet-B2) | 31MB self-contained — MUST be single-file export, NOT split |
| `damage_detector_v3.onnx` (or `digiLifeDoc_damage_detector_v3.onnx`) | YOLO damage localizer (YOLO11m, CarDD, 6 classes) | production model; v3 class order/names differ from v2 — remapped to canonical keys in `YOLO_CLASSES`. v2 kept as fallback. |
| `best_car_model_v2.onnx` (or `digiLifeDoc_best_car_model_v2.onnx`) | car binary classifier | ONNX preferred (see `onnx_inference.BinaryOnnxImageClassifier`); `.keras` is auto-detected as fallback |
| `digiLifeDoc_card_noncard_classifier_model.onnx` | card/non-card classifier | ONNX via `BinaryOnnxImageClassifier`; used in mulkiya path |
| `digiLifeDoc_mulkiya_classifier_model.onnx` | mulkiya front/back side classifier | ONNX via `BinaryOnnxImageClassifier`; `UPSURE_MULKIYA_FRONT_HIGH` flips polarity |

**CRITICAL:** `damage_model.onnx` must be a self-contained single-file ONNX export (~31MB).
A split export (~736KB + `.data` companion) will fail with:
`"ONNX Runtime expects a companion external-data file next to damage_model.onnx: damage_model.onnx.data"`

## Endpoints
- `GET /livez` — k8s liveness (always 200 if alive)
- `GET /readyz` — k8s readiness (200 only when critical models loaded)
- `GET /health` — full component snapshot (legacy)
- `GET /metrics` — Prometheus scrape
- `POST /predict/` — direct car classification
- `POST /predict/damage` — multipart, fields `front` / `back` / `left` / `right` (UploadFile, optional, ≥1 required)
- `POST /api/v1/process` — unified processing, `process_type` ∈ `car` / `mulkiya` / `pdf` / `file`

All responses use the envelope `{ success, data, error, meta }`. Tests in
`tests/test_backend.py` use the `ok()` / `err()` helpers to dig in.

## Damage Pipeline (`/predict/damage`)
Batched damage inference over submitted views via `run_in_threadpool`.

```
Phase 1 (async sequential): read all upload bytes, decode to JPEG, car-gate each view
Phase 2 (batched ONNX call over surviving car views):
  per view (front/back/left/right):
    binary model (damage_model.onnx)
      → if prob_damaged > 0.25: YOLO (damage_detector_v3.onnx)
      → severity from bbox area: <5% minor, 5-15% moderate, >15% severe
      → parts + repair from rule tables
      → fallback: YOLO empty but binary positive → "general-damage" entry

overall_confidence = max(confidence_score) across DAMAGED views only
```

Response:
```json
{
  "damage_detected": true,
  "overall_confidence": 0.82,
  "total_views_analyzed": 4,
  "per_view": { "front": {...}, "back": {...}, ... },
  "any_view_error": false,
  "skipped_views": {}
}
```

Key constants in `poc_api.py`:
```python
DAMAGE_THRESHOLD  = 0.25   # binary model cutoff
YOLO_CONF         = 0.25   # YOLO detection threshold
YOLO_IOU          = 0.45
# v3 model index order, remapped to canonical rule-table keys (v3: dent, scratch, crack, shattered_glass, broken_lamp, flat_tire)
YOLO_CLASSES      = ["deformation", "scratches", "car-part-crack", "glass-crack", "lamp-crack", "flat-tire"]
SEVERITY_MINOR_MAX    = 0.05   # bbox area fraction
SEVERITY_MODERATE_MAX = 0.15
```

## Running Tests
```
cd C:\Users\sriva\Documents\Desktop\data-ingestion
python -m pytest tests\test_backend.py -v
```
Unit tests mock all model calls — run in ~30s without any models loaded.
Integration tests skip automatically if model files are absent.

## OCR Dependency
OCR endpoints invoke a subprocess Python at `UPSURE_OCR_PYTHON` env var, or fallback:
`D:/UpSure/OCR_test/venv/Scripts/python.exe`
If OCR path missing, OCR endpoints fail but damage/card endpoints work fine.

## Environment Variables
| Var | Purpose | Default |
|-----|---------|---------|
| `UPSURE_DAMAGE_MODEL` | override binary model path | `models/damage_model.onnx` |
| `UPSURE_YOLO_MODEL` | override YOLO model path | `models/damage_detector_v3.onnx` |
| `UPSURE_OCR_PYTHON` | path to OCR venv python | `D:/UpSure/OCR_test/venv/Scripts/python.exe` |

## Virtual Environment
`.venv` lives in repo root. All deps in `requirements.txt`.
Key packages: `fastapi`, `uvicorn`, `onnxruntime==1.20.1`, `rapidocr>=3.5.0`, `tensorflow-cpu==2.20.0` (legacy .keras fallback only), `pillow`, `numpy`.
OCR: `rapidocr` v3 (PP-OCR det+rec on onnxruntime). Arabic rec model (`arabic_PP-OCRv4_rec_mobile.onnx`) auto-downloads on first use. No PaddlePaddle.

## Frontend Integration
digi-motor frontend proxies `/damage-api` → `http://localhost:8000` (via `setupProxy.js`).
Both services must be running locally for end-to-end dev testing.
digi-motor repo: `https://gitlab-v2.upsure.io/tameen/digi-motor` — branch `intern`

## Git
- Branch `main` — single active branch, push directly
- No feature branch convention established for this repo
