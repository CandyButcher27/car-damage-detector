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
- `plate_pipeline.py` — ANPR (license plate detection + OCR); imported by poc_api.py
- `card_inference.py` — card/non-card classifier (used by poc_api.py)
- `rag_json_chunker.py` — JSON chunking for RAG (used by poc_api.py)
- `models/` — all model files (see below)
- `tests/test_backend.py` — pytest unit + integration tests (19 unit, 1 integration)

## Models Directory (`models/`)
| File/Dir | Purpose | Notes |
|----------|---------|-------|
| `damage_model.onnx` | binary car damage detector (EfficientNet-B2) | 31MB self-contained — MUST be single-file export, NOT split |
| `damage_detector_v2.onnx` | YOLO damage localizer (6 classes) | production model, mAP50=0.672 |
| `best_car_model_v2.onnx` (or `digiLifeDoc_best_car_model_v2.onnx`) | car binary classifier | ONNX preferred (see `onnx_inference.BinaryOnnxImageClassifier`); `.keras` is auto-detected as fallback |
| `card_noncard_classifier_model.onnx` (or `digiLifeDoc_card_noncard_classifier_model.onnx`) | card/non-card classifier | ONNX-first via `onnx_inference.BinaryOnnxImageClassifier`; `.keras` (`card_inference.CardNonCardModel`) is auto-detected fallback |
| `mulkiya_classifier_model.onnx` (or `digiLifeDoc_mulkiya_classifier_model.onnx`) | Mulkiya / non-Mulkiya classifier | ONNX single-sigmoid; runs only after the card gate passes |
| `anpr_plate_detector/` | YOLOv4 TF SavedModel for license plate detection | SavedModel dir: saved_model.pb + variables/ |

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

### Mulkiya two-stage pipeline (`process_type=mulkiya`, image input)
```
Stage 1 — card / non-card gate (card_noncard ONNX, card_threshold)
   ├─ label != "card"  → short-circuit: classification.label="not card",
   │                      Mulkiya classifier + OCR skipped (note explains why)
   └─ label == "card"  → Stage 2
Stage 2 — Mulkiya classifier (mulkiya ONNX, mulkiya_threshold)
           → classification.{label,probability,stage,card_classification}
           → then OCR extraction (unchanged), unless skip_ocr
```
Form fields: `card_threshold` (default `UPSURE_CARD_THRESHOLD`), `mulkiya_threshold`
(default `UPSURE_MULKIYA_THRESHOLD`). PDF Mulkiya inputs skip the image classifiers
(OCR-only path, unchanged). Both classifiers are `BinaryOnnxImageClassifier`;
card falls back to the `.keras` `CardNonCardModel` when no ONNX is present.
`/readyz` now tracks a `mulkiya` component alongside `card_noncard`.

## Damage + ANPR Pipeline (`/predict/damage`)
Damage and ANPR run **in parallel** using `asyncio.gather` + `run_in_threadpool`.

```
Phase 1 (async sequential): read all upload bytes, decode to JPEG
Phase 2 (parallel via thread pool):
  ├─ per view (front/back/left/right):
  │    binary model (damage_model.onnx)
  │      → if prob_damaged > 0.25: YOLO (damage_detector_v2.onnx)
  │      → severity from bbox area: <5% minor, 5-15% moderate, >15% severe
  │      → parts + repair from rule tables
  │      → fallback: YOLO empty but binary positive → "general-damage" entry
  └─ ANPR (plate_pipeline.py, view priority: front > back > left > right):
       YOLOv4 TF SavedModel → detect plate bbox → PaddleOCR → plate text

overall_confidence = max(confidence_score) across DAMAGED views only
```

Response adds `plate` key:
```json
{
  "damage_detected": true,
  "overall_confidence": 0.82,
  "total_views_analyzed": 4,
  "per_view": { "front": {...}, "back": {...}, ... },
  "plate": {
    "detected": true,
    "plate_text": "12 AB 345",
    "confidence": 0.91,
    "num_plates": 1,
    "source_view": "front"
  }
}
```
If `plate_pipeline` not importable (e.g. missing `paddleocr`): `plate.detected = false`, `plate.error` set.

Key constants in `poc_api.py`:
```python
DAMAGE_THRESHOLD  = 0.25   # binary model cutoff
YOLO_CONF         = 0.25   # YOLO detection threshold
YOLO_IOU          = 0.45
YOLO_CLASSES      = ["car-part-crack", "deformation", "flat-tire", "glass-crack", "lamp-crack", "scratches"]
SEVERITY_MINOR_MAX    = 0.05   # bbox area fraction
SEVERITY_MODERATE_MAX = 0.15
ANPR_VIEW_PRIORITY    = ["front", "back", "left", "right"]
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
| `UPSURE_YOLO_MODEL` | override YOLO model path | `models/damage_detector_v2.onnx` |
| `UPSURE_CARD_MODEL` | override card/non-card model path | auto-detect ONNX→keras |
| `UPSURE_MULKIYA_MODEL` | override Mulkiya model path | auto-detect ONNX |
| `UPSURE_CARD_THRESHOLD` | card-gate sigmoid cutoff | `0.50` |
| `UPSURE_MULKIYA_THRESHOLD` | Mulkiya sigmoid cutoff | `0.50` |
| `UPSURE_CARD_MODEL_POSITIVE_HIGH` | high output = card | `true` |
| `UPSURE_MULKIYA_MODEL_POSITIVE_HIGH` | high output = mulkiya | `true` |
| `UPSURE_OCR_PYTHON` | path to OCR venv python | `D:/UpSure/OCR_test/venv/Scripts/python.exe` |

## Virtual Environment
`.venv` lives in repo root. All deps in `requirements.txt`.
Key packages: `fastapi`, `uvicorn`, `onnxruntime==1.20.1`, `keras==3.12.2`, `tensorflow==2.20.0`, `pillow`, `numpy`.

## Frontend Integration
digi-motor frontend proxies `/damage-api` → `http://localhost:8000` (via `setupProxy.js`).
Both services must be running locally for end-to-end dev testing.
digi-motor repo: `https://gitlab-v2.upsure.io/tameen/digi-motor` — branch `intern`

## Git
- Branch `main` — single active branch, push directly
- No feature branch convention established for this repo
