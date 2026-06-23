# data-ingestion — Claude Instructions

## Repo Purpose
FastAPI backend for UpsureAI's AI document/image processing pipeline (client:
Tameen, motor insurance, UAE/KSA). One app ingests an upload, routes it to the
right pipeline (car classify / Mulkiya OCR+extract / generic PDF / file), and
returns a `{ success, data, error, meta }` envelope.
GitLab: `https://gitlab-v2.upsure.io/ai-cohort/data-ingestion` (main branch)

**Hard constraint: CPU only.** Production container is 2–4 vCPU / 4 GB RAM, no
GPU. This drives every model/runtime decision (RapidOCR over PaddlePaddle, no
PyTorch/VLM/Docling in the request path).

## How to Run
```
cd C:\Users\sriva\Documents\Desktop\data-ingestion
.venv\Scripts\activate   # or create: python -m venv .venv
uvicorn poc_api:app --reload --port 8000
```
Health check: `GET http://localhost:8000/health`

## Key Files
- `poc_api.py` — single-file FastAPI app, all endpoints, model singletons, routing
- `ocr_simple_test.py` — OCR worker: RapidOCR engine + Mulkiya field extraction.
  Has a CLI (`main(argv)`) AND is imported in-process by `poc_api.py` (see below)
- `card_crop.py` — card detect / deskew / 4-way orientation (used by the template extractor)
- `card_inference.py` — legacy `.keras` card/non-card loader (fallback path)
- `onnx_inference.py` — `BinaryOnnxImageClassifier` + `YoloOnnxDetector` ONNX wrappers
- `rag_json_chunker.py` — JSON → RAG chunks
- `models/` — all model files (see below)
- `tests/test_backend.py` — pytest suite (~68 unit + integration; integration tests
  skip when model files are absent)

## Models Directory (`models/`)
| File/Dir | Purpose | Notes |
|----------|---------|-------|
| `damage_model.onnx` | binary car damage detector (EfficientNet-B2) | 31MB self-contained — MUST be single-file export, NOT split |
| `damage_detector_v3.onnx` (or `digiLifeDoc_damage_detector_v3.onnx`) | YOLO damage localizer (YOLO11m, CarDD, 6 classes) | production model; v3 class order/names differ from v2 — remapped to canonical keys in `YOLO_CLASSES`. v2 kept as fallback. |
| `best_car_model_v2.onnx` (or `digiLifeDoc_best_car_model_v2.onnx`) | car binary classifier | ONNX preferred (see `onnx_inference.BinaryOnnxImageClassifier`); `.keras` is auto-detected as fallback |
| `digiLifeDoc_card_noncard_classifier_model.onnx` | card/non-card classifier | ONNX via `BinaryOnnxImageClassifier`; used in mulkiya path. ⚠ unreliable on multi-object scenes |
| `digiLifeDoc_mulkiya_classifier_model.onnx` | mulkiya front/back side classifier | ONNX via `BinaryOnnxImageClassifier`. ⚠ polarity is inverted — `UPSURE_MULKIYA_FRONT_HIGH` now defaults to **False** to compensate (validated on the front/back samples). |

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

`/api/v1/process` form params: `process_type`, `card_threshold` (0.5),
`ocr_lang` (**default `en`**), `prefer_pdf_text`, `skip_ocr`, `translate_to_en`.

All responses use the envelope `{ success, data, error, meta }`. Tests in
`tests/test_backend.py` use the `ok()` / `err()` helpers to dig in.

## Mulkiya Pipeline (`process_type=mulkiya`)
```
upload (image)
→ card/non-card ONNX            → reject "not card"  (recapture, no OCR)
→ mulkiya front/back ONNX       → reject "back"      (recapture, no OCR)   [ALWAYS ON]
→ [skip_ocr] return classification only
→ OCR (in-process) + Mulkiya extraction (English only)
→ anchor gate + quality gate    → document_type / is_mulkiya / accepted / recapture
→ envelope
```

Both ONNX gates run **before** OCR, so a reject costs no OCR. The front/back gate
is hard-on (only the FRONT carries the fields). PDFs skip both gates (no
image-model classification) and get raw OCR lines, not structured extraction.

After OCR, `poc_api.py` surfaces `classification.document_type`,
`classification.is_mulkiya`, `classification.accepted`,
`classification.recapture_required`, `classification.recapture_message`.

### Mulkiya field extraction (`ocr_simple_test.py`, English-only)
1. **Full-image OCR** (one pass) → flat keyword extractor for a first read.
2. **Range extractor** (`_extract_by_range_with_boxes`) — reuses the OCR boxes
   (no extra OCR pass, ~free), classifies numerics by value range + position,
   splits merged weight cells (`5201060` → 520 + 1060). This is the primary
   numeric path.
3. **Positional-template extractor** (`card_crop` deskew/orient + per-cell
   re-OCR) — AUTHORITATIVE but EXPENSIVE (~6s). **Gated**: runs only when
   `_numeric_fields_complete(data)` is False (missing / out-of-range field —
   i.e. rotated / multi-doc / noisy frames). Clean upright cards skip it.
4. **Anchor gate** (`_detect_document_type`) + **vehicle-spec fingerprint** —
   rejects non-Mulkiya docs (licence / ID / receipt). Spec fields
   (engine_cc / weights) don't exist on a licence, so their presence ⇒ Mulkiya.
5. **Quality gate** (`_assess_extraction_quality`) — outcome-based (count of
   valid critical fields), NOT pixel-blur (blur proved unreliable). Drives the
   `accepted` / re-capture decision.

**No Arabic OCR.** The Arabic recogniser pass was removed — it only filled the
Arabic text fields (`make`/`model`/`color`/`vehicle_type`), which the product
does not need. Those fields are now always `null`. English reads all numerics
(plate/VIN/cc/weights/year/dates/seats) and the anchor gate works off the
English anchors + the spec fingerprint.

**Known trade-off:** with the template gated off on clean cards, the range
extractor can mis-bind `seats` (it may read the axle count). The template fixes
this but only fires when the cheap read looks incomplete.

## OCR runtime (in-process, preloaded)
OCR runs **in-process** inside the API worker, reusing a **preloaded singleton**
RapidOCR engine (`ocr_simple_test._get_engine()` / `preload()`), warmed at
startup via the lifespan preload loop. `poc_api._run_ocr_script` /
`_run_make_crop` call `ocr_simple_test.main(argv)` directly (guarded by the OCR
circuit breaker + bulkhead).

There is **no OCR subprocess** and no `UPSURE_OCR_PYTHON` — the old subprocess
existed only to isolate PaddlePaddle crashes, and RapidOCR is onnxruntime-only.
Eliminating the per-request process spawn + model reload (~4s) plus gating the
template pass cut warm Mulkiya latency from ~14s to ~4s.

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
A `policy_decision` (`DENY` / `GRANT_WITH_WARNING` / `GRANT` / `NOT_A_CAR`) is
added from a class+severity matrix (`_policy_decision`).

Response:
```json
{
  "damage_detected": true,
  "overall_confidence": 0.82,
  "total_views_analyzed": 4,
  "per_view": { "front": {...}, "back": {...}, ... },
  "any_view_error": false,
  "skipped_views": {},
  "policy_decision": "GRANT_WITH_WARNING"
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
(Note: `test_policy_general_damage_denies_only_when_severe` is a known
pre-existing failure in the damage-policy matrix, unrelated to OCR.)

## Environment Variables
| Var | Purpose | Default |
|-----|---------|---------|
| `UPSURE_DAMAGE_MODEL` | override binary damage model path | `models/damage_model.onnx` |
| `UPSURE_YOLO_MODEL` | override YOLO damage model path | `models/damage_detector_v3.onnx` |
| `UPSURE_MULKIYA_FRONT_HIGH` | front/back polarity: `True` ⇒ high model output = front | **`False`** (model polarity is inverted) |
| `UPSURE_TEMPLATE_EXTRACTOR` | enable the positional-template extractor | `1` (on) |
| `UPSURE_RANGE_EXTRACTOR` | enable the range numeric extractor | `1` (on) |

(Removed: `UPSURE_OCR_PYTHON` — OCR is in-process now. `UPSURE_MULKIYA_SIDE_GATE`
— the front/back gate is always on. `UPSURE_SPATIAL_NUMERIC` — the spatial
extractor was deleted.)

## Virtual Environment
`.venv` lives in repo root (Python 3.11). All deps in `requirements.txt`.
Key packages: `fastapi`, `uvicorn`, `onnxruntime==1.20.1`, `rapidocr>=3.5.0`,
`tensorflow-cpu==2.20.0` (legacy `.keras` fallback loader only), `pillow`, `numpy`,
`PyMuPDF`, `opencv-python-headless`.
OCR: `rapidocr` v3 (PP-OCR det+rec on onnxruntime). **No PaddlePaddle. No Arabic
recogniser** (the Arabic pass was removed). `pytest` + `httpx` are test-only
(installed separately, not in `requirements.txt`).

## Anti-Patterns / Constraints
- Do NOT reinstall PaddleOCR/PaddlePaddle — it broke production; RapidOCR replaced it.
- Do NOT re-add the OCR subprocess or `UPSURE_OCR_PYTHON` — OCR is in-process with a preloaded engine.
- Do NOT set `ocr_lang="ar"` as default — verified worse on numerics; English-only is the path.
- Do NOT trust fill-rate as accuracy — high fill ≠ correct (verify values).
- Do NOT use GPU-only / PyTorch / VLM / Docling models in the request path — CPU, 4 GB.
- Do NOT use HuggingFace for bounding-box / CV datasets — use Roboflow.

## Frontend Integration
digi-motor frontend proxies `/damage-api` → `http://localhost:8000` (via `setupProxy.js`).
Both services must be running locally for end-to-end dev testing.
digi-motor repo: `https://gitlab-v2.upsure.io/tameen/digi-motor` — branch `intern`

## Git
- Branch `main` — single active branch, push directly.
- Confirm before commit / push (per global rule).
