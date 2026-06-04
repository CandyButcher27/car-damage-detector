# UpSure Data Ingestion — Testbed Checklist

**Project:** UpSure PoC — Document & Vehicle Image Processing Pipeline  
**Version:** 1.0  
**Date:** 2026-06-02  
**Prepared for:** Engineering Review

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not yet tested |
| `[x]` | Passed |
| `[F]` | Failed — needs fix |
| `[S]` | Skipped — out of scope for this run |

> **Note on model accuracy metrics (precision / recall):** The ML models (car classifier, card/non-card classifier, EfficientNet-B2 damage model) have been evaluated during training. Reported baselines — damage model: 95% accuracy, 100% recall, 90.9% precision on a 100-car 4-view test set — are training-time results. The regression tests in Section 7 below verify that the deployed models match those baselines on known sample data. Full retraining-level evaluation is out of scope for this testbed.

---

## Table of Contents

1. [Environment & Setup](#1-environment--setup)
2. [Smoke Tests](#2-smoke-tests)
3. [Car Classification Pipeline](#3-car-classification-pipeline)
4. [Mulkiya Pipeline](#4-mulkiya-pipeline)
5. [PDF Pipeline](#5-pdf-pipeline)
6. [File Inspection Pipeline](#6-file-inspection-pipeline)
7. [Car Damage Detection](#7-car-damage-detection)
8. [Card / Non-Card Classifier (Unit)](#8-card--non-card-classifier-unit)
9. [OCR Pipeline & Mulkiya Field Extraction](#9-ocr-pipeline--mulkiya-field-extraction)
10. [RAG Chunker](#10-rag-chunker)
11. [File Type Routing Matrix](#11-file-type-routing-matrix)
12. [Input Validation & Security](#12-input-validation--security)
13. [Error Handling & HTTP Status Code Contract](#13-error-handling--http-status-code-contract)
14. [Response Schema Completeness](#14-response-schema-completeness)
15. [Latency Benchmarks](#15-latency-benchmarks)
16. [Concurrency & Load Tests](#16-concurrency--load-tests)
17. [Regression / Golden Tests](#17-regression--golden-tests)
18. [Outer Payload & Integration Tests](#18-outer-payload--integration-tests)
19. [Operational & Deployment Checks](#19-operational--deployment-checks)
20. [Per-Insurer Document Validation](#20-per-insurer-document-validation)
21. [Production Readiness Gates](#21-production-readiness-gates)

---

## 1. Environment & Setup

> Verify the full runtime environment is correctly configured before any tests run.

### 1.1 Model Files

- [ ] `models/best_car_model.keras` present
- [ ] `models/mulkiya_classifier_model.keras` present
- [ ] `models/card_noncard_classifier_model.keras` present
- [ ] `models/damage_model.onnx` present
- [ ] If ONNX model uses external data: `models/damage_model.onnx.data` sidecar present alongside it

### 1.2 Python Environment

- [ ] `.venv` created and `requirements.txt` fully installed without errors
- [ ] Separate OCR venv (`OCR_test/venv`) has PaddleOCR installed and working
- [ ] `ocr_simple_test.py` runs successfully from the OCR venv on a sample image
- [ ] `UPSURE_OCR_PYTHON` env var set to the correct OCR Python executable path
- [ ] If `UPSURE_OCR_PYTHON` is not set, fallback path `D:/UpSure/OCR_test/venv/Scripts/python.exe` resolves correctly

### 1.3 Sample Files

- [ ] `Samples/car_10.jpg` present
- [ ] `Samples/car_1001.jpg`, `car_1003.jpg`, `car_1005.jpg`, `car_1007.jpg` present
- [ ] `Samples/Mulkiya_front.jpg`, `Mulkiya_back.jpg` present
- [ ] `Samples/Non_card_image_1.jpeg`, `Non_card_image_2.jpeg` present
- [ ] `Samples/notcar_images (1).jpg` through `notcar_images (13).jpg` present
- [ ] `Samples/sample_pdf.pdf` present

### 1.4 Service Startup

- [ ] Port 8000 free before starting unified API
- [ ] Port 8001 free before starting standalone API
- [ ] `poc_api.py` starts with no import errors or exceptions in stdout
- [ ] `car_classifier_api.py` starts with no errors on port 8001
- [ ] `GET /health` returns `damage_model_ready: true` with correct file paths for all models

---

## 2. Smoke Tests

> Confirm the service is alive and all endpoints respond before deeper testing.

- [ ] `GET /` on port 8000 → HTTP 200, response message contains "UpSure PoC"
- [ ] `GET /health` on port 8000 → HTTP 200, all model paths and script path present in response
- [ ] `GET /` on port 8001 (standalone car API) → HTTP 200
- [ ] `POST /predict/` with a valid car image → HTTP 200
- [ ] `POST /api/v1/process` with `process_type=car` and valid car image → HTTP 200
- [ ] `POST /api/v1/process` with `process_type=mulkiya` and `skip_ocr=true` → HTTP 200
- [ ] `POST /api/v1/process` with `process_type=pdf` and valid PDF → HTTP 200
- [ ] `POST /api/v1/process` with `process_type=file` and any file → HTTP 200
- [ ] `POST /predict/damage` with at least one image → HTTP 200
- [ ] `X-Process-Time-ms` response header present on every response above

---

## 3. Car Classification Pipeline

> Tests for `/predict/` (direct) and `/api/v1/process?process_type=car`.

### 3.1 Correct Classification

- [ ] `car_10.jpg` → `is_car=true`, `confidence ≥ 0.35`
- [ ] `car_1001.jpg`, `car_1003.jpg`, `car_1005.jpg`, `car_1007.jpg` → all return `is_car=true`
- [ ] `Non_card_image_1.jpeg` → `is_car=false`
- [ ] `Non_card_image_2.jpeg` → `is_car=false`
- [ ] All `notcar_images (1)` through `notcar_images (13)` → `is_car=false`
- [ ] `threshold_used` field in response equals `0.35` exactly
- [ ] `raw_score` and `confidence` are internally consistent with `is_car` decision

### 3.2 Image Format Coverage

- [ ] JPEG (`.jpg`) image → accepted and classified
- [ ] PNG (`.png`) image → accepted and classified
- [ ] BMP (`.bmp`) image → accepted and classified
- [ ] WEBP (`.webp`) image → accepted and classified
- [ ] TIFF (`.tif`) image → accepted and classified
- [ ] Greyscale image → converted to RGB internally, no error
- [ ] Image with EXIF rotation tag → EXIF transpose applied before inference (image not classified sideways)

### 3.3 Edge Cases

- [ ] High-resolution image (4K / 12MP) → processed without crash or timeout
- [ ] Very small image (32×32 pixels) → processed without crash
- [ ] Same car image submitted 10 consecutive times → identical `raw_score` each time (deterministic)

### 3.4 Rejection Cases

- [ ] PDF file sent with `process_type=car` → HTTP 400
- [ ] No `filename` on upload → HTTP 400
- [ ] Non-image content-type (e.g., `text/plain`) sent to `/predict/` → HTTP 400
- [ ] `.docx` / `.xlsx` / any non-image-non-PDF sent with `process_type=car` → HTTP 415

### 3.5 API Variant Parity

- [ ] `/predict/` and `/api/v1/process?process_type=car` return the same `is_car` decision on the same image
- [ ] Standalone API on port 8001 `/predict/` returns the same `is_car` as unified API on port 8000

---

## 4. Mulkiya Pipeline

> Tests for `/api/v1/process?process_type=mulkiya`.

### 4.1 Card Classification

- [ ] `Mulkiya_front.jpg` → `classification.label=card`, `probability > 0.5`
- [ ] `Mulkiya_back.jpg` → `classification.label=card`, `probability > 0.5`
- [ ] `Non_card_image_1.jpeg` → `classification.label=not card`
- [ ] `Non_card_image_2.jpeg` → `classification.label=not card`
- [ ] `classification.threshold` in response matches the `card_threshold` value sent in the request (default `0.5`)
- [ ] `card_threshold=0.7` (raised bar) → borderline card image may flip to `not card`
- [ ] `card_threshold=0.1` (lowered bar) → marginal card image flips to `card`
- [ ] PDF sent with `process_type=mulkiya` → `classification.label=unknown`, no crash

### 4.2 skip_ocr Mode

- [ ] `skip_ocr=true` → `extracted_data=null`, `raw_ocr=null`, classification still returned with probability
- [ ] `skip_ocr=true` → response `note` says "Mulkiya classification executed without OCR"
- [ ] `skip_ocr=false` (default) → OCR subprocess is invoked

### 4.3 OCR Path

- [ ] Mulkiya image with OCR enabled → `_ocr.json` produced, `raw_ocr` populated in response
- [ ] When `_mulkya.json` is produced → `extracted_data` contains structured Mulkiya fields
- [ ] When `_mulkya.json` is NOT produced → `extracted_data` falls back to `{ "lines": [...] }` from raw OCR
- [ ] `ocr_lang=ar` (default) forwarded to OCR subprocess
- [ ] `ocr_lang=en` accepted as parameter without causing a 422 error
- [ ] `prefer_pdf_text=true` forwarded to OCR subprocess for PDF Mulkiya inputs
- [ ] OCR subprocess failure (bad env path) → HTTP 500 with stdout/stderr included in error detail
- [ ] `_ocr.json` not created by OCR subprocess → HTTP 500 "OCR JSON output was not created"
- [ ] `confidence_score` = mean of per-line OCR confidences when `_mulkya.json` is absent

### 4.4 RAG Chunks

- [ ] `rag_chunks` is a non-empty list when structured `extracted_data` is available
- [ ] Each chunk contains: `source_file`, `document_type`, `chunk_id`, `text`, `metadata`
- [ ] No single chunk `text` exceeds 1200 characters
- [ ] Consecutive chunks share 3 overlapping lines (`overlap_lines=3`)
- [ ] `source_file` on each chunk matches the originally uploaded filename

---

## 5. PDF Pipeline

> Tests for `/api/v1/process?process_type=pdf`.

### 5.1 Standard PDF Processing

- [ ] `sample_pdf.pdf` → OCR runs, `extracted_data.lines` is populated
- [ ] `confidence_score` = mean of per-line OCR confidences across all pages
- [ ] Multi-page PDF → lines from all pages present, `page` field increments per page
- [ ] `rag_chunks` generated from OCR output, each chunk ≤ 1200 characters
- [ ] `artifacts.chunk_source` field present in response

### 5.2 PDF Text Layer

- [ ] `prefer_pdf_text=true` → embedded PDF text layer used when available (faster, no OCR)
- [ ] `prefer_pdf_text=false` (default) → OCR used even when embedded text layer exists
- [ ] `prefer_pdf_text=true` on a scanned PDF (no text layer) → falls back to OCR without error

### 5.3 Edge Cases

- [ ] Empty PDF (0-byte content or blank pages) → `lines=[]`, `confidence_score=0.0`, no crash
- [ ] PDF with Arabic text → lines extracted correctly, no encoding error
- [ ] Large PDF (10+ pages) → completes without request timeout (within 120-second default)

### 5.4 Rejection Cases

- [ ] Non-PDF file sent with `process_type=pdf` → HTTP 400
- [ ] Image file (`.jpg`) sent with `process_type=pdf` → HTTP 400

---

## 6. File Inspection Pipeline

> Tests for `/api/v1/process?process_type=file`. Accepts ANY file type — no 415 gate.

### 6.1 File Category Detection

- [ ] Image file (`.jpg`, `.png`) → `category=image`, `suggested_process_type=mulkiya`, image dimensions returned
- [ ] PDF file → `category=pdf`, `suggested_process_type=pdf`
- [ ] `.txt` file → `category=text`, `text_preview` ≤ 4000 chars, `line_count_preview` present
- [ ] `.md` file → `category=text`
- [ ] `.html` file → `category=text`
- [ ] `.yaml` / `.yml` file → `category=text`
- [ ] `.py`, `.js`, `.ts`, `.css` files → `category=text`
- [ ] `.log`, `.ini`, `.cfg` files → `category=text`
- [ ] `.json` file with dict root → `category=json`, `top_level_keys` list present (max 25 keys)
- [ ] `.json` file with array root → `category=json`, `item_count` present
- [ ] `.csv` file → `category=tabular`, `preview_rows` has up to 5 rows
- [ ] `.tsv` file → `category=tabular`, tab delimiter used correctly
- [ ] `.zip` file → `category=archive`, `archive_entries` present (max 30 entries)
- [ ] `.docx` file → `category=archive` (zip-based), archive entries shown
- [ ] `.xlsx` file → `category=archive`
- [ ] `.pptx` file → `category=archive`
- [ ] `.xml` file → `category=xml`, `root_tag` present
- [ ] Unknown binary (e.g., `.exe`, `.bin`) → `category=binary`, no crash
- [ ] File with **no extension** → `category=binary`, `extension=null`, no crash

### 6.2 Metadata

- [ ] `confidence_score` always equals `1.0` for file inspection
- [ ] `classification.suggested_process_type` correctly maps to detected category
- [ ] `size_bytes` present and accurate for all file types

### 6.3 Error Handling in File Inspection

- [ ] Malformed JSON file → `json_error` field in response, no crash
- [ ] Corrupt ZIP / DOCX → `archive_error` field in response, no crash
- [ ] UTF-16 encoded text file → UTF-8 decode failure handled gracefully via `errors=replace`
- [ ] Zero-byte file → `size_bytes=0`, no crash

### 6.4 Two-Step Routing (License ID Pattern)

> Based on real production payload: License ID is processed as `file` first, then re-routed.

- [ ] License ID image processed with `process_type=file` → `suggested_process_type=mulkiya` returned
- [ ] Subsequent call with the same image using `process_type=mulkiya` → card classification runs correctly
- [ ] Both calls return valid, non-null responses (two-step flow completes end-to-end)

---

## 7. Car Damage Detection

> Tests for `POST /predict/damage`.

### 7.1 Multi-View Submission

- [ ] All 4 views submitted → `total_views_analyzed=4`, all views present in `per_view`
- [ ] Front view only → `total_views_analyzed=1`, accepted without error
- [ ] 2-view submission → `total_views_analyzed=2`, accepted
- [ ] 3-view submission → `total_views_analyzed=3`, accepted
- [ ] 0 views (empty POST) → HTTP 400 "Provide at least one image"

### 7.2 Damage Logic

- [ ] `damage_detected=true` if ANY single view is damaged (OR across views)
- [ ] `damage_detected=false` only when ALL views are clean
- [ ] `overall_confidence` = max of all per-view `confidence_score` values
- [ ] `per_view.{name}.prob_damaged + per_view.{name}.prob_clean ≈ 1.0` (softmax sum)
- [ ] Per-view `damage_detected=true` when `prob_damaged ≥ 0.25` (threshold = 0.25)
- [ ] Per-view `damage_detected=false` when `prob_damaged < 0.25`

### 7.3 Known Sample Results

- [ ] Known clean car images → `damage_detected=false` for those views
- [ ] Known damaged car images → `damage_detected=true` for those views
- [ ] Same 4-view set returns identical `damage_detected` across 10 repeated runs (deterministic)

### 7.4 Duplicate Views

- [ ] Same image bytes submitted as 2 or more views → accepted, per-view results all identical, no crash
- [ ] All 4 views identical image → `total_views_analyzed=4`, per-view results consistent

### 7.5 Image Preprocessing Verification

- [ ] Input is resized to 260×260 before ONNX inference
- [ ] ImageNet normalization applied: `(pixel/255 - mean) / std` with mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`
- [ ] Array transposed to channel-first (CHW) format before ONNX input

### 7.6 Error Cases

- [ ] `damage_model.onnx` absent → HTTP 503 (not 500), clear file-not-found message
- [ ] `damage_model.onnx` present but `.onnx.data` sidecar missing → HTTP 503 with "external-data file" message
- [ ] ONNX session initialized only once per server lifetime — verify via server logs (not reloaded per request)

---

## 8. Card / Non-Card Classifier (Unit)

> Tests for `card_inference.py` model loading and inference in isolation.

### 8.1 Model Loading

- [ ] Loads from `.keras` zip file without requiring full Keras/TensorFlow (pure H5 weight reading)
- [ ] All 10 weight tensors loaded from correct H5 paths (e.g., `layers/conv2d/vars/0`)
- [ ] Model file not present → `FileNotFoundError` raised with full searched paths listed
- [ ] Falls back to `card_noncard_model.keras` if `card_noncard_classifier_model.keras` not found

### 8.2 Inference

- [ ] Input image resized to 224×224 via BILINEAR resampling before inference
- [ ] EXIF transpose applied before resize
- [ ] `normalize=True` (default): pixel values divided by 255.0
- [ ] `normalize=False` (CLI option): raw pixel values used — produces different probability than normalize=True
- [ ] Output probability always in range (0, 1) — never exactly 0 or 1 due to sigmoid clamp
- [ ] Sigmoid input clipped to `[-60, 60]` — no overflow with extreme activations
- [ ] `predict_probability` returns Python `float`, not NumPy scalar
- [ ] Same image → same probability across 5 repeated calls (no randomness)

---

## 9. OCR Pipeline & Mulkiya Field Extraction

> Tests for `ocr_simple_test.py` subprocess and the rule-based Mulkiya extractor.

### 9.1 Subprocess Invocation

- [ ] Subprocess called with correct args: `{path}`, `--write_text`, `--no_images`, `--lang {ocr_lang}`
- [ ] `--extract_mulkya` flag appended only for `process_type=mulkiya`
- [ ] `--prefer_pdf_text` flag appended only when `prefer_pdf_text=true` and input is a PDF
- [ ] `UPSURE_OCR_PYTHON` env var → that Python executable used for subprocess
- [ ] `UPSURE_OCR_PYTHON` not set AND default path missing → falls back to `sys.executable`
- [ ] Subprocess `cwd` set to project root (not temp dir)
- [ ] Subprocess `returncode != 0` → RuntimeError raised with full stdout and stderr included

### 9.2 OCR Output

- [ ] Arabic text on Mulkiya correctly recognized (PaddleOCR with `lang=ar`)
- [ ] `_ocr.json` written to same temp dir as input image, named `{stem}_ocr.json`
- [ ] `_mulkya.json` written when `--extract_mulkya` is passed, named `{stem}_mulkya.json`
- [ ] Arabic-Indic digits (٠١٢٣) converted to ASCII in extracted text
- [ ] Reversed Arabic character runs fixed (e.g., `ةنطلس` → `سلطنة`)
- [ ] Arabic text reshaped via `arabic_reshaper` for proper glyph joining

### 9.3 Mulkiya Field Extraction (`_extract_mulkya_rulebased`)

- [ ] `plate_number` — 3-7 digit sequence extracted correctly
- [ ] `plate_number` — Arabic-Indic plate format (5 digits + Arabic letter) extracted and digits converted to ASCII
- [ ] `vin_or_chassis` — 17-character VIN preferred over other alphanumeric sequences
- [ ] `vin_or_chassis` — chunked VIN split across lines (with spaces/dashes) reassembled
- [ ] `vin_or_chassis` — common artifacts rejected: "VEHICLE", "SULTANATE", "OMAN", "POLICE" not extracted as VIN
- [ ] `year` — 4-digit year extracted correctly
- [ ] `year` — 2-digit short form: `25` → `2025`, `87` → `1987`
- [ ] `year` — value outside 1900-2100 → `null`
- [ ] `expiry_date` — last date found in document (not first)
- [ ] `issue_date` — second-to-last date (only populated when ≥ 2 dates found)
- [ ] `engine_cc` — number found near keyword "المحرك"
- [ ] `make` — recognized from lookup list: Toyota/تويوتا, Nissan/نيسان, BMW, Lexus, etc.
- [ ] `make` — unlisted vehicle brand → `make=null`
- [ ] `owner_name` — currently returns `null` (hardcoded) — confirm this is expected behaviour
- [ ] `color` — value found after keyword "اللون"

### 9.4 Validation Notes

- [ ] `validation_notes` present in `extracted_data` when VIN length is outside 11-20 chars
- [ ] `validation_notes` present when plate digit count is outside 3-7
- [ ] `validation_notes` present when `year` is outside valid range 1900-2100
- [ ] `validation_notes` present when `engine_cc` is outside 200-10000
- [ ] `validation_notes` present when `empty_weight_kg > max_load_kg`
- [ ] `validation_notes` present when `seats` is outside 1-80
- [ ] `validation_notes` present when dates do not match `DD/MM/YY` or `YYYY/MM/DD` pattern

---

## 10. RAG Chunker

> Tests for `rag_json_chunker.py` in isolation and as called from the API.

- [ ] `chunk_json_file` returns `Chunk` objects with all required fields: `source_file`, `document_type`, `chunk_id`, `text`, `metadata`
- [ ] OCR pages JSON (`{"pages": [...]}`) → uses `ocr_pages` chunker with overlap
- [ ] Structured JSON (any other shape) → uses `structured_json` chunker, no overlap
- [ ] No single chunk `text` exceeds `max_chars=1200` characters
- [ ] OCR chunks: consecutive chunks share `overlap_lines=3` lines
- [ ] Structured JSON: deeply nested dict → correctly flattened to `key.subkey: value` lines
- [ ] Structured JSON: list values → indexed as `key[1]`, `key[2]`, etc.
- [ ] Empty JSON `{}` → returns empty chunk list, no crash
- [ ] JSON with top-level array → `item_count` used, no crash
- [ ] Non-existent file path → returns empty list (does not raise exception to caller)
- [ ] `source_file` on each chunk matches uploaded filename after relativization

---

## 11. File Type Routing Matrix

> Verify the correct HTTP response for every file type across all four `process_type` values.

### 11.1 Expected Routing Table

| File Type | `process_type=car` | `process_type=mulkiya` | `process_type=pdf` | `process_type=file` |
|---|---|---|---|---|
| `.jpg` / `.png` image | ✅ 200 — runs | ✅ 200 — runs | ❌ 400 | ✅ 200 — `category=image` |
| `.pdf` | ❌ 400 (car needs image) | ✅ 200 — runs | ✅ 200 — runs | ✅ 200 — `category=pdf` |
| `.docx` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=archive` |
| `.xlsx` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=archive` |
| `.csv` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=tabular` |
| `.txt` / `.md` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=text` |
| `.json` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=json` |
| `.zip` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=archive` |
| `.xml` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=xml` |
| Binary / `.exe` | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=binary` |
| No extension | ❌ 415 | ❌ 415 | ❌ 415 | ✅ 200 — `category=binary` |

### 11.2 Routing Checklist

- [ ] `.docx` sent with `process_type=mulkiya` → HTTP 415
- [ ] `.xlsx` sent with `process_type=car` → HTTP 415
- [ ] `.csv` sent with `process_type=pdf` → HTTP 415
- [ ] `.zip` sent with `process_type=mulkiya` → HTTP 415
- [ ] `.txt` sent with `process_type=car` → HTTP 415
- [ ] Image (`.jpg`) sent with `process_type=pdf` → HTTP 400
- [ ] PDF sent with `process_type=car` → HTTP 400
- [ ] `.docx` sent with `process_type=file` → HTTP 200, `category=archive`
- [ ] Binary `.exe` file sent with `process_type=file` → HTTP 200, `category=binary`

### 11.3 Extension / Content-Type Spoofing

> The API detects file type from BOTH filename extension AND content-type header. Mismatches reach business logic.

- [ ] **JPEG bytes with `.pdf` filename** + `process_type=mulkiya` → OCR subprocess fails on JPEG-as-PDF → HTTP 500 with descriptive error (not silent garbage result)
- [ ] **JPEG bytes with `.pdf` filename** + `process_type=pdf` → OCR subprocess fails → HTTP 500
- [ ] **PDF bytes with `.jpg` filename** + `process_type=car` → PIL cannot open PDF as image → HTTP 500, error message present
- [ ] **PDF bytes with `.jpg` filename** + `process_type=mulkiya` → PIL fails on card model → HTTP 500
- [ ] **PDF bytes with `.jpg` filename** + `process_type=pdf` → HTTP 400 (detected as image, not PDF — safe)
- [ ] **JPEG with `Content-Type: application/pdf`** + `process_type=mulkiya` → treated as PDF, OCR fails → HTTP 500
- [ ] **DOCX bytes with `.jpg` filename** + `process_type=car` → PIL fails → HTTP 500, error is descriptive

---

## 12. Input Validation & Security

- [ ] No `filename` on upload → HTTP 400 "Uploaded file must have a filename"
- [ ] Invalid `process_type` value (e.g., `process_type=unknown`) → HTTP 422 (Literal type rejection)
- [ ] Zero-byte image file → PIL fails gracefully → HTTP 500, no server crash
- [ ] Filename with path traversal (e.g., `../../etc/passwd.jpg`) → `Path(filename).name` strips to basename only — no directory traversal
- [ ] Filename with special characters and Arabic text (as seen in real uploads) → temp file created safely, no OS error
- [ ] Very large file upload (e.g., 50 MB image) → server handles gracefully (no OOM crash); note: no explicit size limit currently enforced
- [ ] CORS preflight (`OPTIONS`) request → HTTP 200 with `Access-Control-Allow-Origin: *` header
- [ ] `Access-Control-Allow-Origin: *` is acceptable for PoC — **must be restricted to known origins in production**
- [ ] `RequestValidationError` (Pydantic) → HTTP 422 response body is JSON with no raw Python `bytes` objects (sanitizer active)

---

## 13. Error Handling & HTTP Status Code Contract

| Condition | Expected Status | Checked |
|---|---|---|
| No `filename` on upload | 400 | `[ ]` |
| Invalid `process_type` value | 422 | `[ ]` |
| Non-image sent to `/predict/` | 400 | `[ ]` |
| Non-image, non-PDF to `car` / `mulkiya` / `pdf` | 415 | `[ ]` |
| PDF sent to `process_type=car` | 400 | `[ ]` |
| Non-PDF sent to `process_type=pdf` | 400 | `[ ]` |
| Zero views sent to `/predict/damage` | 400 | `[ ]` |
| Damage model file missing | **503** (not 500) | `[ ]` |
| Damage model `.onnx.data` sidecar missing | 503 with specific message | `[ ]` |
| OCR subprocess fails (non-zero exit) | 500 with stdout + stderr | `[ ]` |
| OCR JSON not written by subprocess | 500 "OCR JSON output was not created" | `[ ]` |
| Car model file missing | 500 on first `/predict/` or `process_type=car` | `[ ]` |
| Card model file missing | 500 on `process_type=mulkiya` | `[ ]` |
| Pydantic validation error | 422 with sanitized JSON body | `[ ]` |
| CORS preflight (OPTIONS) | 200 + CORS headers | `[ ]` |

- [ ] All 500 error responses contain a human-readable `detail` string — no raw stack traces or Python exceptions leaked in production
- [ ] All 422 responses contain `detail` list with no raw `bytes` objects (sanitization active in `validation_exception_handler`)

---

## 14. Response Schema Completeness

> Every `/api/v1/process` response must contain these top-level keys regardless of `process_type`.

### 14.1 Unified Process Endpoint

- [ ] `input.filename` — always present, matches uploaded filename
- [ ] `input.kind` — always present (`"image"`, `"pdf"`, `"file"`)
- [ ] `classification` — `null` for `process_type=car`, populated for all others
- [ ] `car_classification` — `null` for all types except `process_type=car`
- [ ] `confidence_score` — always a `float` (never `null`; defaults to `0.0`)
- [ ] `extracted_data` — `null` when not applicable (car type, skip_ocr)
- [ ] `raw_ocr` — `null` when OCR not run
- [ ] `rag_chunks` — always an **array** (empty `[]` when no chunks — never `null`)
- [ ] `artifacts.chunk_source` — string path or `null`
- [ ] `note` — non-empty string describing what processing occurred

### 14.2 Direct Car Predict Endpoint (`/predict/`)

- [ ] `filename` present
- [ ] `is_car` — boolean
- [ ] `confidence` — float
- [ ] `raw_score` — float
- [ ] `threshold_used` — float (should equal `0.35`)

### 14.3 Damage Detect Endpoint (`/predict/damage`)

- [ ] `damage_detected` — boolean
- [ ] `total_views_analyzed` — integer ≥ 1
- [ ] `overall_confidence` — float
- [ ] `per_view` — dict, one key per submitted view name
- [ ] Each view entry contains: `damage_detected`, `confidence_score`, `prob_damaged`, `prob_clean`

### 14.4 Response Headers

- [ ] `X-Process-Time-ms` header present on **all** responses including error responses (middleware applied globally)

---

## 15. Latency Benchmarks

> Use `latency_analyzer.py` for all measurements. Run with `--runs 10 --warmup 1` per scenario. Save output with `--output-json results.json`.

### 15.1 Benchmark Commands

```bat
:: Health check
python latency_analyzer.py --scenario health --runs 10

:: Direct car classification
python latency_analyzer.py --scenario predict-car --runs 10

:: Damage detection (4 views)
python latency_analyzer.py --scenario predict-damage --runs 10 ^
  --front-file Samples\car_1001.jpg --back-file Samples\car_1003.jpg ^
  --left-file Samples\car_1005.jpg --right-file Samples\car_1007.jpg

:: Car via unified API
python latency_analyzer.py --scenario process-car --runs 10

:: Mulkiya classification only (no OCR)
python latency_analyzer.py --scenario process-mulkiya --mulkiya-skip-ocr --runs 10

:: Mulkiya with full OCR
python latency_analyzer.py --scenario process-mulkiya --runs 10

:: PDF with OCR
python latency_analyzer.py --scenario process-pdf --runs 10

:: Standalone car API
python latency_analyzer.py --scenario standalone-car --runs 10

:: Full run, save JSON
python latency_analyzer.py --scenario all --runs 10 --output-json latency_results.json
```

### 15.2 Targets

> ⚠ Rows marked **"Fill after test run"** are placeholders — run the benchmark, record the p95, then set the SLA. Do not go to production with these blank.

| Scenario | Target p95 | Status |
|---|---|---|
| `GET /health` | < 20 ms | ✅ Defined |
| `POST /predict/` (car) | < 300 ms | ✅ Defined |
| `POST /predict/damage` (4 views) | < 2,000 ms | ✅ Defined |
| `process_type=car` | < 300 ms | ✅ Defined |
| `process_type=mulkiya` (no OCR) | < 300 ms | ✅ Defined |
| `process_type=mulkiya` (with OCR) | **_______ ms** | ⚠ Fill after test run |
| `process_type=mulkiya` (with OCR, PDF) | **_______ ms** | ⚠ Fill after test run |
| `process_type=pdf` (1 page) | **_______ ms** | ⚠ Fill after test run |
| `process_type=pdf` (5 pages) | **_______ ms** | ⚠ Fill after test run |
| `process_type=pdf` (10+ pages) | **_______ ms** | ⚠ Fill after test run |
| Standalone car API | < 300 ms | ✅ Defined |

### 15.3 Checklist

- [ ] All scenarios: `success_rate = 100%`
- [ ] No scenario has any errors in the `errors` list
- [ ] `server_ms` (from `X-Process-Time-ms`) vs `elapsed_ms` gap < 100 ms on localhost
- [ ] Second run faster than first for model-heavy endpoints (confirms singleton model loading)
- [ ] Warmup run excluded from reported statistics
- [ ] `--output-json` saves valid parseable JSON report
- [ ] Mulkiya with OCR vs `skip_ocr=true`: OCR subprocess overhead isolated and documented in ms
- [ ] Damage detection: 1-view vs 4-view latency ratio documented (should be roughly linear)
- [ ] **After baseline runs: fill in all blank p95 targets in Section 15.2 above and get sign-off before production**

---

## 16. Concurrency & Load Tests

> Use `latency_analyzer.py --concurrency N` for these tests.  
> ⚠ Max supported concurrency is currently undefined — run escalating tests to find the failure point, then set the production limit at N-1.

- [ ] `--concurrency 5` on `predict-car`: all requests succeed, 0 errors
- [ ] `--concurrency 5` on `predict-damage` (4 views): no race conditions, no 500s
- [ ] `--concurrency 10` on `GET /health`: 100% success, p95 < 50 ms
- [ ] `--concurrency 5` on `process-mulkiya` (skip OCR): no temp dir collisions between requests
- [ ] `--concurrency 3` on `process-pdf`: OCR subprocesses run independently (each uses its own temp directory)
- [ ] Memory is stable after 50 sequential requests — no growing heap or model reload per request
- [ ] Memory ceiling defined: record RSS after 50 requests, set max acceptable MB, document it — **fill after test: _______ MB**
- [ ] Temp directories cleaned up after each request — no accumulation of `upsure-poc-*` dirs in OS temp folder
- [ ] Models (`_card_model`, `_car_model`, `_damage_session`) loaded only once — confirm via server log: "loading model" appears exactly once per model type
- [ ] Escalating concurrency test: run `--concurrency 5`, `10`, `20`, `50` on `predict-car` — record the N where first errors appear — **max supported concurrency = N-1 → fill after test: _______**
- [ ] **After escalating test: document max concurrency value and get sign-off before production**

---

## 17. Regression / Golden Tests

> These use known sample files and must pass on every build. Results should be deterministic — same output on every run.

| Sample File | Pipeline | Expected Result |
|---|---|---|
| `car_10.jpg` | `process_type=car` | `is_car=true` |
| `car_1001.jpg` | `process_type=car` | `is_car=true` |
| `car_1003.jpg` | `process_type=car` | `is_car=true` |
| `car_1005.jpg` | `process_type=car` | `is_car=true` |
| `car_1007.jpg` | `process_type=car` | `is_car=true` |
| `Non_card_image_1.jpeg` | `process_type=car` | `is_car=false` |
| `Non_card_image_2.jpeg` | `process_type=car` | `is_car=false` |
| `Mulkiya_front.jpg` | `process_type=mulkiya, skip_ocr=true` | `label=card`, `probability > 0.9` |
| `Mulkiya_back.jpg` | `process_type=mulkiya, skip_ocr=true` | `label=card`, `probability > 0.9` |
| `Non_card_image_1.jpeg` | `process_type=mulkiya, skip_ocr=true` | `label=not card` |
| `Non_card_image_2.jpeg` | `process_type=mulkiya, skip_ocr=true` | `label=not card` |
| `sample_pdf.pdf` | `process_type=pdf` | `confidence_score > 0`, `lines` non-empty |
| `README.md` | `process_type=file` | `category=text` |
| `car_1001.jpg` (front only) | `POST /predict/damage` | Valid response, no crash |
| All 4 car samples | `POST /predict/damage` | `total_views_analyzed=4` |

- [ ] All golden cases above pass after models are downloaded
- [ ] All golden cases return **identical decisions** across 5 repeated runs (no stochastic variance)

---

## 18. Outer Payload & Integration Tests

> Tests derived from the real insurance order payload structure. The data-ingestion API is called per-document; results are embedded in `processResult` within `insurerJson`.

### 18.1 insurerJson Serialization

- [ ] `insurerJson` is a valid parseable JSON string — `JSON.parse(payload.insurerJson)` succeeds
- [ ] `insurerJson` contains Arabic text in field values — round-trips correctly through string serialization
- [ ] `metaData` top-level field is also a string — parseable separately from `insurerJson`
- [ ] Consumer code handles double-serialization: `insurerJson` is a string inside JSON, not a nested object

### 18.2 Document Types in `documentsData`

- [ ] `Mulkiya Front` → processed with `process_type=mulkiya`, `skip_ocr=true` → `label=card`, `probability > 0.9`
- [ ] `Mulkiya Back` → same as above
- [ ] `License ID` → processed with `process_type=file` → `suggested_process_type=mulkiya`, `category=image`
- [ ] License ID two-step: after `file` inspection returns `suggested_process_type=mulkiya`, re-processing with `process_type=mulkiya` produces card classification
- [ ] `isInvalidDoc=false` present on all valid documents
- [ ] `isInvalidDoc=true` scenario: system flags document as invalid when classification fails or `label=not card` — verify downstream handling

### 18.3 `breakIndocumentsData` (Vehicle Views)

- [ ] `Vehicle Front View` → `process_type=car` → `is_car=true`, `confidence > 0.35`
- [ ] `Vehicle Back View` → `process_type=car` → `is_car=true`
- [ ] `Vehicle Left View` → `process_type=car` → `is_car=true`
- [ ] `Vehicle Right View` → `process_type=car` → `is_car=true`
- [ ] `isInvalidCar=false` present on all valid car views
- [ ] `isBreakIn=false` scenario → no `breakIndocumentsData` → car classification not invoked → system handles gracefully
- [ ] `breakIndocumentsData` empty array → no car views processed, no crash
- [ ] `breakinStatus="success"` present when `isBreakIn=true`

### 18.4 Optional / Empty Member Fields

- [ ] Order with all member string fields empty (`memberId=""`, `title=""`, `firstName=""`, etc.) → accepted, pipeline processes documents normally
- [ ] `emailId=""` → no email validation error
- [ ] `userAge=""` → treated as absent, no numeric parse crash
- [ ] `mobileNum="+919999999393"` with country code prefix → accepted

### 18.5 Optional Quote Fields

- [ ] `addonIdList=[]` (no addons selected) → valid payload, no crash
- [ ] `addonIdList` entries with `isSelected=false` → handled as inactive addons
- [ ] `addonIdList[*].addonTotalPremium="0"` (string zero) → accepted
- [ ] `sumAssured=0` → valid for third-party policy
- [ ] `legalLiabilityCover=[]`, `accessoriesCover=[]`, `unnamedPA=[]` → all empty arrays accepted
- [ ] `vehicleDetails.idv="0"` (string, not integer) → accepted without type error

### 18.6 processResult Null Fields Contract

> As seen in real production data — consumers must handle these null/empty values.

- [ ] `extracted_data=null` (when `skip_ocr=true`) → downstream consumers handle null without crash
- [ ] `raw_ocr=null` (when OCR not run) → consumers handle null
- [ ] `rag_chunks=[]` (empty array, not null) — always an array, confirm never null
- [ ] `car_classification=null` for mulkiya/file documents — consumers check before accessing `.is_car`
- [ ] `classification=null` for car documents — consumers check before accessing `.label`
- [ ] `artifacts.chunk_source=null` when no chunks generated — consumers handle null

---

## 19. Operational & Deployment Checks

- [ ] API cold start (no models pre-loaded) completes and serves first request in < 60 seconds
- [ ] First request to each model endpoint triggers lazy load — server log shows model load message
- [ ] Subsequent requests do not reload models — log shows no additional "loading" messages
- [ ] `PORT` env var overrides default 8000 (e.g., `PORT=9000 python poc_api.py` starts on 9000)
- [ ] `UPSURE_DAMAGE_MODEL` env var overrides default damage model path
- [ ] Both `poc_api` (port 8000) and `car_classifier_api` (port 8001) run simultaneously without conflict
- [ ] `latency_analyzer.py --scenario all --runs 5 --output-json results.json` completes fully and writes valid JSON
- [ ] `results.json` contains all scenario summaries with `elapsed_ms` and `server_ms` breakdowns
- [ ] No dangling `upsure-poc-*` temp directories in OS temp folder after 10 sequential requests
- [ ] Server handles `SIGINT` (Ctrl+C) gracefully — OCR subprocesses do not become zombie processes
- [ ] `GET /health` with damage model missing → `damage_model_ready: false` + `damage_model_error` message (not a crash)

---

## 20. Per-Insurer Document Validation

> Business rule: required documents differ by insurer and by case type (new vehicle / renewal / break-in).  
> The system must validate document presence and classification correctness against the correct ruleset before processing.

---

### Document Requirement Reference

| Insurer | Case Type | Required Documents |
|---|---|---|
| **AssureTech** | All | Mulkiya F+B, License ID F+B, Civil ID F+B |
| **Arabia Falcon** | New Vehicle | License F+B, Purchase Invoice |
| **Arabia Falcon** | Renewal / Other | Mulkiya F+B, License F+B |
| **NIA** | New Vehicle | License ID, Purchase Invoice |
| **NIA** | Renewal / Other | Mulkiya Front only |
| **GIG Motor** | All | Mulkiya F+B, License ID, National ID *(+ 5 MB size limit)* |
| **Al Madina** | All | Mulkiya Front only |
| **MIC** | New Vehicle | License ID, Purchase Invoice |
| **MIC** | Renewal / Other | Mulkiya F+B, License ID |
| **Takaful Commercial** | All | Mulkiya Front, License ID |
| **Liva / Default** | All | Mulkiya F+B, License ID |
| **ALL insurers** | Break-in (expired policy) | Above docs **+** Vehicle Front, Back, Left, Right |

---

### 20.1 Break-in Case (Universal — All Insurers)

- [ ] `isBreakIn=true` → all 4 vehicle views required: Front, Back, Left, Right
- [ ] Any single vehicle view missing when `isBreakIn=true` → validation failure surfaced
- [ ] All 4 views have `is_car=true` → break-in documents accepted
- [ ] Any view returns `is_car=false` → `isInvalidCar=true` set on that view → submission blocked
- [ ] `isBreakIn=false` → vehicle view photos not required → system does not reject for absent views
- [ ] Break-in + any insurer: both the 4 vehicle views AND insurer-specific docs must all pass simultaneously
- [ ] `breakinStatus="success"` present when `isBreakIn=true` — absent or failed status handled gracefully

### 20.2 AssureTech

- [ ] Mulkiya Front present + `label=card` → accepted
- [ ] Mulkiya Back present + `label=card` → accepted
- [ ] License ID Front present + classified → accepted
- [ ] License ID Back present + classified → accepted
- [ ] Civil ID Front present + classified → accepted
- [ ] Civil ID Back present + classified → accepted
- [ ] Any one of the 6 documents missing → validation failure flagged
- [ ] Any document with `isInvalidDoc=true` → entire submission blocked
- [ ] Break-in case: all 6 docs + all 4 vehicle views required (10 total)

### 20.3 Arabia Falcon

- [ ] **New vehicle case:** License Front + Back + Purchase Invoice present → accepted
- [ ] **New vehicle case:** Mulkiya submitted instead of Purchase Invoice → should not satisfy requirement
- [ ] **New vehicle case:** Purchase Invoice absent → validation failure
- [ ] **Renewal case:** Mulkiya F+B + License F+B present → accepted
- [ ] **Renewal case:** Purchase Invoice absent → accepted (not required for renewal)
- [ ] **Renewal case:** Mulkiya Back absent → validation failure
- [ ] **Renewal case:** License Back absent → validation failure
- [ ] Case type (`new vehicle` vs `renewal`) correctly determined from payload fields
- [ ] Break-in + new vehicle: License F+B + Invoice + 4 car views all required

### 20.4 NIA (New India Assurance)

- [ ] **New vehicle:** License ID + Purchase Invoice present → accepted
- [ ] **New vehicle:** Mulkiya submitted → extra doc, must not block submission
- [ ] **New vehicle:** Purchase Invoice absent → validation failure
- [ ] **Renewal:** Mulkiya Front only present → accepted
- [ ] **Renewal:** Mulkiya Back also uploaded → accepted (extra doc, no rejection)
- [ ] **Renewal:** Mulkiya Back absent → **accepted** — Back is NOT required for NIA renewal
- [ ] **Renewal:** No Mulkiya at all → validation failure
- [ ] **Renewal:** Only License ID submitted (no Mulkiya) → validation failure

### 20.5 GIG Motor

- [ ] Mulkiya Front + Back + License ID + National ID all present → accepted
- [ ] National ID absent → validation failure (National ID is unique to GIG — no other insurer requires it)
- [ ] Mulkiya Back absent → validation failure
- [ ] **File size: any uploaded file > 5 MB → rejected with file size error** *(strict limit, unique to GIG)*
- [ ] File exactly 5 MB → accepted (boundary value — must not reject at limit)
- [ ] File 5 MB + 1 byte → rejected
- [ ] Each of the 4 required docs is individually ≤ 5 MB → all accepted
- [ ] One doc ≤ 5 MB, another > 5 MB → only the oversized doc rejected, not the whole set
- [ ] Break-in + GIG: all 4 required docs + 4 vehicle views, every file ≤ 5 MB

### 20.6 Al Madina

- [ ] Mulkiya Front present + `label=card` → accepted
- [ ] Mulkiya Front absent → validation failure
- [ ] Mulkiya Back submitted → accepted (extra doc, must not block)
- [ ] Only License ID submitted with no Mulkiya → validation failure
- [ ] Minimal submission: Mulkiya Front alone → accepted (no other doc required)

### 20.7 MIC (Muscat Insurance Company)

> MIC is the insurer in the example production payload.

- [ ] **New vehicle:** License ID + Purchase Invoice → accepted
- [ ] **New vehicle:** Mulkiya not required → extra Mulkiya upload does not block
- [ ] **New vehicle:** Purchase Invoice absent → validation failure
- [ ] **Renewal:** Mulkiya Front + Back + License ID → accepted
- [ ] **Renewal:** Mulkiya Back absent → validation failure
- [ ] **Renewal:** License ID absent → validation failure
- [ ] **Renewal:** Mulkiya Front absent → validation failure
- [ ] Break-in + MIC renewal: Mulkiya F+B + License ID + 4 vehicle views all required

### 20.8 Takaful Commercial

- [ ] Mulkiya Front + License ID both present → accepted
- [ ] Mulkiya Back absent → accepted (not required)
- [ ] License ID absent → validation failure
- [ ] Mulkiya Front absent → validation failure
- [ ] Only Mulkiya Front without License ID → validation failure

### 20.9 Liva / Default (All Other Insurers)

- [ ] Mulkiya Front + Back + License ID → accepted
- [ ] Mulkiya Back absent → validation failure
- [ ] License ID absent → validation failure
- [ ] Unknown / unrecognised insurer code → falls through to default (Liva) rules, does not crash
- [ ] Default rules applied correctly for any insurer not explicitly listed above

### 20.10 Cross-Cutting Validation Rules

- [ ] `isInvalidDoc=false` required on each document — any `true` blocks the insurer's submission
- [ ] `isInvalidCar=false` required on each vehicle view for break-in — any `true` blocks break-in
- [ ] Document classification confidence below threshold → `isInvalidDoc=true` set automatically
- [ ] Insurer code / company name maps correctly to requirement ruleset — test all 8 insurers + default
- [ ] `new vehicle` vs `renewal` determination reads the correct field from payload — wrong case type routing produces wrong doc requirement set
- [ ] Purchase Invoice: validated as present via `process_type=file` — no content extraction required, only presence check
- [ ] Civil ID (AssureTech): classified as `card` by card/non-card model — classification confidence is above threshold
- [ ] National ID (GIG Motor): classified as `card` — distinct from Civil ID in document type name, same classification path
- [ ] Two-sided documents (Mulkiya F+B, License F+B, Civil ID F+B): both sides must independently pass classification — one side passing does not satisfy the pair requirement

### 20.11 Unique Edge Cases per Insurer Summary

| Insurer | Unique Risk |
|---|---|
| **GIG Motor** | 5 MB hard file size limit — only insurer with this constraint — boundary test critical |
| **NIA Renewal** | Mulkiya Front only required — Back must NOT cause rejection if absent |
| **Al Madina** | Simplest case — one doc only — test that nothing extra is required |
| **Arabia Falcon** | Two completely different doc sets by vehicle age — case routing failure has large impact |
| **AssureTech** | Highest doc count (6 docs) — most permutations for missing-doc failures |
| **MIC** | Same new/renewal split pattern as Arabia Falcon — both must be independently verified |
| **Takaful** | Mulkiya Back explicitly not required — confirm system does not demand it |

---

## 21. Production Readiness Gates

> These items **cannot be signed off based on the checklist alone** — they require real test data runs and a human decision on acceptable thresholds. None of these have defined targets yet. All blanks must be filled and approved before production deployment.

---

### 21.1 OCR Field Extraction Accuracy

> Current state: `_extract_mulkya_rulebased` extracts 14 fields from Mulkiya. No accuracy target exists anywhere in the codebase or docs. Wrong `expiry_date` or `plate_number` means wrong policy issued.

**Test method:** Run Mulkiya OCR on ≥ 50 real labeled Mulkiya images (front side). Compare extracted values against ground truth. Record accuracy per field.

| Field | Target Accuracy | Measured | Pass? |
|---|---|---|---|
| `plate_number` | ≥ **_____%** | ______% | `[ ]` |
| `expiry_date` | ≥ **_____%** | ______% | `[ ]` |
| `vin_or_chassis` | ≥ **_____%** | ______% | `[ ]` |
| `year` | ≥ **_____%** | ______% | `[ ]` |
| `make` | ≥ **_____%** | ______% | `[ ]` |
| `engine_cc` | ≥ **_____%** | ______% | `[ ]` |
| `issue_date` | ≥ **_____%** | ______% | `[ ]` |
| `owner_name` | Currently always `null` — confirm intentional | N/A | `[ ]` |

- [ ] Minimum 50 labeled Mulkiya images collected (mix of clear, blurry, angled, low-light)
- [ ] All field targets above filled in and approved before production
- [ ] `validation_notes` fires correctly on at least 5 intentionally bad images (wrong VIN length, bad year, etc.)

---

### 21.2 Card Classification Accuracy on Real-World Images

> Current state: Tested on 2 Mulkiya samples (`Mulkiya_front.jpg`, `Mulkiya_back.jpg`) and 2 non-card samples. This is a PoC sample size, not a production eval.

**Test method:** Run card/non-card classification on ≥ 50 real Mulkiya images + ≥ 50 real non-card images. Record accuracy, false positive rate, false negative rate.

| Metric | Target | Measured | Pass? |
|---|---|---|---|
| Accuracy on real Mulkiya images | ≥ **_____%** | ______% | `[ ]` |
| False negative rate (Mulkiya classified as not-card) | ≤ **_____%** | ______% | `[ ]` |
| False positive rate (non-card classified as card) | ≤ **_____%** | ______% | `[ ]` |

- [ ] ≥ 50 real Mulkiya images tested (including phone photos, rotated, low resolution)
- [ ] ≥ 50 real non-card images tested (car photos, selfies, random documents)
- [ ] Targets above filled in and approved before production
- [ ] Tested on Civil ID and License ID images as well (AssureTech, GIG require these)

---

### 21.3 Damage Model Evaluation at Production Scale

> Current state: 95% accuracy, 100% recall, 90.9% precision on 100 cars × 4 views. This is a PoC eval size. For insurance, missing damage (false negative) has direct financial impact.

**Test method:** Evaluate on ≥ 500 labeled cars (damaged + clean) with 4 views each.

| Metric | Current (PoC) | Production Target | Measured at Scale | Pass? |
|---|---|---|---|---|
| Accuracy | 95% | ≥ **_____%** | ______% | `[ ]` |
| Recall (no missed damage) | 100% | ≥ **_____%** | ______% | `[ ]` |
| Precision | 90.9% | ≥ **_____%** | ______% | `[ ]` |
| Test set size | 100 cars | ≥ 500 cars | ______ cars | `[ ]` |

- [ ] Labeled dataset of ≥ 500 cars (real insurance claim images if available) collected
- [ ] Eval rerun on expanded dataset — metrics recorded in this table
- [ ] Production targets set and approved — especially recall (insurance cannot miss damage)
- [ ] Single-view accuracy tested separately (real users may not send all 4 views)

---

### 21.4 OCR Subprocess Timeout

> Current state: No timeout is set on the `subprocess.run` call in `_run_ocr_script`. A hung PaddleOCR process blocks the request thread indefinitely. This is a production reliability risk.

- [ ] Timeout value decided (recommended: 60–120 seconds): **_______ seconds**
- [ ] `timeout=` parameter added to `subprocess.run` call in `_run_ocr_script`
- [ ] Test: PaddleOCR given a pathological input → subprocess times out cleanly → HTTP 500 returned in ≤ timeout + 2 seconds
- [ ] Timeout value documented and approved

---

### 21.5 Maximum File Size Limit (Non-GIG Insurers)

> Current state: Only GIG Motor has a defined 5 MB limit (business rule). No other insurer has an enforced size limit in the API. Real phone camera photos can be 8–15 MB.

- [ ] Maximum accepted file size decided for all non-GIG pipelines: **_______ MB**
- [ ] Test: file at the limit → accepted
- [ ] Test: file 1 byte over the limit → rejected with HTTP 413 or clear error
- [ ] Large image (15 MB, 12MP phone photo) behaviour documented: accepted or rejected?
- [ ] Size limit enforced in code or at reverse proxy / API gateway level — confirm which
- [ ] Size limit documented per insurer:

| Insurer | Max File Size |
|---|---|
| GIG Motor | 5 MB (hard — business rule) |
| AssureTech | **_______** |
| Arabia Falcon | **_______** |
| NIA | **_______** |
| Al Madina | **_______** |
| MIC | **_______** |
| Takaful Commercial | **_______** |
| Liva / Default | **_______** |

---

### 21.6 End-to-End Error Rate SLA

> Current state: No acceptable error rate defined. In production, some % of requests will fail (corrupt images, network issues, OCR subprocess crashes). This threshold must be agreed before go-live.

- [ ] Acceptable error rate (HTTP 5xx) decided: ≤ **_______%** over a rolling 1-hour window
- [ ] Monitoring / alerting threshold set at that value
- [ ] Test: inject 5% bad requests (corrupt images, zero-byte files) into a 100-request run — confirm 5xx rate ≤ target
- [ ] OCR subprocess failure rate on real Mulkiya images measured and documented: **_______%**

---

### 21.7 Production Readiness Sign-Off Checklist

> All items below must be checked **after** Sections 21.1–21.6 are completed. This is the final gate.

- [ ] All blank latency targets in Section 15.2 filled in and approved
- [ ] Max concurrency value from Section 16 filled in and approved
- [ ] Memory ceiling from Section 16 filled in and approved
- [ ] OCR field extraction accuracy targets (Section 21.1) filled and approved
- [ ] Card classification accuracy targets (Section 21.2) filled and approved
- [ ] Damage model expanded eval (Section 21.3) completed and approved
- [ ] OCR subprocess timeout (Section 21.4) implemented, tested, approved
- [ ] Max file size per insurer (Section 21.5) decided and enforced
- [ ] Error rate SLA (Section 21.6) decided and monitoring in place
- [ ] All P0 checklist items across Sections 1–20 are marked `[x]`
- [ ] All P1 checklist items across Sections 1–20 are marked `[x]` or `[S]` with documented reason for skip

---

## Summary

| Section | # of Items | Priority |
|---|---|---|
| 1. Environment & Setup | 14 | P0 — Must pass before any other test |
| 2. Smoke Tests | 10 | P0 — Must pass first |
| 3. Car Classification | 21 | P1 |
| 4. Mulkiya Pipeline | 18 | P1 |
| 5. PDF Pipeline | 12 | P1 |
| 6. File Inspection | 20 | P2 |
| 7. Car Damage Detection | 19 | P1 |
| 8. Card/Non-Card Classifier | 11 | P2 |
| 9. OCR & Mulkiya Extraction | 24 | P1 |
| 10. RAG Chunker | 11 | P2 |
| 11. File Type Routing Matrix | 16 | P2 |
| 12. Input Validation & Security | 9 | P1 |
| 13. Error Handling | 16 | P1 |
| 14. Response Schema | 18 | P1 |
| 15. Latency Benchmarks | 13 | P1 |
| 16. Concurrency & Load | 11 | P2 |
| 17. Regression / Golden Tests | 15 | P0 — Must pass before release |
| 18. Outer Payload & Integration | 22 | P1 |
| 19. Operational Checks | 11 | P2 |
| 20. Per-Insurer Document Validation | 58 | P0 — Core business rules |
| 21. Production Readiness Gates | 35 | P0 — Cannot go live without sign-off |
| **Total** | **~383** | |

### Priority Guide

| Priority | Meaning |
|---|---|
| **P0** | Blocker — system cannot be considered functional without these |
| **P1** | Required before any production handoff |
| **P2** | Required before scale-up or broader rollout |
