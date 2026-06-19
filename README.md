# Car Damage Detection + Mulkiya OCR — UpsureAI Internship

My personal write-up of the engineering I did during a Summer 2026 internship at
**UpsureAI** (client: **Tameen**, a motor-insurance broker in the UAE/KSA). The
production service lives on a private company GitLab; this repo collects the work
that is mine, in two phases.

For the full narrative, metrics, and honest limitations, read **[PORTFOLIO.md](PORTFOLIO.md)**
and **[ISSUES.md](ISSUES.md)**.

## Phase 1 — Car damage model (knowledge distillation)

Root of this repo. A VLM-distillation pipeline: a vision LLM auto-labels real car
images, an EfficientNet-B2 student learns from those labels and exports to ONNX
for CPU inference, plus a YOLO localizer for damage type / severity / parts.

| File | What |
|---|---|
| `batch_label.py`, `filter_new_data.py` | VLM auto-labeling + dataset screening |
| `prepare_dataset.py`, `remap_classes.py` | Dataset build + 11→6 YOLO class remap |
| `train.py`, `train_colab.ipynb`, `train_yolo_colab.ipynb` | EfficientNet-B2 + YOLO training |
| `damage_detect.py`, `parts_rules.py`, `repair_rules.py` | Inference + severity/parts/repair rules |
| `api.py` | FastAPI inference server (ONNX runtime) |

## Phase 2 — Mulkiya OCR (`mulkiya_ocr/`)

Reading UAE/Oman vehicle-registration cards (Arabic + English) into structured
fields, CPU-only, on RapidOCR (PP-OCR on onnxruntime).

| File | What |
|---|---|
| `card_crop.py` | Detect + crop the mulkiya card from a photo |
| `ocr_simple_test.py` | Dual-pass Arabic+English OCR engine (see header note on authorship) |
| `prototype_spatial.py` | Spatial field extraction from OCR boxes |
| `prototype_eval_batch.py` | Batch evaluation over a sample set |
| `verify_mulkiya_batch.py` | Verification harness — structural validity scoring |

## Tech stack
FastAPI · onnxruntime · RapidOCR (PP-OCR) · EfficientNet-B2 · YOLO11m · OpenCV ·
PyMuPDF. CPU-only throughout.
