# Document Ingestion & Vehicle Damage AI

> Personal showcase of an internship project at **UpsureAI** (client: **Tameen**,
> a motor-insurance broker in the UAE/KSA). The production version lives on a
> private company GitLab; this repo is my personal write-up of the engineering.

A single CPU-only FastAPI service that ingests car photos and UAE/Oman vehicle
documents, runs ML + OCR, and returns structured JSON — built to cut insurance
claim processing from ~5–7 days to ~24 hours.

## What it does

| Capability | Endpoint | What happens |
|---|---|---|
| **Car damage detection** | `POST /predict/damage` | Binary damage model gates a YOLO localizer; adds severity, parts-at-risk, repair recommendation per view |
| **Mulkiya OCR** | `POST /api/v1/process` (mulkiya) | Reads UAE/Oman vehicle-registration cards (Arabic + English) into structured fields |
| **Car classification** | `POST /predict/` | Is this a car? |

## Engineering highlights

- **Hard constraint: CPU-only, 4 GB.** Every model/runtime choice follows from this.
- **Knowledge distillation for damage detection** — a 235B-parameter vision LLM
  auto-labels 3,519 real car images (969 vehicles); an EfficientNet-B2 (9M params)
  student learns from those labels and exports to ONNX for CPU inference.
- **Replaced PaddleOCR/PaddlePaddle with RapidOCR** (PP-OCR on onnxruntime) after
  PaddlePaddle kept crashing the production container — no PaddlePaddle, no PyTorch.
- **Arabic + English dual-pass OCR** for mulkiya cards, since one recognizer can't
  read both scripts well.
- **Document gating without a bespoke model** — an OCR **anchor gate + vehicle-spec
  fingerprint** rejects non-mulkiya documents (a driving licence *is* a card, so a
  card classifier alone can't catch it).
- **Outcome-based image-quality gate** — instead of pixel-blur (which I measured to
  be unreliable: a clean scan scored the *lowest* sharpness), the "retake the photo"
  decision is driven by how many fields actually validated.
- **Verification with a VLM oracle** — with no ground-truth labels, I built a harness
  that scores structural validity over the whole set and uses a vision model as a
  silver-standard on a sample. It surfaced both real bugs and ~15% dataset
  contamination that simple fill-rate metrics had hidden.

## Results (honest, real-world)

**Binary damage detector** (car-level):
| Accuracy | Recall | Precision | F1 |
|---|---|---|---|
| 81.2% | 82.7% | 65.0% | 72.8% |

**Damage localizer** (YOLO11m, off-the-shelf CarDD model, exported to ONNX):
overall **mAP50 0.844**, mAP50-95 0.716 — strongest on shattered-glass (0.994) and
flat-tire (0.959). Earlier self-trained YOLOv8n (CarDD via Roboflow, 11→6 class
remap) reached mAP50 0.672.

**Mulkiya OCR** — labels, dates, plate and VIN extract reliably on clean cards;
numeric fields are OCR-resolution-limited on phone photos. Full accuracy breakdown
and the remediation roadmap are in [`ISSUES.md`](ISSUES.md).

## Tech stack
FastAPI · onnxruntime · RapidOCR (PP-OCR) · EfficientNet-B2 · YOLO11m · OpenCV ·
PyMuPDF · Prometheus · Docker · Kubernetes. CPU-only throughout.

## A note on honesty
This project taught me to distrust flashy metrics: the binary model's early
"95% accuracy / 100% recall" was an artifact of testing on the training
distribution — the honest numbers above are lower and real. See [`ISSUES.md`](ISSUES.md)
for every known limitation and how I'd fix it.
