# Car Damage Detection + Mulkiya OCR

> My personal write-up of an internship project at **UpsureAI** (Summer 2026,
> Hyderabad; client: **Tameen**, a motor-insurance broker in the UAE/KSA). The
> production service is a team effort on a private company GitLab. **This repo is
> the part I built** — the damage-detection model and the mulkiya OCR — packaged
> as a standalone, CPU-only write-up of the engineering.

The goal of the wider system: ingest car photos and UAE/Oman vehicle-registration
cards, run ML + OCR on CPU, and return structured JSON — cutting insurance claim
processing from ~5–7 days toward ~24 hours. My two contributions:

## Phase 1 — Car damage model (knowledge distillation)

A binary "is this car damaged?" detector, trained without a single hand label.

- **Teacher:** a 235B-parameter vision LLM (qwen3-vl) auto-labels real car images
  damaged/clean. (A Gemini cross-check was tried and dropped — free-tier 20 req/day.)
- **Student:** EfficientNet-B2 (~9M params) trains on those labels in Colab (T4,
  ~20 min), with a `WeightedRandomSampler` for the ~32% damaged / 68% clean split.
- **Data:** 3,519 real-world car images across 969 vehicles.
- **Export:** ONNX, for CPU-only inference. Inference threshold 0.25 to favor recall.

**Honest accuracy (car-level — flagged if any view shows damage):**

| Accuracy | Recall | Precision | F1 |
|---|---|---|---|
| **81.2%** | **82.7%** | **65.0%** | **72.8%** |

Confusion: TP 243 · TN 544 · FP 131 · FN 51. Two retrain attempts were **abandoned**
— persistent overfitting (train 97.7% vs val 71.2%), both worse than the baseline,
so the original model stayed.

**Damage localizer (stage 2):** I first self-trained a **YOLOv8n** on CarDD (via
Roboflow), remapping 11 classes to 6, reaching **mAP50 0.672**. I then swapped in an
off-the-shelf **YOLO11m** (ReverendBayes, MIT, trained on CarDD) — *not my training,
my model-selection call* — lifting overall **mAP50 to 0.844** (mAP50-95 0.716).
Strongest on shattered-glass (0.994) and flat-tire (0.959); weakest on dent (recall
0.52) and crack (mAP50 0.62). Severity / parts-at-risk / repair are pure rule tables
on top, editable without retraining.

## Phase 2 — Mulkiya OCR (`mulkiya_ocr/`)

Reading UAE/Oman vehicle-registration cards (Arabic + English) into structured
fields, CPU-only.

- **Replaced PaddleOCR/PaddlePaddle with RapidOCR** (PP-OCR on onnxruntime) after
  PaddlePaddle kept crashing the production container — no PaddlePaddle, no PyTorch.
- **Dual-pass Arabic + English OCR**, since one recognizer can't read both scripts
  well. (Measured: English-primary recovers Arabic fields at usable rates with
  identical numeric accuracy; Arabic-primary left them at 0%.)
- **Document gating without a bespoke model** — an OCR **anchor gate + vehicle-spec
  fingerprint** rejects non-mulkiya documents (a driving licence *is* a card, so a
  card classifier alone can't catch it).
- **Outcome-based image-quality gate** — instead of pixel-blur (which I measured to
  be unreliable: a clean scan scored the *lowest* sharpness), the "retake the photo"
  decision is driven by how many fields actually validated.
- **Verification with a VLM oracle** — with no ground-truth labels, I built a harness
  (`verify_mulkiya_batch.py`) that scores structural validity over the whole set and
  uses a vision model as a silver standard on a sample. It surfaced real bugs **and
  ~15% dataset contamination** that simple fill-rate metrics had hidden.

**Field extraction, real-world (145-image set, ~15% contaminated, anchor-gated):**
plate 98.6% · VIN 97.8% · seats 97.8% · year 95.7% · engine_cc 89.1% · weights
81.2% · expiry 85.5% · color 78.3% — make/model lower (Arabic brand OCR floor). High
fill is not the same as correct: ~20% of VINs are format-invalid and small-font
digits are OCR-resolution-limited on phone photos. The fixes (positional template +
3× digit upscaling, box-aware spatial binding) are prototyped in `prototype_spatial.py`;
the full breakdown and roadmap are in [`ISSUES.md`](ISSUES.md).

## Tech stack
FastAPI · onnxruntime · RapidOCR (PP-OCR) · EfficientNet-B2 · YOLO11m · OpenCV ·
PyMuPDF. CPU-only throughout.

## A note on honesty
This project taught me to distrust flashy metrics: the damage model's early
"95% accuracy / 100% recall" was an artifact of testing on the training
distribution — the honest numbers above are lower and real. Every known limitation,
with a concrete fix, is in [`ISSUES.md`](ISSUES.md).
