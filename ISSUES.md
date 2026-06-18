# data-ingestion — Models, Accuracy & Known Issues

**Service:** UpsureAI `data-ingestion` — a single FastAPI app that ingests car
images and UAE/Oman vehicle documents, runs ML + OCR, and returns structured JSON.
**Client:** Tameen (motor-insurance broker, UAE/KSA). **Constraint:** CPU-only,
~2–4 vCPU / 4 GB pod (drives every model/runtime choice below).
**Last updated:** 2026-06-18.

> **How to read this file (for humans and AI agents):** it has three capability
> sections — **Car Detection**, **Car Damage Detection**, **Mulkiya OCR** — each
> with what it does, how the models were trained, and measured accuracy. The final
> **Known Issues & Remediation** section lists every open problem with a concrete
> fix and whether it is blocked on data. Accuracy numbers are honest/real-world
> unless marked otherwise. File/function references use `path::symbol`.

## Endpoints (overview)
| Endpoint | Purpose |
|---|---|
| `POST /predict/` | Direct car classification |
| `POST /predict/damage` | Car damage detection (front/back/left/right) |
| `POST /api/v1/process` | Unified router: `process_type` ∈ car / mulkiya / pdf / file |
| `GET /livez` `/readyz` `/health` `/metrics` | k8s probes + Prometheus |

All responses use the envelope `{success, data, error, meta}`.

---

# 1. Car Detection

**What:** A binary "is this a car?" image classifier. Used in two places:
(a) the `/predict/damage` **car gate** — drops non-car views before damage
inference; (b) `process_type=car` on `/api/v1/process` — direct classification.

**Model:** `best_car_model_v2.onnx` (`digiLifeDoc_best_car_model_v2.onnx`), run via
`onnx_inference.py::BinaryOnnxImageClassifier`. ONNX preferred; a `.keras` file is
auto-detected as fallback.

**Threshold:** `UPSURE_CAR_THRESHOLD` default **0.65** (raised from 0.35 to cut
false "car" positives on document photos).

**Accuracy:** ⚠️ **Not formally documented.** No precision/recall/accuracy figures
are recorded for this classifier. See Known Issue **C-1**.

---

# 2. Car Damage Detection (`POST /predict/damage`)

Two-stage pipeline: a binary damage detector gates a YOLO damage localizer. Damage
and the rule-based enrichment run per view; results aggregate to a car-level verdict.

### Pipeline flow
```
multipart front / back / left / right  (≥1 required)
│
├─ Phase 1: read bytes, decode each to JPEG  (PDF → first page)
│
├─ Car gate: best_car_model on every view → drop non-car views (skipped_views)
│            (no car views → "No car images. Please submit images of a vehicle.")
│
├─ Phase 2 (batched, single ONNX call over car views):
│   per view → Stage 1 binary damage model (damage_model.onnx, EfficientNet-B2)
│             prob_damaged
│              ├─ ≤ 0.25 → view clean, stop
│              └─ > 0.25 → Stage 2 YOLO localizer (damage_detector_v3.onnx, YOLO11m)
│                          → boxes {class, confidence, bbox}   (conf ≥ 0.25, IoU 0.45)
│                          → severity from bbox-area fraction:
│                               < 5% minor · 5–15% moderate · > 15% severe
│                          → parts-at-risk  (class, region) rule table
│                          → repair action  (class, severity) rule table
│                          → fallback: binary positive but YOLO empty → "general-damage"
│
└─ overall_confidence = max(confidence_score) across DAMAGED views
```
Key constants (`poc_api.py`): `DAMAGE_THRESHOLD=0.25`, `YOLO_CONF=0.25`,
`YOLO_IOU=0.45`, severity cuts `0.05` / `0.15`.

## Stage 1 — Binary damage detector (self-trained)

**Architecture:** EfficientNet-B2 (~9M params) → exported to ONNX for CPU inference.

**How it was trained — knowledge distillation:**
- A 235B-parameter vision LLM (qwen3-vl, "teacher") auto-labels the images
  damaged/clean — no human annotation. (Gemini verification was tried and
  abandoned: free-tier 20 req/day cap.)
- EfficientNet-B2 ("student") trains on those labels in Google Colab (T4, ~20 min),
  with `WeightedRandomSampler` for the ~32% damaged / 68% clean imbalance.
- **Dataset:** 3,519 real-world car images across 969 vehicles.
- **Inference threshold 0.25** (post-processing) to reduce false negatives.
- Two retrain attempts were **abandoned** (persistent overfitting: train 97.7% vs
  val 71.2%); the original model stays in production.

**Accuracy (car-level, honest — `damage_detected=true` if ANY view flagged):**
| Metric | Value |
|---|---|
| Accuracy | **81.2%** |
| Recall | **82.7%** (51 damaged cars missed) |
| Precision | **65.0%** (131 false positives → manual review) |
| F1 | **72.8%** |
| Confusion | TP 243 · TN 544 · FP 131 · FN 51 |

> Note: earlier "95% acc / 100% recall" figures were inflated (tested on a subset of
> the training distribution). The table above is the honest real-world evaluation.

## Stage 2 — YOLO damage localizer

**Current production model = v3 (off-the-shelf, NOT self-trained).**
- **Source:** `github.com/ReverendBayes/YOLO11m-Car-Damage-Detector` (MIT license).
- **Architecture:** YOLO11m, ~20M params, 231 layers.
- **Trained by the author on** CarDD_COCO. We **use the pretrained checkpoint
  directly** (exported to `damage_detector_v3.onnx`); we did not train v3.
- 6 native classes are **remapped** to the pipeline's canonical rule-table keys:

| v3 native class | canonical (API `type`) | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| shattered_glass | glass-crack | 0.979 | 0.978 | **0.994** | 0.963 |
| flat_tire | flat-tire | 0.943 | 0.919 | 0.959 | 0.932 |
| broken_lamp | lamp-crack | 0.826 | 0.821 | 0.895 | 0.796 |
| scratch | scratches | 0.737 | 0.800 | 0.905 | 0.610 |
| dent | deformation | 0.832 | 0.520 | 0.692 | 0.568 |
| crack | car-part-crack | 0.699 | 0.586 | 0.620 | 0.424 |
| **overall (mean)** | | **0.836** | **0.771** | **0.844** | **0.716** |

(Overall = mean across classes, i.e. YOLO's "all" row. Metrics are the source
repo's published model-card values on CarDD.)
- `UPSURE_YOLO_CONF` lowered 0.50 → **0.25**: v3's confidence distribution sits
  lower than v2; at 0.50 damaged views surfaced no boxes.
- Strong: glass-crack, flat-tire, scratches. Weaker: `dent`/deformation (low recall
  0.52) and `crack`/car-part-crack (mAP50 0.62) — see Known Issue **D-1**.

**Previous versions (self-trained, history):** v2 = YOLOv8n fine-tuned by us on
**CarDD via Roboflow** (11 classes remapped to 6, two training rounds), reaching
**mAP50 0.672 / mAP50-95 0.516**. v3 (the MIT pretrained YOLO11m above) replaced it
and improved overall mAP50 0.672 → **0.844**. v2 is kept as an auto-detected fallback.

**Stages 3–5 (severity / parts / repair):** pure rule tables (not models) — severity
from bbox area; parts-at-risk and repair action from `(class, region)` and
`(class, severity)` dictionaries, editable by Tameen without retraining.

---

# 3. Mulkiya OCR Pipeline (`process_type=mulkiya`)

**What:** Extract structured fields (plate, VIN, engine cc, weights, seats, dates,
make/model/color/vehicle_type) from photos of UAE/Oman vehicle-registration cards
(mulkiya), Arabic + English.

### Flow
```
image → card/non-card classifier + mulkiya front/back classifier
      → OCR worker (subprocess): RapidOCR v3 (PP-OCR det+rec, onnxruntime only)
          · dual-pass: English primary + Arabic auxiliary (merge text fields)
      → rule-based field extraction (_extract_mulkya_rulebased)
      → anchor gate + vehicle-spec fingerprint  → document_type / is_mulkiya
      → outcome-based quality gate              → accepted / recapture_message
      → quality-triggered crop fallback (deskew/orient/crop, retry on failure)
```

**OCR engine:** migrated **off PaddleOCR/PaddlePaddle** (which crashed in
production) to **`rapidocr` v3** — PP-OCR on onnxruntime, no PaddlePaddle, no
PyTorch. Arabic recognition auto-downloads (`arabic_PP-OCRv4_rec_mobile.onnx`).
Default `ocr_lang=en` (verified: recovers Arabic text fields at usable rates with
identical numeric accuracy; `ar`-primary left them at 0%).

### Accuracy / verification (no ground-truth labels → harness + VLM oracle)
Measured on a 145-image real-world set (`Documents_Mulkiya_Front_*.zip`):
- **Dataset is ~15% contaminated** — of 138 processed: 117 mulkiya, 16 other,
  5 driving-licence. The anchor gate flags all of them.
- **Field fill** (anchor-gated, en-primary): plate 98.6%, VIN 97.8%, seats 97.8%,
  year 95.7%, engine_cc 89.1%, weights 81.2%, expiry 85.5%, color 78.3%,
  vehicle_type 43.5%, model 38.4%, make 4.3%.
- **High fill ≠ correct:** ~20% of VINs are format-invalid; numeric fields
  (cc/weights/plate/year) are OCR-resolution-limited on WhatsApp-grade photos
  (~5/13 fields correct even on a clean single card). See Known Issues **M-1/M-2**.

**What is already shipped (integrated):** RapidOCR migration; anchor gate + vehicle-
spec fingerprint (rejects non-mulkiya); outcome-based **quality / re-capture gate**
(`classification.accepted`, `recapture_message` — e.g. "image not clear, retake");
**quality-triggered crop fallback** (deskew/orient, only on failure); `color` fix;
`ocr_lang` default `en`.

> Note on the re-capture gate: pixel-blur (Laplacian) and OCR confidence both proved
> UNRELIABLE (a clean scan scored the lowest blur; OCR is confident even on garbage),
> so the gate is **outcome-based** — it counts validly-extracted critical fields.

---

# 4. Known Issues & Remediation

Severity 🔴 high · 🟠 medium · ⚪ low. "Blocked on YOU" = needs data/decision from the team.

## Quick status
| ID | Area | Issue | Sev | Status | Blocked on |
|---|---|---|---|---|---|
| D-1 | Damage | YOLO weak on dent/crack classes | 🟠 | Known | model ceiling |
| C-1 | Car | Car classifier accuracy undocumented | 🟠 | Open | measure it |
| M-1 | Mulkiya | Numeric fields wrong (cc/weights/plate/year) | 🔴 | Pieces built, not integrated | — (do it) |
| M-2 | Mulkiya | Spatial binding not wired in | 🔴 | Prototype only | — (do it) |
| M-3 | Mulkiya | PDFs → raw OCR, no structured extract | 🟠 | Not done | — (do it) |
| M-4 | Mulkiya | `verify_mulkiya_batch.py` doesn't stream | ⚪ | Not done | — (do it) |
| M-5 | Mulkiya | Front/back classifier polarity inverted | 🔴 | Diagnosed | **YOU: labeled data** |
| M-6 | Mulkiya | Card/non-card classifier unreliable | 🟠 | Diagnosed | **YOU: data/retrain** |
| M-7 | Mulkiya | `make` field ~4% fill | ⚪ | OCR floor | hard, low priority |

### D-1 · YOLO weak on `dent`/`crack` 🟠
**Symptom:** `dent`→deformation recall 0.52; `crack`→car-part-crack mAP50 0.62 (vs
0.99 for glass). **Cause:** inherent to the off-the-shelf v3 model on those classes.
**Fix:** acceptable for an assistive tool; if needed, fine-tune the YOLO11m on extra
dent/crack data (Roboflow), or ensemble. **Blocked on:** data + a retrain decision.

### C-1 · Car classifier accuracy undocumented 🟠
**Symptom:** No precision/recall/accuracy recorded for `best_car_model_v2.onnx`.
**Fix:** evaluate on a held-out labeled car/non-car set; record like Stage-1 damage.
**Blocked on:** nothing (just run an eval).

### M-1 · Mulkiya numeric fields wrong 🔴
**Symptom:** cc/weights/plate/year frequently wrong on real photos (e.g. cc
1295→350, year 2015→2025). **Cause (two):** (1) **binding** — the extractor flattens
OCR to strings and discards boxes, so it grabs the wrong column on the 2-column
layout; (2) **digit OCR floor** — small-font digits below the recognizer's
resolution. **Fix:** **positional template** — on the normalised (cropped/deskewed)
card, each field is at a known relative (x%,y%) region; crop it, **3× upscale**,
OCR with a digit-focused pass. Upscaling is prototyped and already lifts the digit
floor (`prototype_spatial.py::_read_digits_upscaled`). **Blocked on:** nothing.
**Files:** `ocr_simple_test.py::_extract_mulkya_rulebased`, `prototype_spatial.py`,
`card_crop.py`.

### M-2 · Spatial binding not wired in 🔴
**Symptom:** Proven fix for year/color/vehicle_type lives only in the prototype.
**Cause:** `_extract_mulkya_rulebased(lines: list[str])` consumes text only; boxes
are dropped by `_run_ocr`. **Fix:** thread boxes through, cluster into rows (RTL),
bind value→label by geometry (port `prototype_spatial.py::extract_spatial`); keep
flat path as fallback. Verified win: year 2025→**2015**. **Blocked on:** nothing.

### M-3 · Mulkiya PDFs return raw OCR only 🟠
**Symptom:** PDF input yields OCR lines, no structured `_mulkya.json` (~6/145 docs).
**Cause:** `--extract_mulkya` runs only in the image branch of `main()`. **Fix:**
rasterise the first PDF page (PyMuPDF, already a dep) → run the image extraction
path. **Blocked on:** nothing. **Files:** `ocr_simple_test.py` PDF branch, `poc_api.py`.

### M-4 · `verify_mulkiya_batch.py` doesn't stream ⚪
**Symptom:** Writes results only at the end; a kill loses the aggregate. (The
integrated `prototype_eval_batch.py` was already fixed to stream.) **Fix:** mirror
its per-row `writerow`+`flush`. **Blocked on:** nothing.

### M-5 · Front/back classifier polarity inverted 🔴
**Symptom:** `digiLifeDoc_mulkiya_classifier_model.onnx` calls a clear front a
"back" (front_prob ≈ 0.003), inconsistent across images. **Cause:** output index
likely swapped (`[front, back]`), plus a weak model on real photos. **Fix:** with a
labeled front+back set, check the index and set `UPSURE_MULKIYA_FRONT_HIGH` (env,
**no code change**); retrain if still poor. **Blocked on:** **YOU** — the current zip
is all fronts, so polarity can't be validated. **Files:** `poc_api.py::_get_mulkiya_classifier`.

### M-6 · Card/non-card classifier unreliable 🟠
**Symptom:** Returned `card_prob=0.000` on a real mulkiya next to an ID; also passes
driving licences (a licence is a card). **Cause:** trained on clean single-card crops;
confused by clutter; cannot tell mulkiya from other cards. **Fix:** short term — rely
on the **anchor gate + fingerprint** (shipped) to decide "is this a mulkiya"; long
term — train a `mulkiya/licence/id/other` classifier (Roboflow). **Blocked on:**
**YOU** — labeled multi-class data.

### M-7 · `make` field ~4% fill ⚪
**Symptom:** Brand (تويوتا = Toyota) rarely extracted. **Cause:** Arabic recognizer
garbles brand names (تويوتا→كووتا). **Fix:** expand garbled-variant keyword list
(brittle) or a learned doc→JSON model (Donut / small VLM) on field-labeled mulkiyas.
Low priority. **Blocked on:** better Arabic OCR or a learned extractor.

## Recommended order (unblocked)
1. **M-1** positional template + upscaling (biggest mulkiya accuracy lever)
2. **M-2** wire spatial binding (guaranteed win: year/color/type)
3. **M-3** PDF structured extraction
4. **C-1** measure car-classifier accuracy · **M-4** stream the harness

Then, once data is provided: **M-5** (flip front/back), **M-6** (train mulkiya/licence
classifier). **D-1** only if the assistive YOLO accuracy proves insufficient.

## Prototype files (standalone — fold into the pipeline, then deletable)
- `card_crop.py` — card detect + deskew + orientation + tight-card pick (runtime dep of the crop fallback).
- `prototype_spatial.py` — box-aware spatial binding + cell upscaling.
- `prototype_eval_batch.py` — crop + best-of + rejection batch eval (streams).
- `verify_mulkiya_batch.py` — Tier-1 structural verification harness.

---

## Reference: tech stack & constraints
CPU-only, 4 GB. onnxruntime for all models (car, card, damage binary, YOLO,
mulkiya-side). `rapidocr` v3 for OCR (no PaddlePaddle, no PyTorch). `tensorflow-cpu`
only as a legacy `.keras` fallback loader. PyMuPDF for PDF. Do **not** reintroduce
PaddlePaddle, GPU-only models, or HuggingFace bounding-box datasets (use Roboflow).
