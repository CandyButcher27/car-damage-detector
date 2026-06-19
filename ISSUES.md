# Models, Accuracy & Known Issues

My work on the UpsureAI internship project (client: **Tameen**, motor-insurance
broker, UAE/KSA). Two parts, both **CPU-only** (≤4 GB), which drives every
model/runtime choice below: a **car damage detector** and a **mulkiya OCR**
pipeline. Accuracy numbers are honest / real-world unless marked otherwise.

> This file documents the parts I built. The wider production service (API routing,
> deployment, the upstream car-vs-not gate) is a private team repo and out of scope
> here. Code references point at the modules in this repo (`mulkiya_ocr/`).

---

# 1. Car Damage Detection

Two stages: a binary damage detector gates a YOLO damage localizer; rule tables add
severity / parts-at-risk / repair per view, then results aggregate to a car-level
verdict.

```
≥1 view (front / back / left / right)
  → Stage 1: binary damage model (EfficientNet-B2, ONNX)  → prob_damaged
       ≤ 0.25 → clean, stop
       > 0.25 → Stage 2: YOLO11m localizer  → boxes {class, confidence, bbox}
                  → severity from bbox-area fraction  (<5% minor / 5–15% moderate / >15% severe)
                  → parts-at-risk  (class, region) rule table
                  → repair action  (class, severity) rule table
  → car-level confidence = max over damaged views
```

## Stage 1 — Binary damage detector (self-trained, knowledge distillation)

**Architecture:** EfficientNet-B2 (~9M params) → ONNX for CPU inference.

**Training — no human labels:**
- A 235B-parameter vision LLM (qwen3-vl, "teacher") auto-labels images damaged/clean.
  (Gemini verification was tried and abandoned: free-tier 20 req/day cap.)
- EfficientNet-B2 ("student") trains on those labels in Colab (T4, ~20 min), with a
  `WeightedRandomSampler` for the ~32% damaged / 68% clean imbalance.
- **Dataset:** 3,519 real-world car images across 969 vehicles.
- **Inference threshold 0.25** to reduce false negatives.
- Two retrain attempts were **abandoned** (overfitting: train 97.7% vs val 71.2%);
  the original model stays.

**Accuracy (car-level — `damage_detected=true` if ANY view flagged):**
| Metric | Value |
|---|---|
| Accuracy | **81.2%** |
| Recall | **82.7%** (51 damaged cars missed) |
| Precision | **65.0%** (131 false positives → manual review) |
| F1 | **72.8%** |
| Confusion | TP 243 · TN 544 · FP 131 · FN 51 |

> Earlier "95% acc / 100% recall" figures were inflated (tested on a subset of the
> training distribution). The table above is the honest real-world evaluation.

## Stage 2 — YOLO damage localizer

**Current model = off-the-shelf, NOT self-trained** (my model-selection decision).
- **Source:** `github.com/ReverendBayes/YOLO11m-Car-Damage-Detector` (MIT).
- **Architecture:** YOLO11m, ~20M params. Trained by the author on CarDD_COCO; I use
  the pretrained checkpoint directly (exported to ONNX). 6 native classes remapped to
  the rule-table keys:

| native class | canonical type | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| shattered_glass | glass-crack | 0.979 | 0.978 | **0.994** | 0.963 |
| flat_tire | flat-tire | 0.943 | 0.919 | 0.959 | 0.932 |
| broken_lamp | lamp-crack | 0.826 | 0.821 | 0.895 | 0.796 |
| scratch | scratches | 0.737 | 0.800 | 0.905 | 0.610 |
| dent | deformation | 0.832 | 0.520 | 0.692 | 0.568 |
| crack | car-part-crack | 0.699 | 0.586 | 0.620 | 0.424 |
| **overall (mean)** | | **0.836** | **0.771** | **0.844** | **0.716** |

(Metrics are the source repo's published model-card values on CarDD.)
- Confidence threshold lowered 0.50 → **0.25**: this model's confidence distribution
  sits lower; at 0.50, damaged views surfaced no boxes.
- Strong: glass-crack, flat-tire, scratches. Weak: dent (recall 0.52) and crack
  (mAP50 0.62) — see Issue **D-1**.

**My earlier self-trained version:** YOLOv8n fine-tuned on **CarDD via Roboflow**
(11 classes remapped to 6, two training rounds), reaching **mAP50 0.672 /
mAP50-95 0.516**. Replaced by the pretrained YOLO11m above (0.672 → 0.844); kept as a
fallback.

**Stages 3–5 (severity / parts / repair):** pure rule tables, not models — severity
from bbox area; parts-at-risk and repair action from `(class, region)` and
`(class, severity)` dictionaries, editable without retraining.

---

# 2. Mulkiya OCR (`mulkiya_ocr/`)

**What:** extract structured fields (plate, VIN, engine cc, weights, seats, dates,
make/model/color/vehicle_type) from photos of UAE/Oman vehicle-registration cards
(mulkiya), Arabic + English.

```
image → card detect + crop/deskew  (card_crop.py)
      → OCR: RapidOCR v3 (PP-OCR det+rec, onnxruntime)
          · dual-pass: English primary + Arabic auxiliary  (ocr_simple_test.py)
      → rule-based field extraction
      → anchor gate + vehicle-spec fingerprint   → is_mulkiya / document_type
      → outcome-based quality gate               → accepted / recapture_message
```

**OCR engine:** migrated **off PaddleOCR/PaddlePaddle** (which crashed in production)
to **`rapidocr` v3** — PP-OCR on onnxruntime, no PaddlePaddle, no PyTorch. Arabic
recognition auto-downloads (`arabic_PP-OCRv4_rec_mobile.onnx`). Default `ocr_lang=en`
(verified: recovers Arabic text fields at usable rates with identical numeric
accuracy; Arabic-primary left them at 0%).

### Accuracy / verification (no ground-truth labels → harness + VLM oracle)
Measured on a 145-image real-world set:
- **Dataset is ~15% contaminated** — of 138 processed: 117 mulkiya, 16 other,
  5 driving-licence. The anchor gate flags all of them.
- **Field fill** (anchor-gated, en-primary): plate 98.6%, VIN 97.8%, seats 97.8%,
  year 95.7%, engine_cc 89.1%, weights 81.2%, expiry 85.5%, color 78.3%,
  vehicle_type 43.5%, model 38.4%, make 4.3%.
- **High fill ≠ correct:** ~20% of VINs are format-invalid; numeric fields
  (cc/weights/plate/year) are OCR-resolution-limited on WhatsApp-grade photos
  (~5/13 fields correct even on a clean single card). See Issues **M-1/M-2**.

**Shipped:** RapidOCR migration; anchor gate + vehicle-spec fingerprint (rejects
non-mulkiya); outcome-based **quality / re-capture gate**; **quality-triggered crop
fallback** (deskew/orient, only on failure); `ocr_lang` default `en`.

> Note on the re-capture gate: pixel-blur (Laplacian) and OCR confidence both proved
> UNRELIABLE (a clean scan scored the lowest blur; OCR is confident even on garbage),
> so the gate is **outcome-based** — it counts validly-extracted critical fields.

---

# 3. Known Issues & Remediation

Severity 🔴 high · 🟠 medium · ⚪ low.

| ID | Area | Issue | Sev | Status |
|---|---|---|---|---|
| D-1 | Damage | YOLO weak on dent/crack classes | 🟠 | Known (model ceiling) |
| M-1 | Mulkiya | Numeric fields wrong (cc/weights/plate/year) | 🔴 | Fix prototyped, not integrated |
| M-2 | Mulkiya | Spatial binding not wired into main path | 🔴 | Prototype only |
| M-3 | Mulkiya | PDFs → raw OCR, no structured extract | 🟠 | Not done |
| M-4 | Mulkiya | `verify_mulkiya_batch.py` doesn't stream | ⚪ | Not done |
| M-7 | Mulkiya | `make` field ~4% fill | ⚪ | Arabic OCR floor |

### D-1 · YOLO weak on `dent`/`crack` 🟠
**Symptom:** dent→deformation recall 0.52; crack→car-part-crack mAP50 0.62 (vs 0.99
for glass). **Cause:** inherent to the off-the-shelf model on those classes.
**Fix:** acceptable for an assistive tool; if needed, fine-tune YOLO11m on extra
dent/crack data (Roboflow) or ensemble. **Blocked on:** data + a retrain decision.

### M-1 · Mulkiya numeric fields wrong 🔴
**Symptom:** cc/weights/plate/year frequently wrong on real photos (e.g. cc
1295→350, year 2015→2025). **Cause (two):** (1) **binding** — the extractor flattens
OCR to strings and discards boxes, so it grabs the wrong column on the 2-column
layout; (2) **digit OCR floor** — small-font digits below the recognizer's
resolution. **Fix:** **positional template** — on the normalised (cropped/deskewed)
card each field sits at a known relative (x%,y%) region; crop it, **3× upscale**, OCR
with a digit-focused pass. Upscaling is prototyped and already lifts the digit floor
(`prototype_spatial.py::_read_digits_upscaled`). **Files:** `ocr_simple_test.py`
(extraction), `prototype_spatial.py`, `card_crop.py`.

### M-2 · Spatial binding not wired into the main path 🔴
**Symptom:** the proven fix for year/color/vehicle_type lives only in the prototype.
**Cause:** the rule-based extractor consumes text lines only; OCR boxes are dropped.
**Fix:** thread boxes through, cluster into rows (RTL), bind value→label by geometry
(port `prototype_spatial.py::extract_spatial`); keep the flat path as a fallback.
Verified win: year 2025→**2015**.

### M-3 · Mulkiya PDFs return raw OCR only 🟠
**Symptom:** PDF input yields OCR lines, no structured JSON (~6/145 docs). **Cause:**
structured extraction runs only on the image branch. **Fix:** rasterise the first PDF
page (PyMuPDF, already a dep) → run the image extraction path.

### M-4 · `verify_mulkiya_batch.py` doesn't stream ⚪
**Symptom:** writes results only at the end; a kill loses the aggregate. (The sibling
`prototype_eval_batch.py` already streams.) **Fix:** mirror its per-row
`writerow`+`flush`.

### M-7 · `make` field ~4% fill ⚪
**Symptom:** brand (تويوتا = Toyota) rarely extracted. **Cause:** the Arabic
recognizer garbles brand names (تويوتا→كووتا). **Fix:** expand the garbled-variant
keyword list (brittle) or a learned doc→JSON model (Donut / small VLM) on
field-labeled mulkiyas. Low priority.

## Recommended order
1. **M-1** positional template + upscaling (biggest mulkiya accuracy lever)
2. **M-2** wire spatial binding (guaranteed win: year/color/type)
3. **M-3** PDF structured extraction · **M-4** stream the harness
4. **D-1** only if the assistive YOLO accuracy proves insufficient

## Files in this repo (`mulkiya_ocr/`)
- `card_crop.py` — card detect + deskew + orientation + tight-card pick.
- `ocr_simple_test.py` — dual-pass Arabic+English OCR engine + rule-based extraction.
- `prototype_spatial.py` — box-aware spatial binding + cell upscaling (M-1/M-2 fix).
- `prototype_eval_batch.py` — crop + best-of + rejection batch eval (streams).
- `verify_mulkiya_batch.py` — structural verification harness (VLM-oracle scored).

---

## Tech stack & constraints
CPU-only, 4 GB. onnxruntime for all models (damage binary, YOLO localizer, mulkiya
OCR). `rapidocr` v3 for OCR (no PaddlePaddle, no PyTorch). PyMuPDF for PDF. Do **not**
reintroduce PaddlePaddle, GPU-only models, or HuggingFace bounding-box datasets (use
Roboflow).
