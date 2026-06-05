# UpsureAI — Car Damage Detection

## Project
Intern project at UpsureAI (Hyderabad, Summer 2026). Car damage detection system for Tameen (UAE/KSA motor insurance broker). Input: 4 car views (front/back/left/right). Output: damage detected + severity + parts + repair estimate.

## Current State
- **v1 (production):** Binary damage classifier — `models/damage_model_single.onnx`
  - EfficientNet-B2, ONNX, CPU inference
  - 82.7% recall, 81.2% accuracy on 969 cars (car-level eval, threshold=0.25)
  - This is the honest baseline — do not inflate
- **v2 (in progress):** Severity + parts + repair pipeline
  - YOLOv8n fine-tuned on CarDD (6 classes: dent, scratch, crack, glass_shatter, tire_flat, lamp_broken)
  - YOLO notebook: `train_yolo_colab.ipynb`
  - Parts + repair: rule-based (draft in Obsidian: UpSureAI_severity_pipeline.md)

## Key Paths
- `images/`         — 3519 training images, 969 cars (gitignored)
- `labels/`         — 969 JSON files, VLM-generated (qwen3-vl:235b)
- `dataset.csv`     — 3520 rows, 31.2% damaged (gitignored)
- `models/`         — ONNX + PTH weights (gitignored)
- `config.yaml`     — single source of truth for all paths/thresholds
- `api.py`          — FastAPI, POST /predict/damage, multipart FormData
- `filter_new_data.py` — extract zips → car/card classifier → copy to images/
- `batch_label.py`  — VLM labelling (qwen3-vl:235b via Ollama)
- `prepare_dataset.py` — labels/ + images/ → dataset.csv

## Data Pipeline (new batch)
1. Drop zip in project root
2. `python filter_new_data.py --apply`
3. `python batch_label.py`
4. `python prepare_dataset.py`
5. Retrain on Colab

## Constraints
- CPU-only inference: i5-1135G7 production server
- No Gemini (dropped — 20 req/day free tier)
- No damage-type classifier (v2 scope, deleted)

## Notebook Rules
- Every `.ipynb` that exports ONNX must install: `!pip install onnx onnxscript -q`
- Include this in the same pip install cell as other deps

## Git Rules
- Never add `Co-Authored-By` or any AI authorship line to commit messages
- Feature branches: `feat/<short-description>`
- Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `chore:`
