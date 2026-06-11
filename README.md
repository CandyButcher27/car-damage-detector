# Car Damage Detection API

FastAPI service that classifies vehicle photos and detects body damage. A single
endpoint runs each uploaded view through a three-stage ONNX pipeline.

## Pipeline

For every view (in order, each view independent):

1. **`best_car_model`** — is the image a car? If **not**, the view is **skipped**
   (`reason: not_a_car`) and no damage steps run.
2. **`damage_model`** — classifies the car as **damaged** or **clean**.
3. **`damage_detector`** (YOLO) — runs **only when the classifier flags damage**, to
   locate and confirm it with bounding boxes.

A non-car view does not fail the request; other views are still analyzed.

## Models

Place the ONNX models in `models/` (env vars override the path):

| Model            | File                                     | Env var                       |
| ---------------- | ---------------------------------------- | ----------------------------- |
| Car classifier   | `digiLifeDoc_best_car_model_v2.onnx`     | `UPSURE_CAR_MODEL`            |
| Damage classifier| `digiLifeDoc_damage_model.onnx`          | `UPSURE_DAMAGE_MODEL`         |
| Damage detector  | `digiLifeDoc_damage_detector_v2.onnx`    | `UPSURE_DAMAGE_DETECTOR_MODEL`|

## Run

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
python poc_api.py             # serves on :8000 (PORT env to override)
```

Docker:

```bash
docker build -t car-damage .
docker run -p 8000:8000 car-damage
```

## Endpoints

### `GET /health`
Reports load status of all three models.

### `POST /predict/damage`
Multipart form. Provide 1–4 named views, or a single image via `file`/`image`/`upload`.

Fields: `front`, `back`, `left`, `right` (each an image or PDF; first PDF page is used).

```bash
curl -X POST http://localhost:8000/predict/damage \
  -F "front=@Samples/car_1001.jpg" \
  -F "left=@Samples/car_1003.jpg"
```

Response:

```json
{
  "damage_detected": false,
  "total_views_received": 2,
  "views_analyzed": 2,
  "views_skipped": 0,
  "overall_confidence": 0.12,
  "per_view": {
    "front": {
      "status": "analyzed",
      "car": { "is_car": true, "confidence": 0.98, "car_probability": 0.98 },
      "damage_detected": false,
      "damage_confidence": 0.12,
      "classifier": { "damage_detected": false, "prob_damaged": 0.12 },
      "detector": null
    },
    "left": { "status": "skipped", "reason": "not_a_car", "car": { "is_car": false } }
  }
}
```

- `status` is `analyzed` or `skipped`.
- `detector` is `null` when the classifier finds no damage (detector not run).
- `damage_detected` (top level) is true if any analyzed view is damaged.
