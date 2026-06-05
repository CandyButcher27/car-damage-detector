"""
Car Damage Detection API — v2
------------------------------
5-stage pipeline:
  Stage 1: EfficientNet-B2 binary detection (damage / clean)
  Stage 2: YOLOv8n damage localization + classification
  Stage 3: Severity from normalized bbox area
  Stage 4: Parts at risk (rule-based)
  Stage 5: Repair action (rule-based)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/predict \
         -F "front=@front.jpg" -F "back=@back.jpg"
"""

import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from config import CFG
import parts_rules
import repair_rules

_A               = CFG["api"]
_P               = CFG["paths"]
_Y               = CFG["yolo"]

ONNX_PATH        = _P["model_onnx"]
YOLO_PATH        = _P["yolo_onnx"]
IMG_SIZE         = _A["img_size"]
DAMAGE_THRESHOLD = _A["damage_threshold"]
MEAN             = np.array(_A["mean"], dtype=np.float32)
STD              = np.array(_A["std"],  dtype=np.float32)
YOLO_SIZE        = _Y["img_size"]
YOLO_CONF        = _Y["conf_thresh"]
YOLO_IOU         = _Y["iou_thresh"]
YOLO_CLASSES     = _Y["classes"]

app = FastAPI(title="Car Damage Detector", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Model loading ─────────────────────────────────────────────────────────────

try:
    _binary_sess  = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    _binary_input = _binary_sess.get_inputs()[0].name
except Exception as e:
    _binary_sess = None
    print(f"WARNING: binary model not loaded: {e}")

try:
    _yolo_sess  = ort.InferenceSession(YOLO_PATH, providers=["CPUExecutionProvider"])
    _yolo_input = _yolo_sess.get_inputs()[0].name
except Exception as e:
    _yolo_sess = None
    print(f"WARNING: YOLO model not loaded: {e}")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_binary(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def preprocess_yolo(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((YOLO_SIZE, YOLO_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]


# ── NMS ───────────────────────────────────────────────────────────────────────

def _nms(x1, y1, x2, y2, scores, iou_thresh):
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1  = np.maximum(x1[i], x1[order[1:]])
        yy1  = np.maximum(y1[i], y1[order[1:]])
        xx2  = np.minimum(x2[i], x2[order[1:]])
        yy2  = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


# ── Stage 1: binary inference ─────────────────────────────────────────────────

def run_binary(arr: np.ndarray) -> dict:
    logits = _binary_sess.run(None, {_binary_input: arr})[0][0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    label  = 1 if probs[1] >= DAMAGE_THRESHOLD else 0
    return {
        "damage_detected":  bool(label == 1),
        "confidence_score": float(round(float(probs[label]), 4)),
        "prob_damaged":     float(round(float(probs[1]), 4)),
        "prob_clean":       float(round(float(probs[0]), 4)),
    }


# ── Stage 2-5: YOLO + rules ───────────────────────────────────────────────────

def run_yolo_pipeline(img_bytes: bytes) -> list[dict]:
    if _yolo_sess is None:
        return []

    arr    = preprocess_yolo(img_bytes)
    output = _yolo_sess.run(None, {_yolo_input: arr})[0]  # (1, 4+nc, 8400)
    preds  = output[0].T                                   # (8400, 4+nc)

    boxes        = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids    = np.argmax(class_scores, axis=1)
    max_scores   = np.max(class_scores, axis=1)

    mask         = max_scores >= YOLO_CONF
    boxes        = boxes[mask]
    class_ids    = class_ids[mask]
    max_scores   = max_scores[mask]

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    keep = _nms(x1, y1, x2, y2, max_scores, YOLO_IOU)

    detections = []
    for i in keep:
        cls_name   = YOLO_CLASSES[int(class_ids[i])]
        bbox_px    = [float(boxes[i, 0]), float(boxes[i, 1]),
                      float(boxes[i, 2]), float(boxes[i, 3])]
        bbox       = [v / YOLO_SIZE for v in bbox_px]  # normalize to [0,1]

        # Stage 3: severity — normalized bbox area vs image (car bbox detection TODO: Stage 0)
        severity   = repair_rules.get_severity(bbox[2] * bbox[3], 1.0)

        # Stage 4: parts
        region     = parts_rules.get_image_region(bbox, [0.5, 0.5, 1.0, 1.0])
        parts      = parts_rules.lookup(cls_name, region)

        # Stage 5: repair
        repair     = repair_rules.lookup(cls_name, severity)

        detections.append({
            "type":          cls_name,
            "confidence":    float(round(float(max_scores[i]), 4)),
            "severity":      severity,
            "bbox":          [round(v, 4) for v in bbox],
            "parts_at_risk": parts,
            "repair_action": repair["action"],
            "replace":       repair["replace"],
        })

    return detections


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "binary_model": _binary_sess is not None,
        "yolo_model":   _yolo_sess is not None,
    }


@app.post("/predict")
async def predict(
    front: UploadFile | None = File(default=None),
    back:  UploadFile | None = File(default=None),
    left:  UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
):
    views = {k: v for k, v in [("front", front), ("back", back), ("left", left), ("right", right)] if v}

    if not views:
        raise HTTPException(status_code=400, detail="Provide at least one image (front/back/left/right).")
    if _binary_sess is None:
        raise HTTPException(status_code=503, detail="Binary model not loaded.")

    per_view        = {}
    overall_damaged = False
    max_confidence  = 0.0

    for view_name, upload in views.items():
        img_bytes = await upload.read()

        # Stage 1: binary
        arr    = preprocess_binary(img_bytes)
        binary = run_binary(arr)

        view_result = {**binary, "damages": []}

        # Stages 2-5: only if damage detected
        if binary["damage_detected"] and _yolo_sess is not None:
            view_result["damages"] = run_yolo_pipeline(img_bytes)

        if binary["damage_detected"]:
            overall_damaged = True
        if binary["confidence_score"] > max_confidence:
            max_confidence = binary["confidence_score"]

        per_view[view_name] = view_result

    return JSONResponse({
        "damage_detected":      overall_damaged,
        "total_views_analyzed": len(views),
        "overall_confidence":   round(max_confidence, 4),
        "per_view":             per_view,
    })
