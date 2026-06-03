"""
Car Damage Detection API
------------------------
FastAPI web service wrapping the ONNX model for CPU inference.
Accepts image upload(s), returns damage prediction JSON.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/predict \
         -F "front=@front.jpg" \
         -F "back=@back.jpg"
"""

import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from config import CFG

_A               = CFG["api"]
_P               = CFG["paths"]
ONNX_PATH        = _P["model_onnx"]
IMG_SIZE         = _A["img_size"]
DAMAGE_THRESHOLD = _A["damage_threshold"]
MEAN             = np.array(_A["mean"], dtype=np.float32)
STD              = np.array(_A["std"],  dtype=np.float32)

app = FastAPI(title="Car Damage Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX session once at startup
try:
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
except Exception as e:
    session = None
    print(f"⚠️  ONNX model not loaded: {e}")


def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return arr


def run_inference(arr: np.ndarray) -> dict:
    if session is None:
        raise RuntimeError("ONNX model not loaded. Run train.py first.")
    logits = session.run(None, {input_name: arr})[0][0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    label  = 1 if probs[1] >= DAMAGE_THRESHOLD else 0
    return {
        "damage_detected":   bool(label == 1),
        "confidence_score":  float(round(float(probs[label]), 4)),
        "prob_damaged":      float(round(float(probs[1]), 4)),
        "prob_clean":        float(round(float(probs[0]), 4)),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}


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

    results = {}
    overall_damaged = False
    max_confidence  = 0.0

    for view_name, upload in views.items():
        img_bytes = await upload.read()
        arr = preprocess(img_bytes)
        pred = run_inference(arr)
        results[view_name] = pred
        if pred["damage_detected"]:
            overall_damaged = True
        if pred["confidence_score"] > max_confidence:
            max_confidence = pred["confidence_score"]

    return JSONResponse({
        "damage_detected":     overall_damaged,
        "total_views_analyzed": len(views),
        "overall_confidence":  round(max_confidence, 4),
        "per_view":            results,
    })
