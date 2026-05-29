from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Set environment variable to silence some TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

# Monkeypatch Keras layers to support loading legacy models in newer Keras/TensorFlow versions
try:
    import keras
    for layer_cls in [keras.layers.Dense, keras.layers.Conv2D]:
        if hasattr(layer_cls, "__init__"):
            orig_init = layer_cls.__init__
            def _make_patched_init(orig):
                def patched_init(self, *args, **kwargs):
                    kwargs.pop("quantization_config", None)
                    orig(self, *args, **kwargs)
                return patched_init
            layer_cls.__init__ = _make_patched_init(orig_init)
except Exception as e:
    print(f"Warning: Failed to monkeypatch Keras layers: {e}")

app = FastAPI(
    title="Car Classifier API",
    description="An API to classify whether an image contains a car or not.",
    version="1.0"
)

# Configuration
MODEL_PATH = Path(__file__).parent / "models" / "best_car_model.keras"
CAR_THRESHOLD = 0.35  # Using the lowered threshold for better recall
FALLBACK_SIZE = 128

# Global variable to hold the model
model = None
img_size = FALLBACK_SIZE

def get_img_size(loaded_model):
    try:
        shape = loaded_model.input_shape
        for d in shape[1:]:
            if d and d > 3:
                return int(d)
    except Exception:
        pass
    return FALLBACK_SIZE

@app.on_event("startup")
async def load_model_on_startup():
    global model, img_size
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    
    model = load_model(str(MODEL_PATH))
    img_size = get_img_size(model)
    print(f"Model loaded successfully! Expected input size: {img_size}x{img_size}")

@app.get("/")
def root():
    return {"message": "Car Classifier API is running. Send POST requests to /predict/"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the file contents
        contents = await file.read()
        
        # Load and preprocess the image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((img_size, img_size), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Run inference
        pred = model.predict(arr, verbose=0)
        
        # Interpret prediction
        if pred.shape[-1] == 1:
            score = float(pred[0][0])
            is_car = score >= CAR_THRESHOLD
            conf = score if is_car else (1.0 - score)
        else:
            idx = int(np.argmax(pred[0]))
            score = float(pred[0][idx])
            is_car = (idx == 0)
            conf = score

        # Return JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "is_car": is_car,
            "confidence": round(conf, 4),
            "raw_score": round(score, 4),
            "threshold_used": CAR_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This allows you to run the script directly with: python api.py
    print("Starting API Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
