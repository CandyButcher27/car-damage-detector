"""
plate_pipeline.py
-----------------
License plate detection using the custom YOLOv4 TF SavedModel,
combined with EasyOCR for text recognition.
No Tesseract required.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF info/warnings in console
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Fixes protobuf conflicts

import cv2
import numpy as np
from paddleocr import PaddleOCR
import base64
import re

# ── Lazy-loaded singletons ─────────────────────────────────────────────────
_tf = None
_saved_model = None
_reader = None

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'anpr_plate_detector')
INPUT_SIZE = 416
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.50


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def get_model():
    """Load TF SavedModel once and cache it."""
    global _saved_model
    if _saved_model is None:
        tf = _get_tf()
        from tensorflow.python.saved_model import tag_constants
        print("[ANPR] Loading YOLOv4 model from checkpoint…")
        _saved_model = tf.saved_model.load(CHECKPOINT_PATH, tags=[tag_constants.SERVING])
        print("[ANPR] YOLOv4 model ready.")
    return _saved_model


def get_reader():
    """Load PaddleOCR reader once and cache it."""
    global _reader
    if _reader is None:
        print("[ANPR] Loading PaddleOCR…")
        # Removing show_log=False as newer versions don't support it
        _reader = PaddleOCR(use_angle_cls=True, lang='en')
        print("[ANPR] PaddleOCR ready.")
    return _reader


# ── Image helpers ──────────────────────────────────────────────────────────

def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR OpenCV array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def image_to_base64(img: np.ndarray) -> str:
    """Encode a BGR OpenCV image as a base64 PNG string."""
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


# ── YOLOv4 inference ───────────────────────────────────────────────────────

def run_yolo(img_bgr: np.ndarray):
    """
    Run YOLOv4 on a BGR image.
    Returns list of (xmin, ymin, xmax, ymax) boxes in pixel coords.
    """
    tf = _get_tf()
    model = get_model()

    original_h, original_w = img_bgr.shape[:2]

    # Pre-process: RGB, resize to 416×416, normalise
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_data = img_resized / 255.0
    img_data = np.asarray([img_data], dtype=np.float32)

    # Inference
    infer = model.signatures['serving_default']
    batch = tf.constant(img_data)
    pred_bbox = infer(batch)

    boxes_raw = None
    conf_raw = None
    for _, value in pred_bbox.items():
        boxes_raw = value[:, :, 0:4]
        conf_raw  = value[:, :, 4:]

    # Non-max suppression
    boxes_nms, scores, classes, valid = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes_raw, (tf.shape(boxes_raw)[0], -1, 1, 4)),
        scores=tf.reshape(conf_raw, (tf.shape(conf_raw)[0], -1, tf.shape(conf_raw)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD,
    )

    num_detections = int(valid.numpy()[0])
    raw_boxes = boxes_nms.numpy()[0][:num_detections]

    # Convert normalised [ymin,xmin,ymax,xmax] → pixel [xmin,ymin,xmax,ymax]
    result = []
    for box in raw_boxes:
        ymin = int(box[0] * original_h)
        xmin = int(box[1] * original_w)
        ymax = int(box[2] * original_h)
        xmax = int(box[3] * original_w)
        result.append((xmin, ymin, xmax, ymax))

    return result


# ── OCR ────────────────────────────────────────────────────────────────────

def read_plate_text(plate_img: np.ndarray, is_oman_plate: bool = False) -> tuple:
    """
    Run EasyOCR on a cropped plate image.
    For Oman plates, uses precise multi-zone cropping to avoid Arabic characters and separators.
    """
    reader = get_reader()
    h, w = plate_img.shape[:2]

    if is_oman_plate:
        # ── Oman Plate Multi-Zone Cropping ──
        # By cropping slightly inwards (0.04 to 0.49), we completely avoid 
        # the left border and the middle vertical separator line, which EasyOCR 
        # often mistakes for '1'.
        zone1 = plate_img[int(h*0.05):int(h*0.95), int(w*0.04):int(w*0.49)]
        
        # Zone 2 (English Letter): Bottom half of the middle section, avoiding separators.
        # Widened to 0.85 to account for Yellow Commercial plates where the letter is pushed right
        zone2 = plate_img[int(h*0.48):int(h*0.95), int(w*0.45):int(w*0.85)]

        def read_zone(zone_img, allowlist=None):
            zh, zw = zone_img.shape[:2]
            if zh == 0 or zw == 0: return "", []
            scale = max(2, 200 // max(zh, 1))
            upscaled = cv2.resize(zone_img, (zw * scale, zh * scale), interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            res_raw = reader.ocr(enhanced_bgr, cls=True)
            res = res_raw[0] if res_raw and res_raw[0] else None
            
            if not res: 
                res_raw = reader.ocr(upscaled, cls=True)
                res = res_raw[0] if res_raw and res_raw[0] else None
                
            if not res:
                # Fallback: Direct recognition without text detection (forces reading of tiny angled characters)
                res_raw = reader.ocr(enhanced_bgr, det=False, cls=True)
                if res_raw and res_raw[0] and isinstance(res_raw[0][0], tuple):
                    # Format dummy box to match detection parsing logic
                    dummy_box = [[0,0], [zw,0], [zw,zh], [0,zh]]
                    res = [[dummy_box, res_raw[0][0]]]
            
            z_texts, z_confs = [], []
            if res:
                # Sort boxes left-to-right within the zone
                res.sort(key=lambda x: min(pt[0] for pt in x[0]))
                for line in res:
                    text = line[1][0]
                    conf = line[1][1]
                    
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if allowlist:
                        allowed_pattern = f'[^{allowlist}]'
                        cleaned = re.sub(allowed_pattern, '', cleaned)
                        
                    if cleaned:
                        z_texts.append(cleaned)
                        z_confs.append(conf)
            return ' '.join(z_texts), z_confs

        text1, confs1 = read_zone(zone1, allowlist='0123456789')
        text2, confs2 = read_zone(zone2, allowlist='ABCDEFGHJKLMNPQRSTUVWXYZ')

        final_texts = []
        all_confs = confs1 + confs2
        if text1: final_texts.append(text1)
        if text2: final_texts.append(text2)

        if not final_texts:
            return "", 0.0
        return ' '.join(final_texts), float(np.mean(all_confs))

    else:
        # ── Standard Plate Processing ──
        scale = max(2, 200 // max(h, 1))
        upscaled = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        res_raw = reader.ocr(enhanced, cls=True)
        results = res_raw[0] if res_raw and res_raw[0] else None
        
        if not results:
            res_raw = reader.ocr(upscaled, cls=True)
            results = res_raw[0] if res_raw and res_raw[0] else None
            
        if not results:
            return "", 0.0

        filtered = []
        for line in results:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]
            
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if cleaned:
                filtered.append((min(pt[0] for pt in bbox), cleaned, conf))

        if not filtered:
            return "", 0.0

        texts = [item[1] for item in filtered]
        confs = [item[2] for item in filtered]

        return ' '.join(texts), float(np.mean(confs))


# ── Annotation ─────────────────────────────────────────────────────────────

def annotate_image(img: np.ndarray, boxes: list, plate_text: str, confidence: float) -> np.ndarray:
    """Draw glowing bounding boxes and plate text label on the image."""
    out = img.copy()

    for (xmin, ymin, xmax, ymax) in boxes:
        # Glow effect
        overlay = out.copy()
        cv2.rectangle(overlay, (xmin - 3, ymin - 3), (xmax + 3, ymax + 3), (0, 255, 120), 5)
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        # Solid box
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 80), 2)

    if boxes:
        # Label on the first (most confident) box
        xmin, ymin, xmax, ymax = boxes[0]
        label = f"{plate_text}  ({confidence:.0%})" if plate_text else "Plate detected"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale, thick = 0.75, 2
        (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
        ly = ymin - 10 if ymin - 10 > th else ymax + th + 10
        cv2.rectangle(out, (xmin, ly - th - bl - 4), (xmin + tw + 8, ly + bl - 4), (0, 200, 80), cv2.FILLED)
        cv2.putText(out, label, (xmin + 4, ly - bl - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    return out


# ── Full pipeline ──────────────────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, is_oman_plate: bool = False) -> dict:
    """
    End-to-end: bytes → detection → OCR → annotated image.

    Returns dict with:
        plate_text, confidence, detected, annotated_image (base64), plate_crop (base64 or None)
    """
    img = load_image_from_bytes(file_bytes)
    if img is None:
        return {"error": "Could not decode image. Please upload a valid JPG or PNG."}

    # ── Detection ──
    try:
        boxes = run_yolo(img)
    except Exception as e:
        return {"error": f"Detection failed: {str(e)}"}

    plate_text = ""
    confidence = 0.0
    plate_crop_b64 = None

    if boxes:
        # Use the first detected plate box
        xmin, ymin, xmax, ymax = boxes[0]
        pad = 5
        h, w = img.shape[:2]
        x1 = max(0, xmin - pad)
        y1 = max(0, ymin - pad)
        x2 = min(w, xmax + pad)
        y2 = min(h, ymax + pad)
        plate_crop = img[y1:y2, x1:x2]

        if plate_crop.size > 0:
            plate_text, confidence = read_plate_text(plate_crop, is_oman_plate)
            plate_crop_b64 = image_to_base64(plate_crop)

    annotated = annotate_image(img, boxes, plate_text, confidence)
    annotated_b64 = image_to_base64(annotated)

    return {
        "plate_text": plate_text,
        "confidence": round(confidence, 4),
        "detected": len(boxes) > 0,
        "num_plates": len(boxes),
        "annotated_image": annotated_b64,
        "plate_crop": plate_crop_b64,
    }
