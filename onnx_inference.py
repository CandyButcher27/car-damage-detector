from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


ONNX_PROVIDERS = ["CPUExecutionProvider"]


def resolve_model_path(base_dir: Path, env_var: str | None, candidates: list[str]) -> Path:
    if env_var:
        override = os.getenv(env_var)
        if override:
            candidate = Path(override)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"{env_var} points to a missing model file: {candidate}")

    for relative_path in candidates:
        candidate = base_dir / relative_path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(base_dir / path) for path in candidates)
    raise FileNotFoundError(f"Could not find a model file. Looked for: {searched}")


def _as_probability(value: float) -> float:
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-max(min(value, 60.0), -60.0)))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _input_layout_and_size(input_shape: list[Any]) -> tuple[str, int]:
    if len(input_shape) != 4:
        return "NHWC", 224

    if input_shape[-1] == 3:
        height = input_shape[1] if isinstance(input_shape[1], int) else 224
        return "NHWC", int(height)

    if input_shape[1] == 3:
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        return "NCHW", int(height)

    return "NHWC", 224


def _round4(value: float) -> float:
    return round(float(value), 4)


@dataclass(slots=True)
class BinaryOnnxImageClassifier:
    path: Path
    positive_label: str
    negative_label: str
    positive_when_output_high: bool = True
    positive_index: int = 1
    session: Any = field(init=False, repr=False)
    input_name: str = field(init=False)
    input_layout: str = field(init=False)
    input_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.session = ort.InferenceSession(str(self.path), providers=ONNX_PROVIDERS)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_layout, self.input_size = _input_layout_and_size(model_input.shape)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        prepared = prepared.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        if self.input_layout == "NCHW":
            array = array.transpose(2, 0, 1)
        return array[np.newaxis, ...]

    def probabilities(self, image: Image.Image) -> dict[str, float]:
        output = self.session.run(None, {self.input_name: self._preprocess(image)})[0]
        flat = np.asarray(output).reshape(-1)

        if flat.size == 1:
            model_probability = _as_probability(float(flat[0]))
            positive_probability = (
                model_probability if self.positive_when_output_high else 1.0 - model_probability
            )
            negative_probability = 1.0 - positive_probability
        else:
            values = flat.astype(np.float64)
            if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
                values = _softmax(values)
            positive_index = min(max(self.positive_index, 0), values.size - 1)
            positive_probability = float(values[positive_index])
            negative_probability = float(1.0 - positive_probability)
            model_probability = positive_probability

        return {
            f"{self.positive_label}_probability": _round4(positive_probability),
            f"{self.negative_label}_probability": _round4(negative_probability),
            "model_output": _round4(model_probability),
        }

    def classify(self, image: Image.Image, threshold: float) -> dict[str, Any]:
        probabilities = self.probabilities(image)
        positive_probability = probabilities[f"{self.positive_label}_probability"]
        negative_probability = probabilities[f"{self.negative_label}_probability"]
        is_positive = positive_probability >= threshold
        return {
            "label": self.positive_label if is_positive else self.negative_label,
            "confidence": positive_probability if is_positive else negative_probability,
            "threshold": threshold,
            **probabilities,
        }


def _parse_class_names(raw_names: str | None) -> dict[int, str]:
    if not raw_names:
        return {}
    try:
        parsed = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    names: dict[int, str] = {}
    for key, value in parsed.items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return names


def _parse_image_size(raw_size: str | None, fallback: int = 640) -> int:
    if not raw_size:
        return fallback
    try:
        parsed = ast.literal_eval(raw_size)
    except (SyntaxError, ValueError):
        return fallback
    if isinstance(parsed, (list, tuple)) and parsed:
        try:
            return int(parsed[0])
        except (TypeError, ValueError):
            return fallback
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return fallback


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


@dataclass(slots=True)
class YoloOnnxDetector:
    path: Path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 20
    session: Any = field(init=False, repr=False)
    input_name: str = field(init=False)
    class_names: dict[int, str] = field(init=False)
    image_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.session = ort.InferenceSession(str(self.path), providers=ONNX_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.class_names = _parse_class_names(metadata.get("names"))
        self.image_size = _parse_image_size(metadata.get("imgsz"), fallback=640)

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, float, int, int, int, int]:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = prepared.size
        ratio = min(self.image_size / original_width, self.image_size / original_height)
        new_width = int(round(original_width * ratio))
        new_height = int(round(original_height * ratio))
        resized = prepared.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        pad_x = (self.image_size - new_width) // 2
        pad_y = (self.image_size - new_height) // 2
        canvas.paste(resized, (pad_x, pad_y))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)[np.newaxis, ...]
        return array, ratio, pad_x, pad_y, original_width, original_height

    def detect(self, image: Image.Image) -> dict[str, Any]:
        array, ratio, pad_x, pad_y, original_width, original_height = self._preprocess(image)
        output = np.asarray(self.session.run(None, {self.input_name: array})[0])
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            return {"damage_detected": False, "max_confidence": 0.0, "detections": []}
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] <= 4:
            return {"damage_detected": False, "max_confidence": 0.0, "detections": []}

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        if np.any(class_scores < 0.0) or np.any(class_scores > 1.0):
            class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -60.0, 60.0)))

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return {"damage_detected": False, "max_confidence": 0.0, "detections": []}

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = np.empty((boxes_xywh.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)

        detections: list[dict[str, Any]] = []
        for index in _nms(boxes, scores, self.iou_threshold, self.max_detections):
            class_id = int(class_ids[index])
            detections.append(
                {
                    "label": self.class_names.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "confidence": _round4(float(scores[index])),
                    "box": {
                        "x1": _round4(float(boxes[index, 0])),
                        "y1": _round4(float(boxes[index, 1])),
                        "x2": _round4(float(boxes[index, 2])),
                        "y2": _round4(float(boxes[index, 3])),
                    },
                }
            )

        max_confidence = max((item["confidence"] for item in detections), default=0.0)
        return {
            "damage_detected": bool(detections),
            "max_confidence": max_confidence,
            "detections": detections,
        }
