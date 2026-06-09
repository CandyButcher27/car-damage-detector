"""
Backend tests for the UpSure PoC API.

Unit tests: mock model calls, fast, no GPU/CPU inference.
Integration tests: require model files on disk, marked with @pytest.mark.skipif.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

SAMPLES = Path(__file__).parent.parent / "Samples"
MODELS  = Path(__file__).parent.parent / "models"
CAR_IMG = SAMPLES / "car_10.jpg"

# Lazy import so model loading doesn't happen at collection time
import poc_api
from poc_api import app

client = TestClient(app, raise_server_exceptions=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _car_file(name: str = "car.jpg") -> tuple:
    return (name, CAR_IMG.read_bytes(), "image/jpeg")


def _mock_damage_pred(damaged: bool = False, conf: float = 0.1) -> dict:
    return {
        "damage_detected":  damaged,
        "confidence_score": conf,
        "prob_damaged":     conf if damaged else 0.05,
        "prob_clean":       1.0 - conf,
    }


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_status_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_has_expected_keys():
    resp = client.get("/health")
    data = resp.json()
    for key in ("damage_model_ready", "yolo_model_ready", "anpr_model_ready",
                "damage_model_path", "yolo_model_path", "anpr_model_path"):
        assert key in data, f"missing key: {key}"


def test_health_anpr_model_path_is_correct():
    resp = client.get("/health")
    path = resp.json()["anpr_model_path"]
    assert "anpr_plate_detector" in path


# ── /predict/damage — validation ─────────────────────────────────────────────

def test_damage_no_files_returns_400():
    resp = client.post("/predict/damage")
    assert resp.status_code == 400


def test_damage_non_image_bytes_returns_415():
    with patch("poc_api._get_damage_session"):
        resp = client.post(
            "/predict/damage",
            files={"front": ("test.jpg", b"not-an-image", "image/jpeg")},
        )
    assert resp.status_code == 415


# ── /predict/damage — response shape ─────────────────────────────────────────

def test_damage_response_has_required_keys():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    data = resp.json()
    for key in ("damage_detected", "total_views_analyzed", "overall_confidence", "per_view", "plate"):
        assert key in data, f"missing top-level key: {key}"


def test_damage_plate_key_has_required_fields():
    mock_anpr = {"detected": True, "plate_text": "12 AB 345", "confidence": 0.88,
                 "num_plates": 1, "annotated_image": "", "plate_crop": ""}

    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", True),
        patch("poc_api._run_anpr_pipeline", return_value=mock_anpr),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    plate = resp.json()["plate"]
    for key in ("detected", "plate_text", "confidence", "source_view"):
        assert key in plate, f"missing plate key: {key}"


def test_damage_per_view_contains_submitted_views():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post(
            "/predict/damage",
            files={
                "front": _car_file("front.jpg"),
                "back":  _car_file("back.jpg"),
                "left":  _car_file("left.jpg"),
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_views_analyzed"] == 3
    assert set(data["per_view"].keys()) == {"front", "back", "left"}


# ── ANPR integration logic ────────────────────────────────────────────────────

def test_anpr_uses_front_view_first():
    called_on: list[bytes] = []

    def mock_anpr(img_bytes, is_oman_plate=False):
        called_on.append(img_bytes)
        return {"detected": False, "plate_text": "", "confidence": 0.0,
                "num_plates": 0, "annotated_image": "", "plate_crop": None}

    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", True),
        patch("poc_api._run_anpr_pipeline", side_effect=mock_anpr),
    ):
        resp = client.post(
            "/predict/damage",
            files={
                "back":  _car_file("back.jpg"),
                "front": _car_file("front.jpg"),
            },
        )

    assert resp.status_code == 200
    assert resp.json()["plate"]["source_view"] == "front"
    assert len(called_on) == 1  # ANPR only called once (on front)


def test_anpr_falls_back_to_back_when_no_front():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", True),
        patch("poc_api._run_anpr_pipeline", return_value={
            "detected": False, "plate_text": "", "confidence": 0.0,
            "num_plates": 0, "annotated_image": "", "plate_crop": None,
        }),
    ):
        resp = client.post(
            "/predict/damage",
            files={"back": _car_file("back.jpg"), "left": _car_file("left.jpg")},
        )

    assert resp.status_code == 200
    assert resp.json()["plate"]["source_view"] == "back"


def test_anpr_unavailable_returns_plate_with_error():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    plate = resp.json()["plate"]
    assert plate["detected"] is False
    assert "error" in plate


def test_anpr_result_included_in_plate():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred()),
        patch("poc_api._ANPR_AVAILABLE", True),
        patch("poc_api._run_anpr_pipeline", return_value={
            "detected": True, "plate_text": "AB 1234", "confidence": 0.92,
            "num_plates": 1, "annotated_image": "", "plate_crop": "",
        }),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    plate = resp.json()["plate"]
    assert plate["detected"] is True
    assert plate["plate_text"] == "AB 1234"
    assert plate["confidence"] == pytest.approx(0.92)
    assert plate["source_view"] == "front"


# ── Damage business logic ─────────────────────────────────────────────────────

def test_overall_confidence_only_from_damaged_views():
    """
    overall_confidence must be the max confidence_score among DAMAGED views only.
    A clean view with confidence 0.99 must not inflate the result.
    """
    call_count = {"n": 0}

    def side_effect(_arr):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _mock_damage_pred(damaged=True,  conf=0.60)
        return     _mock_damage_pred(damaged=False, conf=0.99)

    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", side_effect=side_effect),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post(
            "/predict/damage",
            files={
                "front": _car_file("front.jpg"),
                "back":  _car_file("back.jpg"),
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["damage_detected"] is True
    assert data["overall_confidence"] == pytest.approx(0.60, abs=0.01)


def test_fallback_general_damage_when_yolo_empty():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred(damaged=True, conf=0.85)),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    damages = resp.json()["per_view"]["front"]["damages"]
    assert len(damages) == 1
    assert damages[0]["type"] == "general-damage"


def test_fallback_severity_severe_at_high_prob():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value={
            "damage_detected": True, "confidence_score": 0.90,
            "prob_damaged": 0.90, "prob_clean": 0.10,
        }),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.json()["per_view"]["front"]["damages"][0]["severity"] == "severe"


def test_fallback_severity_moderate_at_mid_prob():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value={
            "damage_detected": True, "confidence_score": 0.70,
            "prob_damaged": 0.70, "prob_clean": 0.30,
        }),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.json()["per_view"]["front"]["damages"][0]["severity"] == "moderate"


def test_no_damage_returns_empty_damages_list():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference", return_value=_mock_damage_pred(damaged=False)),
        patch("poc_api._ANPR_AVAILABLE", False),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    data = resp.json()
    assert data["damage_detected"] is False
    assert data["overall_confidence"] == 0.0
    assert data["per_view"]["front"]["damages"] == []


# ── Integration tests (skipped when model files absent) ──────────────────────

@pytest.mark.skipif(
    not (MODELS / "damage_model.onnx").exists(),
    reason="damage_model.onnx not present",
)
def test_integration_real_damage_inference():
    resp = client.post(
        "/predict/damage",
        files={"front": ("car.jpg", CAR_IMG.read_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["damage_detected"], bool)
    assert 0.0 <= data["overall_confidence"] <= 1.0
    assert "damages" in data["per_view"]["front"]
    assert "plate" in data


@pytest.mark.skipif(
    not (MODELS / "damage_model.onnx").exists(),
    reason="damage_model.onnx not present",
)
def test_integration_all_four_views():
    img_bytes = CAR_IMG.read_bytes()
    resp = client.post(
        "/predict/damage",
        files={
            "front": ("front.jpg", img_bytes, "image/jpeg"),
            "back":  ("back.jpg",  img_bytes, "image/jpeg"),
            "left":  ("left.jpg",  img_bytes, "image/jpeg"),
            "right": ("right.jpg", img_bytes, "image/jpeg"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_views_analyzed"] == 4
    assert set(data["per_view"].keys()) == {"front", "back", "left", "right"}


@pytest.mark.skipif(
    not (MODELS / "anpr_plate_detector" / "saved_model.pb").exists() or not poc_api._ANPR_AVAILABLE,
    reason="ANPR SavedModel not present or plate_pipeline not importable",
)
def test_integration_anpr_model_loadable():
    from plate_pipeline import get_model, get_reader
    model = get_model()
    assert model is not None
    reader = get_reader()
    assert reader is not None
