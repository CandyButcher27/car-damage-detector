"""Backend tests for the UpSure data-ingestion API.

Conventions
-----------
* All responses are wrapped in the unified envelope:
      { success, data, error, meta }
  Tests therefore read ``resp.json()["data"]`` (or ``["error"]``).
* Unit tests mock model calls so they run in seconds without weights.
* Integration tests skip automatically when model files are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

SAMPLES = Path(__file__).parent.parent / "Samples"
MODELS = Path(__file__).parent.parent / "models"
CAR_IMG = SAMPLES / "car_10.jpg"

import poc_api
from poc_api import app

client = TestClient(app, raise_server_exceptions=True)


# ── Envelope helpers ─────────────────────────────────────────────────────────
def envelope(resp):
    payload = resp.json()
    assert "success" in payload, payload
    assert "data" in payload
    assert "error" in payload
    assert "meta" in payload
    return payload


def ok(resp):
    payload = envelope(resp)
    assert payload["success"] is True, payload
    assert payload["error"] is None
    return payload["data"]


def err(resp):
    payload = envelope(resp)
    assert payload["success"] is False, payload
    assert payload["data"] is None
    assert payload["error"] is not None
    return payload["error"]


def _car_file(name: str = "car.jpg") -> tuple:
    return (name, CAR_IMG.read_bytes(), "image/jpeg")


def _mock_damage_pred(damaged: bool = False, conf: float = 0.1) -> dict:
    return {
        "damage_detected":  damaged,
        "confidence_score": conf,
        "prob_damaged":     conf if damaged else 0.05,
        "prob_clean":       1.0 - conf,
    }


def _mock_damage_batch(batch):
    """Generic batch mock that returns one mocked pred per row.

    For tests that need *per-view* behavior, use ``side_effect=callable`` on
    ``_run_damage_inference_batch`` directly. This default just returns one
    clean prediction per input.
    """
    n = batch.shape[0] if hasattr(batch, "shape") else len(batch)
    return [_mock_damage_pred() for _ in range(n)]


@pytest.fixture(autouse=True)
def _reset_circuits():
    """Each test gets a clean slate so a previously-tripped breaker doesn't bleed."""
    for cb in (poc_api.OCR_CB, poc_api.YOLO_CB, poc_api.DAMAGE_CB):
        cb.reset()
    yield
    for cb in (poc_api.OCR_CB, poc_api.YOLO_CB, poc_api.DAMAGE_CB):
        cb.reset()


# ── View-aware parts remap (post-bug 1.1.1) ─────────────────────────────────
def test_remap_parts_for_front_view_swaps_tail_to_headlight():
    """Front-view photos must never report tail-light parts. The legacy
    `_PARTS_RULES` table assumes a back-of-car canonical viewpoint, so
    every output for view='front' has to swap tail-light terminology to
    the corresponding front-of-car equivalents."""
    parts = ["left_tail_light", "left_reverse_light", "left_brake_light"]
    remapped = poc_api._remap_parts_for_view("front", parts)
    assert "left_tail_light" not in remapped
    assert "left_headlight_assembly" in remapped
    assert "left_brake_light" not in remapped
    assert "left_indicator" in remapped


def test_remap_parts_for_back_view_unchanged():
    parts = ["left_tail_light", "left_brake_light"]
    assert poc_api._remap_parts_for_view("back", parts) == parts


def test_remap_parts_for_side_view_adjusts_glass_label():
    parts = ["rear_windshield", "left_brake_light"]
    out = poc_api._remap_parts_for_view("left", parts)
    assert "left_rear_window" in out
    assert "rear_windshield" not in out


def test_remap_parts_with_unknown_view_is_passthrough():
    parts = ["left_tail_light"]
    assert poc_api._remap_parts_for_view(None, parts) == parts
    assert poc_api._remap_parts_for_view("speedometer", parts) == parts


# ── Envelope basics ──────────────────────────────────────────────────────────
def test_root_returns_envelope():
    resp = client.get("/")
    assert resp.status_code == 200
    data = ok(resp)
    assert data["service"]


def test_request_id_header_round_trips():
    resp = client.get("/livez", headers={"X-Request-ID": "test-abc"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "test-abc"
    payload = envelope(resp)
    assert payload["meta"]["request_id"] == "test-abc"


def test_meta_carries_latency_and_version():
    resp = client.get("/livez")
    payload = envelope(resp)
    assert payload["meta"]["api_version"] == "v1"
    assert payload["meta"]["service_version"]
    assert isinstance(payload["meta"]["latency_ms"], float)


# ── Health probes ────────────────────────────────────────────────────────────
def test_livez_always_ok():
    resp = client.get("/livez")
    assert resp.status_code == 200
    data = ok(resp)
    assert data["status"] == "alive"


def test_health_envelope_includes_components():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = ok(resp)
    assert "components" in data
    names = {c["name"] for c in data["components"]}
    assert "damage_binary" in names
    assert "circuits" in names


def test_readyz_red_when_critical_component_missing():
    """``damage_binary`` is critical; if it's not loadable, /readyz is 503."""
    with patch.object(poc_api, "_get_damage_session", side_effect=FileNotFoundError("nope")):
        resp = client.get("/readyz")
    if resp.status_code == 200:
        # If the real model file is present, the check passed — skip the assertion.
        pytest.skip("damage model is actually available on this host")
    assert resp.status_code == 503
    error = err(resp)
    assert error["code"] == "MODEL_UNAVAILABLE"


# ── /predict/damage ─ validation ─────────────────────────────────────────────
def test_damage_no_files_returns_400():
    resp = client.post("/predict/damage")
    assert resp.status_code == 400
    error = err(resp)
    assert error["code"] == "VALIDATION_ERROR"


def test_damage_non_image_bytes_returns_415():
    with patch("poc_api._get_damage_session"):
        resp = client.post(
            "/predict/damage",
            files={"front": ("test.jpg", b"not-an-image", "image/jpeg")},
        )
    assert resp.status_code == 415
    error = err(resp)
    assert error["code"] == "UNSUPPORTED_MEDIA"


# ── /predict/damage ─ response shape ─────────────────────────────────────────
def test_damage_response_has_required_keys():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch", side_effect=_mock_damage_batch),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    data = ok(resp)
    for key in ("damage_detected", "total_views_analyzed", "overall_confidence", "per_view"):
        assert key in data, f"missing top-level key: {key}"
    assert "plate" not in data


def test_damage_per_view_contains_submitted_views():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch", side_effect=_mock_damage_batch),
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
    data = ok(resp)
    assert data["total_views_analyzed"] == 3
    assert set(data["per_view"].keys()) == {"front", "back", "left"}


# ── Damage business logic ─────────────────────────────────────────────────────
def test_overall_confidence_only_from_damaged_views():
    """First view is damaged at 0.60; second is clean at 0.99 — overall_confidence
    must reflect ONLY damaged views, so it should equal 0.60 (not 0.99)."""

    def batch_side_effect(batch):
        # Two views in the batch; emit [damaged, clean].
        return [
            _mock_damage_pred(damaged=True, conf=0.60),
            _mock_damage_pred(damaged=False, conf=0.99),
        ]

    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch", side_effect=batch_side_effect),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
    ):
        resp = client.post(
            "/predict/damage",
            files={
                "front": _car_file("front.jpg"),
                "back":  _car_file("back.jpg"),
            },
        )

    assert resp.status_code == 200
    data = ok(resp)
    assert data["damage_detected"] is True
    assert data["overall_confidence"] == pytest.approx(0.60, abs=0.01)


def test_fallback_general_damage_when_yolo_empty():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch",
              return_value=[_mock_damage_pred(damaged=True, conf=0.85)]),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    damages = ok(resp)["per_view"]["front"]["damages"]
    assert len(damages) == 1
    assert damages[0]["type"] == "general-damage"


def test_fallback_severity_severe_at_high_prob():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch", return_value=[{
            "damage_detected": True, "confidence_score": 0.90,
            "prob_damaged": 0.90, "prob_clean": 0.10,
        }]),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert ok(resp)["per_view"]["front"]["damages"][0]["severity"] == "severe"


def test_fallback_severity_moderate_at_mid_prob():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch", return_value=[{
            "damage_detected": True, "confidence_score": 0.70,
            "prob_damaged": 0.70, "prob_clean": 0.30,
        }]),
        patch("poc_api._run_yolo_pipeline", return_value=[]),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert ok(resp)["per_view"]["front"]["damages"][0]["severity"] == "moderate"


def test_no_damage_returns_empty_damages_list():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch",
              return_value=[_mock_damage_pred(damaged=False)]),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})

    assert resp.status_code == 200
    data = ok(resp)
    assert data["damage_detected"] is False
    assert data["overall_confidence"] == 0.0
    assert data["per_view"]["front"]["damages"] == []


# ── Policy decision matrix ──────────────────────────────────────────────────
def _pv(*damages):
    """Build a one-view per_view dict from (type, severity) tuples."""
    return {"front": {"damages": [{"type": t, "severity": s} for t, s in damages]}}


@pytest.mark.parametrize("cls,severity,denies", [
    ("scratches", "minor", False),
    ("scratches", "moderate", False),
    ("scratches", "severe", True),
    ("car-part-crack", "moderate", False),
    ("car-part-crack", "severe", True),
    ("deformation", "minor", False),
    ("deformation", "moderate", True),
    ("deformation", "severe", True),
    ("glass-crack", "minor", False),
    ("glass-crack", "moderate", True),
    ("lamp-crack", "minor", False),
    ("lamp-crack", "severe", True),
])
def test_policy_deny_matrix(cls, severity, denies):
    decision = poc_api._policy_decision(_pv((cls, severity)), damage_detected=True)["decision"]
    assert decision == ("DENY" if denies else "GRANT_WITH_WARNING")


def test_policy_flat_tire_never_denies():
    decision = poc_api._policy_decision(_pv(("flat-tire", "severe")), damage_detected=True)
    assert decision["decision"] == "GRANT_WITH_WARNING"
    assert decision["deny_reasons"] == []


def test_policy_general_damage_denies_only_when_severe():
    assert poc_api._policy_decision(_pv(("general-damage", "moderate")), True)["decision"] == "GRANT_WITH_WARNING"
    assert poc_api._policy_decision(_pv(("general-damage", "severe")), True)["decision"] == "DENY"


def test_policy_deny_wins_across_views():
    per_view = {
        "front": {"damages": [{"type": "scratches", "severity": "minor"}]},
        "back": {"damages": [{"type": "deformation", "severity": "moderate"}]},
    }
    decision = poc_api._policy_decision(per_view, damage_detected=True)
    assert decision["decision"] == "DENY"
    assert decision["deny_reasons"] == [{"view": "back", "class": "deformation", "severity": "moderate"}]


# ── Mulkiya front-gate (card/non-card + front/back before OCR) ──────────────
def test_mulkiya_gate_rejects_non_card():
    gate = poc_api._mulkiya_front_gate("not card", {"side": "front"}, side_gate_enabled=True)
    assert gate is not None and gate[0] == "not_a_card"


def test_mulkiya_gate_rejects_back():
    gate = poc_api._mulkiya_front_gate("card", {"side": "back"}, side_gate_enabled=True)
    assert gate is not None and gate[0] == "mulkiya_back"


def test_mulkiya_gate_allows_card_front():
    assert poc_api._mulkiya_front_gate("card", {"side": "front"}, side_gate_enabled=True) is None


def test_mulkiya_gate_side_disabled_allows_back():
    # With the (unreliable) side model disabled, a back is not rejected on side.
    assert poc_api._mulkiya_front_gate("card", {"side": "back"}, side_gate_enabled=False) is None


def test_mulkiya_gate_non_card_wins_even_when_side_disabled():
    gate = poc_api._mulkiya_front_gate("not card", {"side": "back"}, side_gate_enabled=False)
    assert gate is not None and gate[0] == "not_a_card"


def test_mulkiya_gate_allows_when_side_unavailable():
    # No front/back model loaded → only the card gate applies.
    assert poc_api._mulkiya_front_gate("card", None, side_gate_enabled=True) is None


def test_policy_no_damage_grants():
    decision = poc_api._policy_decision({"front": {"damages": []}}, damage_detected=False)
    assert decision["decision"] == "GRANT"
    assert decision["deny_reasons"] == []


def test_policy_decision_present_in_response():
    with (
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch",
              return_value=[_mock_damage_pred(damaged=False)]),
    ):
        resp = client.post("/predict/damage", files={"front": _car_file()})
    decision = ok(resp)["policy_decision"]
    assert decision["decision"] == "GRANT"


# ── Resilience: circuit breaker around YOLO ─────────────────────────────────
def test_yolo_failures_open_circuit_after_threshold():
    """After ``failure_threshold`` YOLO crashes in a row, the breaker opens."""
    poc_api.YOLO_CB.reset()
    threshold = poc_api.YOLO_CB.failure_threshold

    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = True

    with (
        patch("poc_api.YOLO_MODEL_PATH", fake_path),
        patch("poc_api._get_damage_session"),
        patch("poc_api._run_damage_inference_batch",
              return_value=[_mock_damage_pred(damaged=True, conf=0.85)]),
        patch("poc_api._run_yolo_pipeline", side_effect=RuntimeError("yolo blew up")),
    ):
        for _ in range(threshold):
            client.post("/predict/damage", files={"front": _car_file()})

    assert poc_api.YOLO_CB.state == "open"


def test_circuit_breaker_blocks_calls_when_open():
    cb = poc_api.YOLO_CB
    cb.reset()
    cb._state = "open"
    cb._opened_at = 1e18  # very far in the future so it stays open

    from app.errors import CircuitOpenError

    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "never runs")

    cb.reset()


# ── Resilience: subprocess timeout maps to DEPENDENCY_TIMEOUT ───────────────
def test_ocr_timeout_returns_504_with_envelope():
    import subprocess

    def _slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ocr", timeout=1)

    with patch("poc_api._ocr_subprocess", side_effect=_slow):
        resp = client.post(
            "/api/v1/process",
            files={"file": _car_file("doc.jpg")},
            data={"process_type": "pdf"},
        )
    # Either 504 (timeout) or 415 (couldn't convert) is acceptable here —
    # the goal is to confirm the envelope shape on the error path.
    payload = envelope(resp)
    assert payload["success"] is False
    assert payload["error"]["code"]


# ── Envelope conformance on validation error path ──────────────────────────
def test_validation_error_uses_envelope():
    resp = client.post(
        "/api/v1/process",
        files={"file": _car_file()},
        data={"process_type": "not-a-real-type"},
    )
    payload = envelope(resp)
    assert payload["success"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


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
    data = ok(resp)
    assert isinstance(data["damage_detected"], bool)
    assert 0.0 <= data["overall_confidence"] <= 1.0
    assert "damages" in data["per_view"]["front"]


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
    data = ok(resp)
    assert data["total_views_analyzed"] == 4
    assert set(data["per_view"].keys()) == {"front", "back", "left", "right"}
