"""Tiny tool to count damage detections + view-aware parts.

Hits /predict/damage with the four car_* samples and prints a compact
table: per-view damage_detected + damage type list + parts_at_risk.

Run after standing up a container with different
UPSURE_DAMAGE_THRESHOLD / UPSURE_YOLO_CONF values to see how the
detection rate moves on a clean-car sample.

Usage:
    python tests/damage_threshold_sweep.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import httpx

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "Samples"
VIEWS = {"front": "car_1001.jpg", "back": "car_1003.jpg", "left": "car_1005.jpg", "right": "car_1007.jpg"}


def run(base_url: str) -> int:
    files = {v: (n, (SAMPLES / n).read_bytes(), "image/jpeg") for v, n in VIEWS.items() if (SAMPLES / n).exists()}
    if not files:
        print("Samples missing — expected car_1001/1003/1005/1007.jpg in Samples/")
        return 2

    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/predict/damage", files=files, timeout=120.0)
    elapsed = time.perf_counter() - t0

    try:
        body = r.json()
    except Exception:
        print("Non-JSON:", r.text[:300]); return 2

    if not body.get("success"):
        print("Failed:", body.get("error")); return 2

    data = body["data"]
    meta = body.get("meta") or {}
    print(f"latency_ms = {meta.get('latency_ms', 0):.0f}   wall = {elapsed * 1000:.0f}")
    print(f"damage_detected = {data.get('damage_detected')}   overall_confidence = {data.get('overall_confidence')}")
    print()
    print(f"{'view':6s}  {'dmg':3s}  {'prob_dmg':>8s}  {'n_dmg':>5s}  damage types + parts (top 3)")
    print("-" * 110)
    for view, view_data in (data.get("per_view") or {}).items():
        det = view_data.get("damage_detected")
        prob = view_data.get("prob_damaged", 0)
        damages = view_data.get("damages") or []
        n = len(damages)
        sample = []
        for d in damages[:3]:
            parts = d.get("parts_at_risk") or []
            sample.append(f"{d.get('type')}({d.get('confidence'):.2f}) → {','.join(parts[:2])}")
        print(f"{view:6s}  {str(det):3s}  {prob:8.4f}  {n:5d}  {'; '.join(sample)}")

    return 0 if not data.get("damage_detected") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(run(args.base_url))
