"""End-to-end smoke for a running container.

Run against the container exposed on http://localhost:8000.

Usage:
    python tests/e2e_smoke.py [--base-url http://localhost:8000]

The script exercises every public endpoint, verifies the envelope shape,
and prints a tabular summary. Non-zero exit on any envelope failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "Samples"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {text} ==={RESET}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    marker = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{marker}] {label}{suffix}")
    return ok


def expect_envelope(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except json.JSONDecodeError:
        return {"_invalid_json": True, "_text": resp.text[:400]}
    return body


def assert_envelope_shape(body: dict) -> tuple[bool, str]:
    if "_invalid_json" in body:
        return False, f"response not JSON: {body.get('_text', '')[:120]!r}"
    for key in ("success", "data", "error", "meta"):
        if key not in body:
            return False, f"missing top-level key: {key!r}"
    meta = body["meta"]
    for key in ("request_id", "endpoint", "api_version", "service_version", "latency_ms", "timestamp"):
        if key not in meta:
            return False, f"meta missing key: {key!r}"
    return True, ""


def run(base_url: str) -> int:
    failures = 0
    timeout = httpx.Timeout(60.0, connect=10.0)
    client = httpx.Client(base_url=base_url, timeout=timeout)

    # ── 1. Probes ────────────────────────────────────────────────────────
    banner("Probes")
    for path in ("/livez", "/health", "/readyz"):
        try:
            r = client.get(path)
        except httpx.RequestError as exc:
            failures += 1
            check(f"GET {path}", False, f"connect error: {exc}")
            continue
        body = expect_envelope(r)
        shape_ok, shape_detail = assert_envelope_shape(body)
        # /readyz returns 503 if a critical model isn't loaded; that's still a valid envelope.
        status_ok = r.status_code in (200, 503)
        ok = shape_ok and status_ok
        detail = f"status={r.status_code}  req_id={(body.get('meta') or {}).get('request_id','-')[:8]}…"
        if not shape_ok:
            detail = f"status={r.status_code}  {shape_detail}"
        if not check(f"GET {path}", ok, detail):
            failures += 1
        if path == "/health" and ok:
            comps = body.get("data", {}).get("components", [])
            for c in comps:
                ready = c.get("ready")
                tag = f"{GREEN}ready{RESET}" if ready else f"{YELLOW}not-ready{RESET}"
                print(f"        - {c.get('name'):20s}  {tag}  critical={c.get('critical')}  {c.get('detail') or ''}")

    # ── 2. Metrics ──────────────────────────────────────────────────────
    banner("Metrics")
    r = client.get("/metrics")
    if not check("GET /metrics", r.status_code == 200, f"status={r.status_code}"):
        failures += 1
    else:
        body = r.text
        for needle in (
            "upsure_http_requests_total",
            "upsure_http_request_duration_seconds",
            "upsure_model_ready",
            "upsure_circuit_state",
        ):
            if not check(f"  metric '{needle}'", needle in body):
                failures += 1

    # ── 3. Request-id round-trip ────────────────────────────────────────
    banner("Request-id round-trip")
    r = client.get("/livez", headers={"X-Request-ID": "e2e-test-fixed-id"})
    if not check("X-Request-ID echoed", r.headers.get("X-Request-ID") == "e2e-test-fixed-id",
                 f"got={r.headers.get('X-Request-ID')!r}"):
        failures += 1
    body = expect_envelope(r)
    if not check("meta.request_id matches",
                 body.get("meta", {}).get("request_id") == "e2e-test-fixed-id"):
        failures += 1

    # ── 4. Validation error envelope ────────────────────────────────────
    banner("Validation error path")
    r = client.post("/api/v1/process",
                    files={"file": ("x.jpg", SAMPLES.joinpath("car_10.jpg").read_bytes(), "image/jpeg")},
                    data={"process_type": "totally-bogus"})
    body = expect_envelope(r)
    if not check("envelope shape", assert_envelope_shape(body)[0]):
        failures += 1
    if not check("error.code == VALIDATION_ERROR",
                 (body.get("error") or {}).get("code") == "VALIDATION_ERROR"):
        failures += 1

    # ── 5. /predict/ (car classifier) ───────────────────────────────────
    banner("POST /predict/  (car classification)")
    car_file = SAMPLES / "car_1001.jpg"
    if not car_file.exists():
        check("car sample present", False, str(car_file))
        failures += 1
    else:
        r = client.post("/predict/", files={"file": (car_file.name, car_file.read_bytes(), "image/jpeg")})
        body = expect_envelope(r)
        ok_status = r.status_code in (200, 503)  # 503 ok if Keras model not present
        check(f"status={r.status_code}", ok_status,
              f"data keys: {sorted((body.get('data') or {}).keys())[:5]}")
        if not ok_status:
            failures += 1
        if r.status_code == 200:
            data = body.get("data") or {}
            for k in ("filename", "is_car", "confidence", "raw_score", "threshold_used"):
                if not check(f"data has '{k}'", k in data):
                    failures += 1

    # ── 6. /predict/damage ──────────────────────────────────────────────
    banner("POST /predict/damage  (all four views)")
    views = {
        "front": SAMPLES / "car_1001.jpg",
        "back":  SAMPLES / "car_1003.jpg",
        "left":  SAMPLES / "car_1005.jpg",
        "right": SAMPLES / "car_1007.jpg",
    }
    files = {
        name: (p.name, p.read_bytes(), "image/jpeg")
        for name, p in views.items() if p.exists()
    }
    if not files:
        check("damage samples present", False, "no car_100[1357].jpg under Samples/")
        failures += 1
    else:
        r = client.post("/predict/damage", files=files)
        body = expect_envelope(r)
        check(f"status={r.status_code}", r.status_code == 200,
              f"req_id={(body.get('meta') or {}).get('request_id','-')[:8]}…")
        if r.status_code != 200:
            failures += 1
            print(f"        error: {(body.get('error') or {})}")
        else:
            data = body.get("data") or {}
            for k in ("damage_detected", "total_views_analyzed", "overall_confidence", "per_view"):
                if not check(f"data has '{k}'", k in data):
                    failures += 1
            if "per_view" in data:
                print(f"        per_view keys: {list(data['per_view'].keys())}")
                for view, payload in data["per_view"].items():
                    if "error" in payload:
                        print(f"        - {view}: ERROR {payload['error']}")
                    else:
                        print(f"        - {view}: damage_detected={payload.get('damage_detected')}  "
                              f"conf={payload.get('confidence_score'):.3f}  "
                              f"damages={len(payload.get('damages', []))}")

    # ── 7. /api/v1/process — process_type=file ──────────────────────────
    banner("POST /api/v1/process  (process_type=file)")
    readme = REPO / "README.md"
    r = client.post(
        "/api/v1/process",
        files={"file": (readme.name, readme.read_bytes(), "text/markdown")},
        data={"process_type": "file"},
    )
    body = expect_envelope(r)
    if not check(f"status={r.status_code}", r.status_code == 200):
        failures += 1
    if r.status_code == 200:
        data = body.get("data") or {}
        cls = (data.get("classification") or {})
        print(f"        category={cls.get('label')} mime={cls.get('mime_type')} "
              f"suggested={cls.get('suggested_process_type')}")

    # ── 8. /api/v1/process — process_type=car ───────────────────────────
    banner("POST /api/v1/process  (process_type=car)")
    if car_file.exists():
        r = client.post(
            "/api/v1/process",
            files={"file": (car_file.name, car_file.read_bytes(), "image/jpeg")},
            data={"process_type": "car"},
        )
        body = expect_envelope(r)
        status_ok = r.status_code in (200, 503)
        check(f"status={r.status_code}", status_ok)
        if not status_ok:
            failures += 1
        if r.status_code == 200:
            car_clf = (body.get("data") or {}).get("car_classification") or {}
            print(f"        is_car={car_clf.get('is_car')}  conf={car_clf.get('confidence')}  "
                  f"raw={car_clf.get('raw_score')}")
        elif r.status_code == 503:
            print(f"        503 expected — Keras car model not installed in this image build")

    # ── 9. /api/v1/process — process_type=pdf (only if OCR worker exists) ─
    banner("POST /api/v1/process  (process_type=pdf)  [best-effort, OCR-dependent]")
    pdf = SAMPLES / "sample_pdf.pdf"
    if pdf.exists():
        r = client.post(
            "/api/v1/process",
            files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
            data={"process_type": "pdf", "prefer_pdf_text": "true"},
        )
        body = expect_envelope(r)
        # Acceptable: 200 with OCR, or 503/502/504/500 with error envelope (OCR missing).
        envelope_ok = assert_envelope_shape(body)[0]
        if not check(f"envelope shape (status={r.status_code})", envelope_ok):
            failures += 1
        if body.get("success") is False:
            err = body.get("error") or {}
            print(f"        error.code={err.get('code')}  message={err.get('message')!r}  "
                  f"retryable={err.get('retryable')}")

    # ── Summary ─────────────────────────────────────────────────────────
    banner("Summary")
    if failures == 0:
        print(f"  {GREEN}{BOLD}ALL CHECKS PASSED{RESET}")
    else:
        print(f"  {RED}{BOLD}{failures} CHECK(S) FAILED{RESET}")

    # Print circuit + readiness snapshot
    h = client.get("/health").json()
    print()
    print("  Components after run:")
    for c in (h.get("data") or {}).get("components", []):
        readytag = f"{GREEN}ready{RESET}" if c.get("ready") else f"{YELLOW}not-ready{RESET}"
        print(f"    {c.get('name'):20s} {readytag}  {c.get('detail') or ''}")

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    # Wait for the server to come up (k8s style retry).
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            httpx.get(f"{args.base_url}/livez", timeout=2).raise_for_status()
            break
        except Exception:
            time.sleep(1)
    else:
        print(f"{RED}Server did not respond to /livez within 120s.{RESET}")
        sys.exit(2)

    sys.exit(1 if run(args.base_url) else 0)
