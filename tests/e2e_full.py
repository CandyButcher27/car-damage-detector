"""Comprehensive end-to-end smoke + scenario tests.

Exercises *all* sample files in `Samples/` against the relevant endpoints
and dumps the full envelope shape for each so we can see how the new
response structure behaves across success/failure/edge cases.

Usage:
    python tests/e2e_full.py [--base-url http://localhost:8000] [--save reports/e2e.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "Samples"


# ── Pretty printing ────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}==== {text} ===={RESET}")


def truncate(obj: Any, *, max_len: int = 200) -> Any:
    """Recursively trim long strings/lists so the envelope dump is readable."""
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + f"…(+{len(obj)-max_len} chars)"
    if isinstance(obj, list):
        if len(obj) > 8:
            head = [truncate(x, max_len=max_len) for x in obj[:4]]
            return head + [f"…(+{len(obj)-4} more)"]
        return [truncate(x, max_len=max_len) for x in obj]
    if isinstance(obj, dict):
        return {k: truncate(v, max_len=max_len) for k, v in obj.items()}
    return obj


def render_envelope(body: dict, prefix: str = "  ") -> str:
    """Compact, readable envelope printout."""
    trimmed = truncate(body, max_len=140)
    text = json.dumps(trimmed, indent=2, ensure_ascii=False, default=str)
    return "\n".join(prefix + line for line in text.splitlines())


# ── Result collector ──────────────────────────────────────────────────────
@dataclass
class CaseResult:
    name: str
    endpoint: str
    request: dict
    status: int
    success: bool | None
    error_code: str | None
    latency_ms: float | None
    request_id: str | None
    interesting: dict = field(default_factory=dict)
    envelope_summary: dict = field(default_factory=dict)


@dataclass
class Report:
    base_url: str
    started: str
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "started": self.started,
            "cases": [asdict(c) for c in self.cases],
        }


# ── Helpers ──────────────────────────────────────────────────────────────
def envelope_ok(body: dict) -> bool:
    return all(k in body for k in ("success", "data", "error", "meta"))


def collect(report: Report, name: str, endpoint: str, request: dict, resp: httpx.Response,
            interesting: dict | None = None) -> CaseResult:
    try:
        body = resp.json()
    except Exception:
        body = {"_raw_text": resp.text[:200]}

    meta = body.get("meta") or {}
    err = body.get("error") or {}

    # Strip the heavy bits but keep enough for diagnosis
    summary = {
        "success": body.get("success"),
        "data_keys": sorted((body.get("data") or {}).keys()) if isinstance(body.get("data"), dict) else None,
        "error_code": err.get("code"),
        "error_message": err.get("message"),
        "error_retryable": err.get("retryable"),
        "meta_keys": sorted(meta.keys()),
        "endpoint_in_meta": meta.get("endpoint"),
        "api_version": meta.get("api_version"),
    }

    result = CaseResult(
        name=name,
        endpoint=endpoint,
        request=request,
        status=resp.status_code,
        success=body.get("success"),
        error_code=err.get("code"),
        latency_ms=meta.get("latency_ms"),
        request_id=meta.get("request_id"),
        interesting=interesting or {},
        envelope_summary=summary,
    )
    report.cases.append(result)

    # Pretty header
    status_tag = (
        f"{GREEN}{resp.status_code}{RESET}" if resp.status_code < 400
        else f"{YELLOW}{resp.status_code}{RESET}" if resp.status_code < 500
        else f"{RED}{resp.status_code}{RESET}"
    )
    success_tag = (
        f"{GREEN}success=True{RESET}" if body.get("success") is True
        else f"{RED}success=False{RESET}"
    )
    lat = meta.get("latency_ms")
    lat_str = f"{lat:.1f}ms" if isinstance(lat, (int, float)) else "n/a"
    print(f"{DIM}{name:40s}{RESET}  {endpoint:30s}  {status_tag}  {success_tag}  "
          f"req_id={(meta.get('request_id') or '-')[:8]}  "
          f"latency={lat_str}")

    if err:
        print(f"      {RED}error.code{RESET}={err.get('code')}  "
              f"{RED}retryable{RESET}={err.get('retryable')}  "
              f"message={err.get('message','')[:120]!r}")

    if interesting:
        for k, v in interesting.items():
            print(f"      {YELLOW}{k}{RESET} = {v}")

    return result


# ── Test cases ───────────────────────────────────────────────────────────
def health_probes(client: httpx.Client, report: Report) -> None:
    banner("Probes (envelope shape, no inputs)")
    for path in ("/livez", "/health", "/readyz"):
        try:
            r = client.get(path)
            collect(report, f"probe {path}", path, {}, r)
        except httpx.RequestError as exc:
            print(f"{RED}{path} connect failed: {exc}{RESET}")


def request_id_round_trip(client: httpx.Client, report: Report) -> None:
    banner("Request-ID round-trip")
    r = client.get("/livez", headers={"X-Request-ID": "trace-12345"})
    received = r.headers.get("X-Request-ID")
    body = r.json()
    in_meta = (body.get("meta") or {}).get("request_id")
    collect(report, "X-Request-ID echo", "/livez", {"sent": "trace-12345"}, r,
            interesting={"response_header": received, "meta.request_id": in_meta,
                         "matches": received == "trace-12345" and in_meta == "trace-12345"})


def car_classification_positive(client: httpx.Client, report: Report) -> None:
    banner("/predict/  car classification — POSITIVE samples")
    for name in ("car_1001.jpg", "car_1003.jpg", "car_1005.jpg", "car_1007.jpg", "car_10.jpg"):
        f = SAMPLES / name
        if not f.exists():
            continue
        r = client.post("/predict/", files={"file": (name, f.read_bytes(), "image/jpeg")})
        data = (r.json().get("data") or {}) if r.status_code == 200 else {}
        collect(report, f"car  {name}", "/predict/", {"file": name}, r,
                interesting={
                    "is_car": data.get("is_car"),
                    "confidence": data.get("confidence"),
                    "raw_score": data.get("raw_score"),
                } if data else {})


def car_classification_negative(client: httpx.Client, report: Report) -> None:
    """Push not-car images and inspect false-positive behaviour."""
    banner("/predict/  car classification — NEGATIVE samples (false-positive check)")
    candidates = [
        "Mulkiya_front.jpg",
        "Mulkiya_back.jpg",
        "Non_card_image_1.jpeg",
        "Non_card_image_2.jpeg",
        "aadhar_card.jpg",
        "notcar_images (1).jpg",
        "notcar_images (10).jpg",
        "notcar_images (11).jpg",
        "notcar_images (12).jpg",
        "notcar_images (13).jpg",
    ]
    for name in candidates:
        f = SAMPLES / name
        if not f.exists():
            continue
        r = client.post("/predict/", files={"file": (name, f.read_bytes(), "image/jpeg")})
        data = (r.json().get("data") or {}) if r.status_code == 200 else {}
        collect(report, f"not-car  {name}", "/predict/", {"file": name}, r,
                interesting={
                    "is_car": data.get("is_car"),
                    "false_positive": data.get("is_car") is True,
                    "confidence": data.get("confidence"),
                } if data else {})


def damage_pipeline(client: httpx.Client, report: Report) -> None:
    banner("/predict/damage  per-view pipeline")
    # 1) single-view
    f = SAMPLES / "car_1001.jpg"
    if f.exists():
        r = client.post("/predict/damage", files={"front": (f.name, f.read_bytes(), "image/jpeg")})
        data = (r.json().get("data") or {}) if r.status_code == 200 else {}
        per_view = (data.get("per_view") or {})
        collect(report, "damage  1 view (front=car_1001)", "/predict/damage",
                {"front": "car_1001.jpg"}, r,
                interesting={
                    "damage_detected": data.get("damage_detected"),
                    "overall_confidence": data.get("overall_confidence"),
                    "per_view": {k: {"damage_detected": v.get("damage_detected"),
                                       "confidence": v.get("confidence_score"),
                                       "damages": len(v.get("damages", []))}
                                   for k, v in per_view.items()},
                })

    # 2) all four views
    views = {
        "front": SAMPLES / "car_1001.jpg",
        "back":  SAMPLES / "car_1003.jpg",
        "left":  SAMPLES / "car_1005.jpg",
        "right": SAMPLES / "car_1007.jpg",
    }
    files = {n: (p.name, p.read_bytes(), "image/jpeg") for n, p in views.items() if p.exists()}
    if files:
        r = client.post("/predict/damage", files=files)
        data = (r.json().get("data") or {}) if r.status_code == 200 else {}
        per_view = (data.get("per_view") or {})
        collect(report, "damage  4 views", "/predict/damage", {k: v[0] for k, v in files.items()}, r,
                interesting={
                    "damage_detected": data.get("damage_detected"),
                    "overall_confidence": data.get("overall_confidence"),
                    "views": {k: {"dmg": v.get("damage_detected"),
                                    "conf": round(v.get("confidence_score", 0), 3),
                                    "types": [d.get("type") for d in v.get("damages", [])]}
                                for k, v in per_view.items()},
                })

    # 3) non-image: validation/conversion failure shape
    txt = REPO / "README.md"
    r = client.post("/predict/damage", files={"front": ("README.md", txt.read_bytes(), "text/markdown")})
    collect(report, "damage  bad input (markdown)", "/predict/damage", {"front": "README.md"}, r)


def process_endpoint(client: httpx.Client, report: Report) -> None:
    banner("/api/v1/process  by process_type")

    # file inspection
    r = client.post("/api/v1/process",
                    files={"file": ("README.md", (REPO / "README.md").read_bytes(), "text/markdown")},
                    data={"process_type": "file"})
    cls = ((r.json().get("data") or {}).get("classification") or {})
    collect(report, "process  file (README.md)", "/api/v1/process",
            {"process_type": "file", "file": "README.md"}, r,
            interesting={"category": cls.get("label"), "mime": cls.get("mime_type"),
                         "suggested": cls.get("suggested_process_type")})

    # car
    f = SAMPLES / "car_1001.jpg"
    if f.exists():
        r = client.post("/api/v1/process",
                        files={"file": (f.name, f.read_bytes(), "image/jpeg")},
                        data={"process_type": "car"})
        car = ((r.json().get("data") or {}).get("car_classification") or {})
        collect(report, "process  car (car_1001)", "/api/v1/process",
                {"process_type": "car", "file": "car_1001.jpg"}, r,
                interesting={"is_car": car.get("is_car"), "confidence": car.get("confidence")})

    # mulkiya skip_ocr
    f = SAMPLES / "Mulkiya_front.jpg"
    if f.exists():
        r = client.post("/api/v1/process",
                        files={"file": (f.name, f.read_bytes(), "image/jpeg")},
                        data={"process_type": "mulkiya", "skip_ocr": "true"})
        cls = ((r.json().get("data") or {}).get("classification") or {})
        collect(report, "process  mulkiya skip_ocr (front)", "/api/v1/process",
                {"process_type": "mulkiya", "skip_ocr": True}, r,
                interesting={"label": cls.get("label"), "probability": cls.get("probability")})

    # mulkiya full (slow)
    f = SAMPLES / "Mulkiya_front.jpg"
    if f.exists():
        print(f"{DIM}    (running full Mulkiya OCR — this can take 10-30 s){RESET}")
        try:
            r = client.post("/api/v1/process",
                            files={"file": (f.name, f.read_bytes(), "image/jpeg")},
                            data={"process_type": "mulkiya", "ocr_lang": "ar"},
                            timeout=httpx.Timeout(180.0, connect=10.0))
            data = (r.json().get("data") or {})
            collect(report, "process  mulkiya full OCR (front)", "/api/v1/process",
                    {"process_type": "mulkiya"}, r,
                    interesting={
                        "ocr_line_count": len(((data.get("raw_ocr") or {}).get("lines") or [])),
                        "extracted_keys": sorted((data.get("extracted_data") or {}).keys())[:10],
                        "rag_chunks_count": len(data.get("rag_chunks") or []),
                        "confidence_score": data.get("confidence_score"),
                    })
        except httpx.RequestError as exc:
            print(f"      {RED}OCR call failed: {exc}{RESET}")

    # PDF
    f = SAMPLES / "sample_pdf.pdf"
    if f.exists():
        print(f"{DIM}    (running PDF OCR — this can take 10-30 s){RESET}")
        try:
            r = client.post("/api/v1/process",
                            files={"file": (f.name, f.read_bytes(), "application/pdf")},
                            data={"process_type": "pdf", "prefer_pdf_text": "true"},
                            timeout=httpx.Timeout(180.0, connect=10.0))
            data = (r.json().get("data") or {})
            collect(report, "process  pdf (sample_pdf.pdf)", "/api/v1/process",
                    {"process_type": "pdf", "prefer_pdf_text": True}, r,
                    interesting={
                        "ocr_line_count": len(((data.get("raw_ocr") or {}).get("pages") or [{}])[0].get("lines") or []),
                        "rag_chunks_count": len(data.get("rag_chunks") or []),
                        "confidence_score": data.get("confidence_score"),
                    })
        except httpx.RequestError as exc:
            print(f"      {RED}PDF OCR call failed: {exc}{RESET}")


def edge_cases(client: httpx.Client, report: Report) -> None:
    banner("Edge cases (validation, payload limits, unknown process_type)")

    # 1. Missing file
    r = client.post("/predict/damage")
    collect(report, "damage no files", "/predict/damage", {}, r)

    # 2. Bad process_type
    r = client.post("/api/v1/process",
                    files={"file": ("x.jpg", (SAMPLES / "car_10.jpg").read_bytes(), "image/jpeg")},
                    data={"process_type": "this_does_not_exist"})
    collect(report, "process bad process_type", "/api/v1/process",
            {"process_type": "this_does_not_exist"}, r)

    # 3. Empty bytes
    r = client.post("/predict/", files={"file": ("nothing.jpg", b"", "image/jpeg")})
    collect(report, "car empty bytes", "/predict/", {"file_bytes": 0}, r)

    # 4. Corrupt JPEG
    r = client.post("/predict/", files={"file": ("corrupt.jpg", b"\xff\xd8\xff junk bytes", "image/jpeg")})
    collect(report, "car corrupt jpeg", "/predict/", {"file": "garbage"}, r)

    # 5. Oversized upload (synthesized 30 MiB of zero-bytes JPEG header)
    big = b"\xff\xd8" + b"\x00" * (30 * 1024 * 1024)
    r = client.post("/predict/", files={"file": ("huge.jpg", big, "image/jpeg")})
    collect(report, "car oversized upload (30MB)", "/predict/", {"file_bytes": len(big)}, r)


def dump_one_envelope_in_full(client: httpx.Client, report: Report) -> None:
    """Print one full success envelope so the structure is clear."""
    banner("Full envelope dump (one success, one failure)")

    f = SAMPLES / "car_1001.jpg"
    if f.exists():
        r = client.post("/predict/damage", files={"front": (f.name, f.read_bytes(), "image/jpeg")})
        print(f"{GREEN}-- SUCCESS envelope ({r.status_code}) --{RESET}")
        print(render_envelope(r.json()))

    r = client.post("/api/v1/process",
                    files={"file": ("x.jpg", (SAMPLES / "car_10.jpg").read_bytes(), "image/jpeg")},
                    data={"process_type": "nope"})
    print(f"\n{YELLOW}-- FAILURE envelope ({r.status_code}) --{RESET}")
    print(render_envelope(r.json()))


# ── Driver ───────────────────────────────────────────────────────────────
def run(base_url: str, save: Path | None) -> int:
    client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0, connect=10.0))

    # Wait for server.
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            httpx.get(f"{base_url}/livez", timeout=2).raise_for_status()
            break
        except Exception:
            time.sleep(1)
    else:
        print(f"{RED}Server did not respond.{RESET}"); return 2

    report = Report(base_url=base_url, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    health_probes(client, report)
    request_id_round_trip(client, report)
    car_classification_positive(client, report)
    car_classification_negative(client, report)
    damage_pipeline(client, report)
    process_endpoint(client, report)
    edge_cases(client, report)
    dump_one_envelope_in_full(client, report)

    # Summary
    banner("Summary")
    total = len(report.cases)
    successes = sum(1 for c in report.cases if c.success is True)
    failures = sum(1 for c in report.cases if c.success is False)
    print(f"  total={total}  success={GREEN}{successes}{RESET}  failure={YELLOW}{failures}{RESET}")
    by_code: dict[str, int] = {}
    for c in report.cases:
        key = c.error_code or "OK"
        by_code[key] = by_code.get(key, 0) + 1
    print(f"  by error_code: {by_code}")

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        print(f"\n  Full report written to: {save}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--save", type=Path, default=Path("reports/e2e.json"))
    args = parser.parse_args()
    sys.exit(run(args.base_url, args.save))
