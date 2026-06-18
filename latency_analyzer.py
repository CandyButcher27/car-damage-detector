from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT_DIR = Path(__file__).resolve().parent
URL_OPENER = build_opener(ProxyHandler({}))
DEFAULT_SAMPLE_FILES = {
    "car": ROOT_DIR / "Samples" / "car_10.jpg",
    "card": ROOT_DIR / "Samples" / "Mulkiya_front.jpg",
    "document": ROOT_DIR / "Samples" / "sample_pdf.pdf",
    "noncar": ROOT_DIR / "Samples" / "Non_card_image_1.jpeg",
    # damage views default to same car sample — override with --front-file etc.
    "front": ROOT_DIR / "Samples" / "car_10.jpg",
    "back": ROOT_DIR / "Samples" / "car_10.jpg",
    "left": ROOT_DIR / "Samples" / "car_10.jpg",
    "right": ROOT_DIR / "Samples" / "car_10.jpg",
}
CONVERTED_SAMPLE_DIR = ROOT_DIR / "Samples" / "converted"
SAMPLE_FORMATS = ("original", "png", "webp", "bmp", "tiff", "pdf")
CONVERTED_SAMPLE_FILES = {
    "png": {
        "car": CONVERTED_SAMPLE_DIR / "car_10.png",
        "mulkiya": CONVERTED_SAMPLE_DIR / "Mulkiya_front.png",
        "pdf": CONVERTED_SAMPLE_DIR / "Mulkiya_front.png",
        "front": CONVERTED_SAMPLE_DIR / "car_1001.png",
        "back": CONVERTED_SAMPLE_DIR / "car_1003.png",
        "left": CONVERTED_SAMPLE_DIR / "car_1005.png",
        "right": CONVERTED_SAMPLE_DIR / "car_1007.png",
    },
    "webp": {
        "car": CONVERTED_SAMPLE_DIR / "car_10.webp",
        "mulkiya": CONVERTED_SAMPLE_DIR / "Mulkiya_front.webp",
        "pdf": CONVERTED_SAMPLE_DIR / "Mulkiya_front.webp",
        "front": CONVERTED_SAMPLE_DIR / "car_1001.webp",
        "back": CONVERTED_SAMPLE_DIR / "car_1003.webp",
        "left": CONVERTED_SAMPLE_DIR / "car_1005.webp",
        "right": CONVERTED_SAMPLE_DIR / "car_1007.webp",
    },
    "bmp": {
        "car": CONVERTED_SAMPLE_DIR / "car_10.bmp",
        "mulkiya": CONVERTED_SAMPLE_DIR / "Mulkiya_front.bmp",
        "pdf": CONVERTED_SAMPLE_DIR / "Mulkiya_front.bmp",
        "front": CONVERTED_SAMPLE_DIR / "car_1001.bmp",
        "back": CONVERTED_SAMPLE_DIR / "car_1003.bmp",
        "left": CONVERTED_SAMPLE_DIR / "car_1005.bmp",
        "right": CONVERTED_SAMPLE_DIR / "car_1007.bmp",
    },
    "tiff": {
        "car": CONVERTED_SAMPLE_DIR / "car_10.tiff",
        "mulkiya": CONVERTED_SAMPLE_DIR / "Mulkiya_front.tiff",
        "pdf": CONVERTED_SAMPLE_DIR / "Mulkiya_front.tiff",
        "front": CONVERTED_SAMPLE_DIR / "car_1001.tiff",
        "back": CONVERTED_SAMPLE_DIR / "car_1003.tiff",
        "left": CONVERTED_SAMPLE_DIR / "car_1005.tiff",
        "right": CONVERTED_SAMPLE_DIR / "car_1007.tiff",
    },
    "pdf": {
        "car": CONVERTED_SAMPLE_DIR / "car_10.pdf",
        "mulkiya": CONVERTED_SAMPLE_DIR / "Mulkiya_front.pdf",
        "pdf": CONVERTED_SAMPLE_DIR / "Mulkiya_front.pdf",
        "front": CONVERTED_SAMPLE_DIR / "car_1001.pdf",
        "back": CONVERTED_SAMPLE_DIR / "car_1003.pdf",
        "left": CONVERTED_SAMPLE_DIR / "car_1005.pdf",
        "right": CONVERTED_SAMPLE_DIR / "car_1007.pdf",
    },
}


@dataclass(slots=True)
class LatencySample:
    name: str
    method: str
    url: str
    status_code: int | None
    elapsed_ms: float
    server_ms: float | None
    ok: bool
    error: str | None = None
    upload_files: dict[str, str] | None = None
    sample_formats: dict[str, str] | None = None


@dataclass(slots=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str]

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


@dataclass(slots=True)
class Scenario:
    key: str
    title: str
    make_request: Any
    request_form: dict[str, Any] | None = None


class SamplePicker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.random_seed)
        self.lock = threading.Lock()

    def pick(self, key: str, override: Path | None) -> tuple[Path, str]:
        if override is not None:
            return _ensure_file(override), "override"

        if self.args.random_sample_format:
            with self.lock:
                sample_format = self.rng.choice(SAMPLE_FORMATS)
        else:
            sample_format = self.args.sample_format

        return _ensure_file(_preset_sample_for_format(sample_format, key)), sample_format


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark latency across the explicit UpSure process_type API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the unified API.")
    parser.add_argument(
        "--standalone-base-url",
        default="http://127.0.0.1:8001",
        help="Base URL of the standalone car API.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup calls to make before recording samples for each scenario.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of measured requests to collect per scenario.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent workers per scenario. Keep at 1 for pure latency analysis.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[
            "health",
            "predict-car",
            "predict-damage",
            "process-car",
            "process-mulkiya",
            "process-pdf",
            "standalone-car",
            "all",
        ],
        default=None,
        help=(
            "Scenario to run. May be passed more than once. "
            "Defaults to all unified scenarios plus standalone unless --skip-standalone is set."
        ),
    )
    parser.add_argument(
        "--sample-format",
        choices=SAMPLE_FORMATS,
        default="original",
        help=(
            "Use prepared samples in this format for process-car, process-mulkiya, "
            "process-pdf, predict-car, and predict-damage. Explicit file flags override this."
        ),
    )
    parser.add_argument(
        "--random-sample-format",
        action="store_true",
        help=(
            "Choose a random prepared sample format for every upload request. "
            "Explicit file flags still override random selection for that field."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for repeatable --random-sample-format runs.",
    )
    parser.add_argument(
        "--car-file",
        type=Path,
        default=None,
        help="Image file used for process_type=car and /predict/ benchmarks.",
    )
    parser.add_argument(
        "--mulkiya-file",
        type=Path,
        default=None,
        help="Image or PDF file used for process_type=mulkiya benchmarks.",
    )
    parser.add_argument(
        "--pdf-file",
        type=Path,
        default=None,
        help="File used for process_type=pdf benchmarks. Images are accepted to test API image-to-PDF conversion.",
    )
    parser.add_argument(
        "--card-threshold",
        type=float,
        default=0.5,
        help="Card threshold passed to /api/v1/process when process_type=mulkiya.",
    )
    parser.add_argument(
        "--prefer-pdf-text",
        action="store_true",
        help="Ask the PDF and Mulkiya PDF pipelines to prefer embedded PDF text when available.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="en",
        help="OCR language form field passed to process_type=mulkiya and process_type=pdf.",
    )
    parser.add_argument(
        "--mulkiya-skip-ocr",
        action="store_true",
        help="Pass skip_ocr=true for process_type=mulkiya to benchmark classification without OCR.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the raw benchmark results as JSON.")
    parser.add_argument("--skip-standalone", action="store_true", help="Skip benchmarking the standalone car API.")
    parser.add_argument("--front-file", type=Path, default=None,
                        help="Front view image for /predict/damage benchmark.")
    parser.add_argument("--back-file",  type=Path, default=None,
                        help="Back view image for /predict/damage benchmark.")
    parser.add_argument("--left-file",  type=Path, default=None,
                        help="Left view image for /predict/damage benchmark.")
    parser.add_argument("--right-file", type=Path, default=None,
                        help=(
                            "Right view image for /predict/damage benchmark. "
                            "Note: /predict/damage runs a two-stage pipeline. "
                            "Stage 1 (EfficientNet-B2 binary) always runs. "
                            "Stage 2 (YOLOv8n damage-type) runs only when Stage 1 detects damage. "
                            "Use damaged car images to benchmark the full v2 pipeline latency."
                        ))
    return parser


def _ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    return path


def _preset_sample_for_format(sample_format: str, key: str) -> Path:
    if sample_format == "original":
        original_key = "card" if key == "mulkiya" else "document" if key == "pdf" else key
        return DEFAULT_SAMPLE_FILES[original_key]
    return CONVERTED_SAMPLE_FILES[sample_format][key]


def _normalize_base_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname
    if hostname != "localhost":
        return url.rstrip("/")

    port = f":{parts.port}" if parts.port else ""
    netloc = f"127.0.0.1{port}"
    if parts.username:
        auth = parts.username
        if parts.password:
            auth = f"{auth}:{parts.password}"
        netloc = f"{auth}@{netloc}"

    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


def _request_elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _parse_server_ms(response: HttpResponse) -> float | None:
    for header_name in ("X-Process-Time-ms", "X-Process-Time-MS", "X-Process-Time"):
        value = response.headers.get(header_name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _send_request(request: Request, timeout: float) -> HttpResponse:
    try:
        with URL_OPENER.open(request, timeout=timeout) as response:
            response.read()
            return HttpResponse(response.status, dict(response.headers.items()))
    except HTTPError as exc:
        exc.read()
        return HttpResponse(exc.code, dict(exc.headers.items()))


def _send_get(url: str, timeout: float, name: str) -> LatencySample:
    started_at = time.perf_counter()
    try:
        response = _send_request(Request(url, method="GET"), timeout)
        return LatencySample(
            name=name,
            method="GET",
            url=url,
            status_code=response.status_code,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=_parse_server_ms(response),
            ok=response.ok,
        )
    except Exception as exc:
        return LatencySample(
            name=name,
            method="GET",
            url=url,
            status_code=None,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=None,
            ok=False,
            error=str(exc),
        )


def _build_multipart_body(file_path: Path, extra_data: dict[str, Any] | None) -> tuple[bytes, str]:
    boundary = f"----upsure-latency-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in (extra_data or {}).items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                f"{value}\r\n".encode("utf-8"),
            ]
        )

    file_bytes = file_path.read_bytes()
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {_guess_mime_type(file_path)}\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    return b"".join(chunks), boundary


def _build_multipart_damage_body(view_files: dict[str, Path]) -> tuple[bytes, str]:
    """Build multipart body with named view fields (front/back/left/right)."""
    boundary = f"----upsure-damage-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for field_name, file_path in view_files.items():
        file_bytes = file_path.read_bytes()
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
                f"Content-Type: {_guess_mime_type(file_path)}\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            b"\r\n",
        ])

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def _send_damage_post(
    url: str,
    timeout: float,
    name: str,
    view_files: dict[str, Path],
    sample_formats: dict[str, str] | None = None,
) -> LatencySample:
    started_at = time.perf_counter()
    try:
        body, boundary = _build_multipart_damage_body(view_files)
        request = Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        response = _send_request(request, timeout)
        return LatencySample(
            name=name,
            method="POST",
            url=url,
            status_code=response.status_code,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=_parse_server_ms(response),
            ok=response.ok,
            upload_files={key: str(path) for key, path in view_files.items()},
            sample_formats=sample_formats,
        )
    except (OSError, URLError, ValueError) as exc:
        return LatencySample(
            name=name,
            method="POST",
            url=url,
            status_code=None,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=None,
            ok=False,
            error=str(exc),
            upload_files={key: str(path) for key, path in view_files.items()},
            sample_formats=sample_formats,
        )


def _send_multipart_post(
    url: str,
    timeout: float,
    name: str,
    file_path: Path,
    extra_data: dict[str, Any] | None = None,
    sample_format: str | None = None,
) -> LatencySample:
    started_at = time.perf_counter()
    form_data = {key: str(value) for key, value in (extra_data or {}).items()}
    try:
        body, boundary = _build_multipart_body(file_path, form_data)
        request = Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        response = _send_request(request, timeout)
        return LatencySample(
            name=name,
            method="POST",
            url=url,
            status_code=response.status_code,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=_parse_server_ms(response),
            ok=response.ok,
            upload_files={"file": str(file_path)},
            sample_formats={"file": sample_format} if sample_format else None,
        )
    except (OSError, URLError, ValueError) as exc:
        return LatencySample(
            name=name,
            method="POST",
            url=url,
            status_code=None,
            elapsed_ms=_request_elapsed_ms(started_at),
            server_ms=None,
            ok=False,
            error=str(exc),
            upload_files={"file": str(file_path)},
            sample_formats={"file": sample_format} if sample_format else None,
        )


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "application/octet-stream"


def _run_scenario(make_request, warmup: int, runs: int, concurrency: int) -> list[LatencySample]:
    for _ in range(max(warmup, 0)):
        make_request()

    samples: list[LatencySample] = []
    if concurrency <= 1:
        for _ in range(max(runs, 0)):
            samples.append(make_request())
        return samples

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(max(runs, 0))]
        for future in as_completed(futures):
            samples.append(future.result())
    return samples


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize(name: str, samples: list[LatencySample]) -> dict[str, Any]:
    elapsed = [sample.elapsed_ms for sample in samples]
    server = [sample.server_ms for sample in samples if sample.server_ms is not None]
    success = [sample for sample in samples if sample.ok]
    format_counts: dict[str, int] = {}
    upload_examples: dict[str, str] = {}
    for sample in samples:
        for sample_format in (sample.sample_formats or {}).values():
            format_counts[sample_format] = format_counts.get(sample_format, 0) + 1
        for field_name, upload_path in (sample.upload_files or {}).items():
            upload_examples.setdefault(field_name, upload_path)

    return {
        "name": name,
        "requests": len(samples),
        "success_rate": round((len(success) / len(samples)) * 100.0, 2) if samples else 0.0,
        "elapsed_ms": {
            "min": round(min(elapsed), 2) if elapsed else 0.0,
            "avg": round(statistics.mean(elapsed), 2) if elapsed else 0.0,
            "p50": round(_percentile(elapsed, 0.50), 2) if elapsed else 0.0,
            "p95": round(_percentile(elapsed, 0.95), 2) if elapsed else 0.0,
            "p99": round(_percentile(elapsed, 0.99), 2) if elapsed else 0.0,
            "max": round(max(elapsed), 2) if elapsed else 0.0,
        },
        "server_ms": {
            "min": round(min(server), 2) if server else None,
            "avg": round(statistics.mean(server), 2) if server else None,
            "p50": round(_percentile(server, 0.50), 2) if server else None,
            "p95": round(_percentile(server, 0.95), 2) if server else None,
            "p99": round(_percentile(server, 0.99), 2) if server else None,
            "max": round(max(server), 2) if server else None,
        },
        "errors": [sample.error for sample in samples if sample.error],
        "sample_format_counts": format_counts,
        "upload_examples": upload_examples,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{summary['name']}")
    print(f"  Requests: {summary['requests']}")
    print(f"  Success rate: {summary['success_rate']}%")
    elapsed = summary["elapsed_ms"]
    print(
        "  Client latency ms: "
        f"min={elapsed['min']}, avg={elapsed['avg']}, p50={elapsed['p50']}, p95={elapsed['p95']}, p99={elapsed['p99']}, max={elapsed['max']}"
    )
    server = summary["server_ms"]
    if server["avg"] is not None:
        print(
            "  Server process ms: "
            f"min={server['min']}, avg={server['avg']}, p50={server['p50']}, p95={server['p95']}, p99={server['p99']}, max={server['max']}"
        )
    if summary["errors"]:
        print(f"  Errors: {summary['errors']}")
    if summary["sample_format_counts"]:
        print(f"  Sample formats: {summary['sample_format_counts']}")


def _selected_scenarios(args: argparse.Namespace) -> set[str]:
    requested = set(args.scenario or ["all"])
    if "all" in requested:
        requested.update({"health", "predict-car", "predict-damage", "process-car", "process-mulkiya", "process-pdf"})
        if not args.skip_standalone:
            requested.add("standalone-car")
    if args.skip_standalone:
        requested.discard("standalone-car")
    requested.discard("all")
    return requested


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    picker = SamplePicker(args)

    def pick_upload(key: str, override: Path | None = None) -> tuple[Path, str]:
        return picker.pick(key, override)

    def pick_damage_views() -> tuple[dict[str, Path], dict[str, str]]:
        selected = {
            "front": pick_upload("front", args.front_file),
            "back": pick_upload("back", args.back_file),
            "left": pick_upload("left", args.left_file),
            "right": pick_upload("right", args.right_file),
        }
        return (
            {field_name: item[0] for field_name, item in selected.items()},
            {field_name: item[1] for field_name, item in selected.items()},
        )

    results: list[dict[str, Any]] = []

    print("Benchmarking unified API...")
    unified_base = _normalize_base_url(args.base_url)

    process_car_form = {"process_type": "car"}
    process_mulkiya_form = {
        "process_type": "mulkiya",
        "card_threshold": args.card_threshold,
        "ocr_lang": args.ocr_lang,
        "prefer_pdf_text": str(args.prefer_pdf_text).lower(),
        "skip_ocr": str(args.mulkiya_skip_ocr).lower(),
    }
    process_pdf_form = {
        "process_type": "pdf",
        "ocr_lang": args.ocr_lang,
        "prefer_pdf_text": str(args.prefer_pdf_text).lower(),
    }

    scenarios: list[Scenario] = [
        Scenario(
            "health",
            "Unified GET /",
            lambda: _send_get(f"{unified_base}/", args.timeout, "Unified GET /"),
        ),
        Scenario(
            "health",
            "Unified GET /health",
            lambda: _send_get(f"{unified_base}/health", args.timeout, "Unified GET /health"),
        ),
        Scenario(
            "predict-car",
            "Unified POST /predict/ (car)",
            lambda: (
                lambda chosen: _send_multipart_post(
                    f"{unified_base}/predict/",
                    args.timeout,
                    "Unified POST /predict/ (car)",
                    chosen[0],
                    sample_format=chosen[1],
                )
            )(pick_upload("car", args.car_file)),
        ),
        Scenario(
            "predict-damage",
            "Unified POST /predict/damage (4 views)",
            lambda: (
                lambda chosen: _send_damage_post(
                    f"{unified_base}/predict/damage",
                    args.timeout,
                    "Unified POST /predict/damage (4 views)",
                    chosen[0],
                    chosen[1],
                )
            )(pick_damage_views()),
        ),
        Scenario(
            "process-car",
            "Unified POST /api/v1/process (process_type=car)",
            lambda: (
                lambda chosen: _send_multipart_post(
                    f"{unified_base}/api/v1/process",
                    args.timeout,
                    "Unified POST /api/v1/process (process_type=car)",
                    chosen[0],
                    process_car_form,
                    sample_format=chosen[1],
                )
            )(pick_upload("car", args.car_file)),
            process_car_form,
        ),
        Scenario(
            "process-mulkiya",
            "Unified POST /api/v1/process (process_type=mulkiya)",
            lambda: (
                lambda chosen: _send_multipart_post(
                    f"{unified_base}/api/v1/process",
                    args.timeout,
                    "Unified POST /api/v1/process (process_type=mulkiya)",
                    chosen[0],
                    process_mulkiya_form,
                    sample_format=chosen[1],
                )
            )(pick_upload("mulkiya", args.mulkiya_file)),
            process_mulkiya_form,
        ),
        Scenario(
            "process-pdf",
            "Unified POST /api/v1/process (process_type=pdf)",
            lambda: (
                lambda chosen: _send_multipart_post(
                    f"{unified_base}/api/v1/process",
                    args.timeout,
                    "Unified POST /api/v1/process (process_type=pdf)",
                    chosen[0],
                    process_pdf_form,
                    sample_format=chosen[1],
                )
            )(pick_upload("pdf", args.pdf_file)),
            process_pdf_form,
        ),
    ]

    if not args.skip_standalone:
        standalone_base = _normalize_base_url(args.standalone_base_url)
        scenarios.extend(
            [
                Scenario(
                    "standalone-car",
                    "Standalone GET /",
                    lambda: _send_get(f"{standalone_base}/", args.timeout, "Standalone GET /"),
                ),
                Scenario(
                    "standalone-car",
                    "Standalone POST /predict/ (car)",
                    lambda: (
                        lambda chosen: _send_multipart_post(
                            f"{standalone_base}/predict/",
                            args.timeout,
                            "Standalone POST /predict/ (car)",
                            chosen[0],
                            sample_format=chosen[1],
                        )
                    )(pick_upload("car", args.car_file)),
                ),
            ]
        )

    selected = _selected_scenarios(args)
    runnable = [scenario for scenario in scenarios if scenario.key in selected]

    for scenario in runnable:
        samples = _run_scenario(scenario.make_request, args.warmup, args.runs, args.concurrency)
        summary = _summarize(scenario.title, samples)
        if scenario.request_form:
            summary["request_form"] = scenario.request_form
        results.append(summary)
        _print_summary(summary)

    report = {
        "base_url": args.base_url,
        "standalone_base_url": args.standalone_base_url,
        "warmup": args.warmup,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "scenarios": sorted(selected),
        "sample_format": args.sample_format,
        "random_sample_format": args.random_sample_format,
        "random_seed": args.random_seed,
        "explicit_file_overrides": {
            "car": str(args.car_file) if args.car_file else None,
            "mulkiya": str(args.mulkiya_file) if args.mulkiya_file else None,
            "pdf": str(args.pdf_file) if args.pdf_file else None,
            "front": str(args.front_file) if args.front_file else None,
            "back": str(args.back_file) if args.back_file else None,
            "left": str(args.left_file) if args.left_file else None,
            "right": str(args.right_file) if args.right_file else None,
        },
        "card_threshold": args.card_threshold,
        "prefer_pdf_text": args.prefer_pdf_text,
        "ocr_lang": args.ocr_lang,
        "results": results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved raw results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
