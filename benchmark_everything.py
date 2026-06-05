from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parent
DEFAULTS = {
    "car": ROOT_DIR / "Samples" / "car_10.jpg",
    "mulkiya": ROOT_DIR / "Samples" / "Mulkiya_front.jpg",
    "pdf": ROOT_DIR / "Samples" / "sample_pdf.pdf",
    "file": ROOT_DIR / "README.md",
    "front": ROOT_DIR / "Samples" / "car_1001.jpg",
    "back": ROOT_DIR / "Samples" / "car_1003.jpg",
    "left": ROOT_DIR / "Samples" / "car_1005.jpg",
    "right": ROOT_DIR / "Samples" / "car_1007.jpg",
}


@dataclass(slots=True)
class Scenario:
    name: str
    method: str
    url: str
    form: dict[str, Any] | None = None
    files: dict[str, Path] | None = None


@dataclass(slots=True)
class Sample:
    scenario: str
    run_index: int
    method: str
    url: str
    status_code: int | None
    ok: bool
    elapsed_ms: float
    server_ms: float | None
    response_bytes: int
    confidence_score: float | None
    hitl_decision: str | None
    classification_label: str | None
    is_car: bool | None
    damage_detected: bool | None
    aadhaar_number_found: bool | None
    required_field_count: int | None
    ocr_line_count: int | None
    error: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark all UpSure PoC API routes and write latency/quality metrics."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Unified API base URL.")
    parser.add_argument(
        "--standalone-base-url",
        default="http://127.0.0.1:8001",
        help="Standalone car classifier base URL.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per scenario.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per scenario.")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel workers per scenario.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds.")
    parser.add_argument("--ocr-lang", default="en", help="OCR language for OCR scenarios. Use en for Aadhaar.")
    parser.add_argument(
        "--prefer-pdf-text",
        action="store_true",
        help="Pass prefer_pdf_text=true for PDF OCR.",
    )
    parser.add_argument(
        "--include-standalone",
        action="store_true",
        help="Also benchmark the standalone car API on port 8001.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "benchmark_results",
        help="Directory for JSON and CSV result files when --save-results is used.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write JSON and CSV result files. By default the benchmark only prints a summary.",
    )
    parser.add_argument("--car-file", type=Path, default=DEFAULTS["car"])
    parser.add_argument("--mulkiya-file", type=Path, default=DEFAULTS["mulkiya"])
    parser.add_argument("--pdf-file", type=Path, default=DEFAULTS["pdf"])
    parser.add_argument("--file-file", type=Path, default=DEFAULTS["file"])
    parser.add_argument("--front-file", type=Path, default=DEFAULTS["front"])
    parser.add_argument("--back-file", type=Path, default=DEFAULTS["back"])
    parser.add_argument("--left-file", type=Path, default=DEFAULTS["left"])
    parser.add_argument("--right-file", type=Path, default=DEFAULTS["right"])
    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=0.90,
        help="Confidence at or above this is counted as auto_accept.",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.60,
        help="Confidence below this is counted as reject_or_reupload.",
    )
    return parser


def make_scenarios(args: argparse.Namespace) -> list[Scenario]:
    base = args.base_url.rstrip("/")
    standalone_base = args.standalone_base_url.rstrip("/")
    scenarios = [
        Scenario("root", "GET", f"{base}/"),
        Scenario("health", "GET", f"{base}/health"),
        Scenario(
            "predict-car",
            "POST",
            f"{base}/predict/",
            files={"file": require_file(args.car_file)},
        ),
        Scenario(
            "process-car",
            "POST",
            f"{base}/api/v1/process",
            form={"process_type": "car"},
            files={"file": require_file(args.car_file)},
        ),
        Scenario(
            "process-mulkiya-skip-ocr",
            "POST",
            f"{base}/api/v1/process",
            form={"process_type": "mulkiya", "ocr_lang": args.ocr_lang, "skip_ocr": "true"},
            files={"file": require_file(args.mulkiya_file)},
        ),
        Scenario(
            "process-mulkiya-ocr",
            "POST",
            f"{base}/api/v1/process",
            form={
                "process_type": "mulkiya",
                "ocr_lang": args.ocr_lang,
                "skip_ocr": "false",
                "prefer_pdf_text": str(args.prefer_pdf_text).lower(),
            },
            files={"file": require_file(args.mulkiya_file)},
        ),
        Scenario(
            "process-pdf-ocr",
            "POST",
            f"{base}/api/v1/process",
            form={
                "process_type": "pdf",
                "ocr_lang": args.ocr_lang,
                "prefer_pdf_text": str(args.prefer_pdf_text).lower(),
            },
            files={"file": require_file(args.pdf_file)},
        ),
        Scenario(
            "process-file",
            "POST",
            f"{base}/api/v1/process",
            form={"process_type": "file"},
            files={"file": require_file(args.file_file)},
        ),
        Scenario(
            "predict-damage",
            "POST",
            f"{base}/predict/damage",
            files={
                "front": require_file(args.front_file),
                "back": require_file(args.back_file),
                "left": require_file(args.left_file),
                "right": require_file(args.right_file),
            },
        ),
    ]
    if args.include_standalone:
        scenarios.append(
            Scenario(
                "standalone-predict-car",
                "POST",
                f"{standalone_base}/predict/",
                files={"file": require_file(args.car_file)},
            )
        )
    return scenarios


def require_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark sample does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Benchmark sample is not a file: {resolved}")
    return resolved


def request_once(
    scenario: Scenario,
    *,
    run_index: int,
    timeout: float,
    auto_accept_threshold: float,
    reject_threshold: float,
) -> Sample:
    started = time.perf_counter()
    try:
        with requests.Session() as session:
            response = send_request(session, scenario, timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = parse_json(response)
        metrics = extract_metrics(payload, auto_accept_threshold, reject_threshold)
        error = None if response.ok else summarize_error(response, payload)
        return Sample(
            scenario=scenario.name,
            run_index=run_index,
            method=scenario.method,
            url=scenario.url,
            status_code=response.status_code,
            ok=response.ok,
            elapsed_ms=round(elapsed_ms, 2),
            server_ms=parse_float(response.headers.get("X-Process-Time-ms")),
            response_bytes=len(response.content or b""),
            error=error,
            **metrics,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Sample(
            scenario=scenario.name,
            run_index=run_index,
            method=scenario.method,
            url=scenario.url,
            status_code=None,
            ok=False,
            elapsed_ms=round(elapsed_ms, 2),
            server_ms=None,
            response_bytes=0,
            confidence_score=None,
            hitl_decision=None,
            classification_label=None,
            is_car=None,
            damage_detected=None,
            aadhaar_number_found=None,
            required_field_count=None,
            ocr_line_count=None,
            error=str(exc),
        )


def send_request(session: requests.Session, scenario: Scenario, timeout: float) -> requests.Response:
    if scenario.method == "GET":
        return session.get(scenario.url, timeout=timeout)

    opened_files = []
    try:
        files = None
        if scenario.files:
            files = {}
            for field, path in scenario.files.items():
                handle = path.open("rb")
                opened_files.append(handle)
                files[field] = (path.name, handle)
        return session.post(scenario.url, data=scenario.form or {}, files=files, timeout=timeout)
    finally:
        for handle in opened_files:
            handle.close()


def run_scenario(scenario: Scenario, args: argparse.Namespace) -> tuple[list[Sample], float]:
    for index in range(args.warmup):
        request_once(
            scenario,
            run_index=-(index + 1),
            timeout=args.timeout,
            auto_accept_threshold=args.auto_accept_threshold,
            reject_threshold=args.reject_threshold,
        )

    started = time.perf_counter()
    samples: list[Sample] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                request_once,
                scenario,
                run_index=index + 1,
                timeout=args.timeout,
                auto_accept_threshold=args.auto_accept_threshold,
                reject_threshold=args.reject_threshold,
            )
            for index in range(args.runs)
        ]
        for future in as_completed(futures):
            samples.append(future.result())

    wall_ms = (time.perf_counter() - started) * 1000.0
    samples.sort(key=lambda item: item.run_index)
    return samples, wall_ms


def parse_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def extract_metrics(payload: Any, auto_accept_threshold: float, reject_threshold: float) -> dict[str, Any]:
    confidence = find_confidence(payload)
    lines = flatten_ocr_lines(payload)
    aadhaar_found = aadhaar_number_found(lines)
    required_field_count = count_aadhaar_fields(lines)
    return {
        "confidence_score": confidence,
        "hitl_decision": hitl_decision(confidence, auto_accept_threshold, reject_threshold),
        "classification_label": find_classification_label(payload),
        "is_car": find_bool(payload, "is_car"),
        "damage_detected": find_bool(payload, "damage_detected"),
        "aadhaar_number_found": aadhaar_found if lines else None,
        "required_field_count": required_field_count if lines else None,
        "ocr_line_count": len(lines) if lines else None,
    }


def find_confidence(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("confidence_score"),
        payload.get("overall_confidence"),
        payload.get("confidence"),
        nested_get(payload, ["classification", "probability"]),
        nested_get(payload, ["classification", "confidence"]),
        nested_get(payload, ["car_classification", "confidence"]),
    ]
    for value in candidates:
        number = parse_float(value)
        if number is not None:
            return number
    return None


def nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def find_classification_label(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = nested_get(payload, ["classification", "label"])
    return str(value) if value is not None else None


def find_bool(payload: Any, key: str) -> bool | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    for child_key in ("car_classification",):
        child = payload.get(child_key)
        if isinstance(child, dict) and isinstance(child.get(key), bool):
            return child[key]
    return None


def flatten_ocr_lines(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    lines: list[str] = []
    for source in (payload.get("extracted_data"), payload.get("raw_ocr")):
        collect_lines(source, lines)
    return lines


def collect_lines(value: Any, output: list[str]) -> None:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            output.append(text.strip())
        for child in value.values():
            collect_lines(child, output)
    elif isinstance(value, list):
        for item in value:
            collect_lines(item, output)


def aadhaar_number_found(lines: list[str]) -> bool:
    text = " ".join(lines)
    digits_only = re.sub(r"\D", "", text)
    masked_pattern = re.compile(r"(?:x{4}|X{4}|\*{4})\s*(?:x{4}|X{4}|\*{4})\s*\d{4}")
    spaced_digits_pattern = re.compile(r"\b\d{4}\s+\d{4}\s+\d{4}\b")
    return bool(masked_pattern.search(text) or spaced_digits_pattern.search(text) or re.search(r"\d{12}", digits_only))


def count_aadhaar_fields(lines: list[str]) -> int:
    text = " ".join(line.lower() for line in lines)
    score = 0
    if aadhaar_number_found(lines):
        score += 1
    if re.search(r"\b(?:dob|date of birth|year of birth|yob|birth)\b", text) or re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text):
        score += 1
    if re.search(r"\b(?:male|female|transgender|m|f)\b", text):
        score += 1
    if re.search(r"\b(?:address|s/o|d/o|w/o|c/o|pin|pincode)\b", text):
        score += 1
    return score


def hitl_decision(confidence: float | None, auto_accept_threshold: float, reject_threshold: float) -> str | None:
    if confidence is None:
        return None
    if confidence >= auto_accept_threshold:
        return "auto_accept"
    if confidence < reject_threshold:
        return "reject_or_reupload"
    return "human_review"


def summarize_error(response: requests.Response, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("detail") is not None:
        return str(payload["detail"])[:500]
    return response.text[:500]


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def summarize(samples: list[Sample], wall_ms: float) -> dict[str, Any]:
    elapsed = [sample.elapsed_ms for sample in samples]
    server = [sample.server_ms for sample in samples if sample.server_ms is not None]
    confidence = [sample.confidence_score for sample in samples if sample.confidence_score is not None]
    ok_count = sum(1 for sample in samples if sample.ok)
    return {
        "runs": len(samples),
        "ok": ok_count,
        "failed": len(samples) - ok_count,
        "success_rate": round(ok_count / len(samples), 4) if samples else 0.0,
        "throughput_rps": round((len(samples) / (wall_ms / 1000.0)), 4) if wall_ms > 0 else None,
        "latency_ms": numeric_summary(elapsed),
        "server_ms": numeric_summary(server),
        "confidence_score": numeric_summary(confidence),
        "hitl_decisions": counts(sample.hitl_decision for sample in samples),
        "status_codes": counts(str(sample.status_code) for sample in samples),
        "errors": counts(sample.error for sample in samples if sample.error),
    }


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 2),
        "mean": round(statistics.mean(ordered), 2),
        "p50": round(percentile(ordered, 50), 2),
        "p90": round(percentile(ordered, 90), 2),
        "p95": round(percentile(ordered, 95), 2),
        "p99": round(percentile(ordered, 99), 2),
        "max": round(ordered[-1], 2),
    }


def percentile(ordered: list[float], percentile_value: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def counts(values: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        output[key] = output.get(key, 0) + 1
    return output


def write_outputs(output_dir: Path, payload: dict[str, Any], samples: list[Sample]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_everything_{stamp}.json"
    csv_path = output_dir / f"benchmark_everything_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()) if samples else [])
        if samples:
            writer.writeheader()
            for sample in samples:
                writer.writerow(asdict(sample))

    return json_path, csv_path


def print_summary(summary_by_scenario: dict[str, Any]) -> None:
    header = (
        f"{'scenario':28} {'ok':>7} {'fail':>5} {'p50':>10} {'p95':>10} "
        f"{'p99':>10} {'rps':>8} {'hitl':>22}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in summary_by_scenario.items():
        latency = summary["latency_ms"]
        hitl = ",".join(f"{k}:{v}" for k, v in summary["hitl_decisions"].items()) or "-"
        print(
            f"{name:28} {summary['ok']:>7} {summary['failed']:>5} "
            f"{format_metric(latency['p50']):>10} {format_metric(latency['p95']):>10} "
            f"{format_metric(latency['p99']):>10} {format_metric(summary['throughput_rps']):>8} "
            f"{hitl[:22]:>22}"
        )


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> int:
    args = build_parser().parse_args()
    scenarios = make_scenarios(args)
    all_samples: list[Sample] = []
    summary_by_scenario: dict[str, Any] = {}
    started_at = datetime.now().isoformat(timespec="seconds")

    for scenario in scenarios:
        print(f"Running {scenario.name}...")
        samples, wall_ms = run_scenario(scenario, args)
        all_samples.extend(samples)
        summary_by_scenario[scenario.name] = summarize(samples, wall_ms)

    payload = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "base_url": args.base_url,
            "runs": args.runs,
            "warmup": args.warmup,
            "concurrency": args.concurrency,
            "ocr_lang": args.ocr_lang,
            "auto_accept_threshold": args.auto_accept_threshold,
            "reject_threshold": args.reject_threshold,
        },
        "summary": summary_by_scenario,
        "samples": [asdict(sample) for sample in all_samples],
    }
    print()
    print_summary(summary_by_scenario)

    if args.save_results:
        json_path, csv_path = write_outputs(args.output_dir, payload, all_samples)
        print()
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")

    return 0 if all(sample.ok for sample in all_samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
