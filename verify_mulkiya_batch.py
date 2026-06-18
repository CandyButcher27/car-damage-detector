"""Batch verification harness for the Mulkiya front pipeline.

Runs the real OCR + extraction path (ocr_simple_test.py as a subprocess, exactly
like production) over every image/PDF in a zip of Mulkiya-front documents, then
reports two things that need no ground-truth labels:

  Tier 1  structural health   — per-field fill rate + format validity over ALL docs
  Tier 2  VLM sample export    — writes a representative subset for a VLM oracle pass

Usage:
    python verify_mulkiya_batch.py Documents_Mulkiya_Front_17-06-2026.zip
    python verify_mulkiya_batch.py <zip> --limit 20 --sample 30 --lang ar

Outputs (under verify_out/ next to the zip):
    results.csv          one row per document, every extracted field + validity flags
    summary.json         aggregate fill-rate + validity stats
    sample_for_vlm/      copied subset of source images for the Tier-2 VLM check
    per_doc/<stem>.json  raw _mulkya.json for each document
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIELDS = [
    "plate_number", "plate_text", "vehicle_type", "make", "model", "color",
    "year", "model_year", "country_of_origin", "vin_or_chassis",
    "engine_number", "engine_cc", "empty_weight_kg", "max_load_kg",
    "seats", "issue_date", "expiry_date", "owner_name",
]

# Fields whose correctness can be machine-checked without ground truth.
NUMERIC_RANGES = {
    "engine_cc": (200, 10000),
    "empty_weight_kg": (100, 10000),
    "max_load_kg": (100, 100000),
    "seats": (1, 80),
    "year": (1950, 2100),
    "model_year": (1950, 2100),
}
DATE_RE = re.compile(r"^\d{2,4}/\d{1,2}/\d{1,2}$")
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{11,20}$")  # ISO VINs exclude I,O,Q


def _is_valid(field: str, value) -> bool | None:
    """True/False if checkable, None if field empty or has no machine check."""
    if value in (None, "", "NIL"):
        return None
    if field in NUMERIC_RANGES:
        try:
            lo, hi = NUMERIC_RANGES[field]
            return lo <= int(value) <= hi
        except (TypeError, ValueError):
            return False
    if field in ("issue_date", "expiry_date"):
        return bool(DATE_RE.match(str(value)))
    if field == "vin_or_chassis":
        return bool(VIN_RE.match(str(value).upper()))
    return None


def run_pipeline(py: str, script: Path, img: Path, lang: str, timeout: int) -> dict | None:
    """Invoke ocr_simple_test.py exactly as poc_api.py does; return parsed _mulkya.json."""
    cmd = [py, str(script), str(img), "--extract_mulkya", "--write_text", "--lang", lang]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}
    if proc.returncode != 0:
        return {"_error": f"rc={proc.returncode}", "_stderr": (proc.stderr or "")[-300:]}
    out = img.with_name(f"{img.stem}_mulkya.json")
    if not out.exists():
        return {"_error": "no_mulkya_json"}
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-verify the Mulkiya front pipeline.")
    ap.add_argument("zip", type=Path, help="Zip of Mulkiya-front documents.")
    ap.add_argument("--lang", default="en", help="OCR primary language (default en; best for mulkiya dual-pass).")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N docs (0 = all).")
    ap.add_argument("--sample", type=int, default=30, help="Images to copy for VLM check.")
    ap.add_argument("--timeout", type=int, default=120, help="Per-doc subprocess timeout (s).")
    ap.add_argument("--out", type=Path, default=None, help="Output dir (default verify_out/).")
    args = ap.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"Zip not found: {args.zip}")

    out_dir = args.out or args.zip.with_name("verify_out")
    per_doc = out_dir / "per_doc"
    sample_dir = out_dir / "sample_for_vlm"
    for d in (out_dir, per_doc, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "ocr_simple_test.py"
    py = sys.executable

    # Copy to ASCII-safe names: Arabic filenames get mangled through the
    # subprocess argv (UTF-8 -> cp1252) and cv2.imread then can't find them.
    work = Path(tempfile.mkdtemp(prefix="mulkiya_verify_"))
    orig_name: dict[Path, str] = {}
    with zipfile.ZipFile(args.zip) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png", ".pdf"))]
        names.sort()
        if args.limit:
            names = names[: args.limit]
        for idx, n in enumerate(names):
            ext = Path(n).suffix.lower()
            target = work / f"doc_{idx:04d}{ext}"
            with zf.open(n) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            orig_name[target] = Path(n).name

    imgs = sorted(work.glob("*"))
    print(f"Processing {len(imgs)} document(s)…")

    rows: list[dict] = []
    sample_step = max(1, len(imgs) // args.sample) if args.sample else 0
    for i, img in enumerate(imgs):
        data = run_pipeline(py, script, img, args.lang, args.timeout) or {}
        err = data.get("_error")
        name = orig_name.get(img, img.name)
        row: dict = {"doc": name, "error": err or "",
                     "document_type": "" if err else (data.get("document_type") or "")}
        for f in FIELDS:
            row[f] = data.get(f) if not err else None
        # machine validity flags
        for f in FIELDS:
            v = _is_valid(f, row.get(f))
            row[f"{f}__valid"] = "" if v is None else ("ok" if v else "BAD")
        rows.append(row)
        (per_doc / f"{img.stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if sample_step and i % sample_step == 0:
            shutil.copy2(img, sample_dir / name)
        flag = f"ERROR {err}" if err else "ok"
        print(f"  [{i+1}/{len(imgs)}] {name[:50]:50}  {flag}")

    # ── aggregate ────────────────────────────────────────────────────────────
    n = len(rows)
    ok_docs = [r for r in rows if not r["error"]]
    doc_types: dict[str, int] = {}
    for r in ok_docs:
        dt = r.get("document_type") or "unknown"
        doc_types[dt] = doc_types.get(dt, 0) + 1
    summary: dict = {
        "total_docs": n,
        "pipeline_errors": n - len(ok_docs),
        "document_types": doc_types,
        "fields": {},
    }
    for f in FIELDS:
        filled = sum(1 for r in ok_docs if r.get(f) not in (None, "", "NIL"))
        checked = [r for r in ok_docs if r.get(f"{f}__valid") in ("ok", "BAD")]
        bad = sum(1 for r in checked if r[f"{f}__valid"] == "BAD")
        summary["fields"][f] = {
            "fill_rate": round(filled / len(ok_docs), 3) if ok_docs else 0.0,
            "filled": filled,
            "format_checked": len(checked),
            "format_bad": bad,
        }

    fieldnames = ["doc", "error", "document_type"] + FIELDS + [f"{f}__valid" for f in FIELDS]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.rmtree(work, ignore_errors=True)

    print("\n── Tier 1: structural health ──────────────────────────────")
    print(f"docs={n}  pipeline_errors={summary['pipeline_errors']}")
    print(f"document_types: {doc_types}")
    print(f"{'field':<18} {'fill':>6}  {'fmt_bad':>7}")
    for f in FIELDS:
        s = summary["fields"][f]
        print(f"{f:<18} {s['fill_rate']*100:5.1f}%  {s['format_bad']:>3}/{s['format_checked']:<3}")
    print(f"\nOutputs → {out_dir}")
    print(f"VLM sample ({len(list(sample_dir.glob('*')))} imgs) → {sample_dir}")


if __name__ == "__main__":
    main()
