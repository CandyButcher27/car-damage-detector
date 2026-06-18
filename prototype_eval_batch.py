"""Tier-1 integrated evaluation: crop + best-of + non-Mulkiya rejection.

For every document in the zip:
  1. Extract on the ORIGINAL image (real pipeline, subprocess).
  2. Build a cropped/deskewed/oriented card (card_crop) and extract on it.
  3. best-of: keep whichever result is a valid Mulkiya with more filled fields.
  4. REJECT the image if neither original nor crop is a Mulkiya (anchor gate).

Reports: rejection count, which source won (orig vs crop), and field fill for
accepted docs — so we can see the lift vs original-only at full scale.

Usage:
    python prototype_eval_batch.py Documents_Mulkiya_Front_17-06-2026.zip [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2

import card_crop

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIELDS = [
    "plate_number", "vehicle_type", "make", "model", "color", "year",
    "vin_or_chassis", "engine_cc", "empty_weight_kg", "max_load_kg",
    "seats", "issue_date", "expiry_date",
]
# Fields that signal a real successful extraction (for the orig-vs-crop tiebreak).
CRITICAL = ["plate_number", "vin_or_chassis", "engine_cc", "empty_weight_kg",
            "max_load_kg", "seats", "year", "issue_date", "expiry_date",
            "vehicle_type", "color", "model"]


def extract(py: str, script: Path, img: Path, timeout: int) -> dict | None:
    cmd = [py, str(script), str(img), "--extract_mulkya", "--write_text", "--lang", "en"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = img.with_name(f"{img.stem}_mulkya.json")
    if p.returncode != 0 or not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def filled(d: dict | None) -> int:
    if not d:
        return 0
    return sum(1 for k in CRITICAL if d.get(k) not in (None, "", "NIL"))


def is_mulkiya(d: dict | None) -> bool:
    return bool(d) and d.get("document_type") == "mulkiya"


def best_of(do: dict | None, dc: dict | None) -> tuple[dict | None, str]:
    om, cm = is_mulkiya(do), is_mulkiya(dc)
    if om and cm:
        return (do, "orig") if filled(do) >= filled(dc) else (dc, "crop")
    if om:
        return do, "orig"
    if cm:
        return dc, "crop"
    return (do or dc), "rejected"  # neither is a Mulkiya


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or args.zip.with_name("eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "ocr_simple_test.py"
    py = sys.executable

    work = Path(tempfile.mkdtemp(prefix="eval_"))
    orig_name: dict[Path, str] = {}
    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.limit:
            names = names[: args.limit]
        for idx, n in enumerate(names):
            t = work / f"doc_{idx:04d}.jpg"
            with zf.open(n) as s, t.open("wb") as d:
                shutil.copyfileobj(s, d)
            orig_name[t] = Path(n).name

    imgs = sorted(work.glob("doc_*.jpg"))
    print(f"Evaluating {len(imgs)} docs (crop + best-of + rejection)…")

    # Stream rows to CSV as they complete so a kill mid-run keeps partial results.
    fieldnames = ["doc", "source", "rejected", "orig_type", "crop_type",
                  "orig_filled", "crop_filled", "best_filled"] + FIELDS
    csv_fh = (out_dir / "eval.csv").open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()
    csv_fh.flush()

    rows: list[dict] = []
    src_count = {"orig": 0, "crop": 0, "rejected": 0}
    for i, img in enumerate(imgs):
        do = extract(py, script, img, args.timeout)
        bgr = cv2.imread(str(img))
        crop_path = img.with_name(f"{img.stem}_crop.jpg")
        try:
            crop, _reason = card_crop.choose_mulkiya_crop(bgr)
            cv2.imwrite(str(crop_path), crop)
            dc = extract(py, script, crop_path, args.timeout)
        except Exception:
            dc = None
        best, src = best_of(do, dc)
        src_count[src] += 1
        rejected = src == "rejected"
        row = {
            "doc": orig_name.get(img, img.name),
            "source": src,
            "rejected": rejected,
            "orig_type": (do or {}).get("document_type", ""),
            "crop_type": (dc or {}).get("document_type", ""),
            "orig_filled": filled(do),
            "crop_filled": filled(dc),
            "best_filled": 0 if rejected else filled(best),
        }
        for f in FIELDS:
            row[f] = "" if rejected else (best or {}).get(f)
        rows.append(row)
        writer.writerow(row)
        csv_fh.flush()  # survive a mid-run kill
        print(f"  [{i+1}/{len(imgs)}] {row['doc'][:42]:42} src={src:8} "
              f"orig={row['orig_filled']:2} crop={row['crop_filled']:2}")

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(rows)
    accepted = [r for r in rows if not r["rejected"]]
    # Fair lift: compare orig vs best-of on ACCEPTED docs only (rejected docs
    # would be dropped in production, so counting their fields as "lost" is
    # misleading). This isolates the crop contribution.
    acc_orig = sum(r["orig_filled"] for r in accepted)
    acc_best = sum(r["best_filled"] for r in accepted)
    summary = {
        "total": n,
        "rejected_non_mulkiya": src_count["rejected"],
        "accepted": len(accepted),
        "source_used": src_count,
        "accepted_sum_filled_orig": acc_orig,
        "accepted_sum_filled_best_of": acc_best,
        "crop_improved_count": sum(1 for r in accepted if r["crop_filled"] > r["orig_filled"]),
    }

    csv_fh.close()  # CSV already written row-by-row above
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)

    print("\n── Integrated eval ────────────────────────────────")
    print(f"docs={n}  rejected(non-mulkiya)={src_count['rejected']}  accepted={len(accepted)}")
    print(f"source used: orig={src_count['orig']}  crop={src_count['crop']}  rejected={src_count['rejected']}")
    print(f"crop improved {summary['crop_improved_count']} of {len(accepted)} accepted docs")
    print(f"accepted-doc fields filled — orig={acc_orig}  best-of={acc_best}  "
          f"(+{acc_best - acc_orig}, {100*(acc_best-acc_orig)/max(1,acc_orig):.0f}%)")
    print(f"\nOutputs → {out_dir}")


if __name__ == "__main__":
    main()
