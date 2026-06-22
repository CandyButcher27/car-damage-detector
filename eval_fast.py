"""Fast in-process Mulkiya numeric eval — for iterating on the template extractor.

Unlike verify_mulkiya_batch.py (subprocess per doc → reloads RapidOCR 3x each,
~30s/doc), this loads ONE engine and reuses it across all docs, and skips the
Arabic make/model/color pass (numbers only). It calls the SAME functions the
real pipeline uses (card_crop + _best_template_orientation + _extract_by_template),
so the numeric results match production; only the plumbing differs.

Writes results.csv + per_doc/*.json compatible with make_review_html.py.

Usage:
    python eval_fast.py <zip> [--limit N] [--out DIR]
"""
import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

import card_crop
import ocr_simple_test as O

SCHEMA = O._TEMPLATE_TO_SCHEMA if hasattr(O, "_TEMPLATE_TO_SCHEMA") else {
    "plate_number": "plate_number", "model_year": "model_year",
    "manufacturing_year": "year", "engine_cc": "engine_cc",
    "empty_weight_kg": "empty_weight_kg", "max_load_kg": "max_load_kg",
    "seats": "seats", "no_of_axles": "no_of_axles",
    "vin_or_chassis": "vin_or_chassis", "engine_number": "engine_number",
    "valid_from": "issue_date", "valid_until": "expiry_date",
}

COLS = ["doc", "reason", "document_type", "plate_number", "vin_or_chassis", "engine_cc",
        "empty_weight_kg", "max_load_kg", "seats", "no_of_axles",
        "year", "model_year", "issue_date", "expiry_date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="eval_fast_out")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "per_doc").mkdir(parents=True, exist_ok=True)
    engine = O._create_ocr_engine()  # loaded ONCE, reused for every doc

    zf = zipfile.ZipFile(args.zip)
    names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if args.limit:
        names = names[: args.limit]

    rows = []
    t0 = time.time()
    for i, n in enumerate(names):
        base = os.path.basename(n)
        img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            crop, reason = card_crop.choose_mulkiya_crop(img)
            _oc, _lines, tmpl = O._best_template_orientation(crop, engine, False)
        except Exception as exc:
            rows.append({"doc": base, "reason": f"ERROR:{exc}"})
            continue
        rec = {SCHEMA.get(k, k): v for k, v in tmpl.items()}
        rec["doc"] = base
        rec["reason"] = reason
        # Mulkiya if the trusted anchors bound; else uncertain. (The fast path is
        # numeric-only, so this is a light proxy for the full anchor-gate.)
        rec["document_type"] = (
            "mulkiya" if ("vin_or_chassis" in tmpl and len(tmpl) >= 5) else "uncertain"
        )
        rows.append(rec)
        (out / "per_doc" / f"{base}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{i+1}/{len(names)}] {base[:42]:42} {reason}")

    with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    dt = time.time() - t0
    n = len(rows)
    filled = lambda k: sum(1 for r in rows if r.get(k) not in (None, "", "ERROR"))
    print(f"\n{n} docs in {dt:.1f}s ({dt/max(n,1):.1f}s/doc)")
    for k in ("plate_number", "vin_or_chassis", "engine_cc", "empty_weight_kg",
              "max_load_kg", "seats", "year", "issue_date", "expiry_date"):
        print(f"  {k:18} {100*filled(k)/max(n,1):5.1f}%")
    print(f"Outputs → {out}")


if __name__ == "__main__":
    main()
