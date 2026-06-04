"""
Dataset Preparation
-------------------
Reads labels/ folder (output of batch_label.py) + carstrain/ images.
Builds a flat CSV dataset ready for model training.

Outputs:
    dataset.csv   columns: image_path, car_id, view, label (0/1), damage_type
"""

import csv
import json
import re
from pathlib import Path

from config import CFG

IMAGE_DIRS = [Path(CFG["paths"]["image_dir"])]
LABEL_DIR  = Path(CFG["paths"]["label_dir"])
OUT_CSV    = Path(CFG["paths"]["dataset_csv"])

IMAGE_EXTS  = set(CFG["image"]["extensions"])
CAR_PATTERN = re.compile(
    r"^([0-9a-f]+)_Vehicle_(Front|Back|Left|Right)_View_",
    re.IGNORECASE,
)


def build_image_index(image_dirs: list[Path]) -> dict[str, dict[str, str]]:
    """Returns {car_id: {view: abs_path_str}} — O(n) single pass across all dirs."""
    index: dict[str, dict[str, str]] = {}
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            m = CAR_PATTERN.match(img_path.name)
            if m:
                car_id = m.group(1).lower()
                view   = m.group(2).lower()
                index.setdefault(car_id, {})[view] = str(img_path.resolve())
    return index


def build_dataset(out_csv: Path = OUT_CSV):
    image_index = build_image_index(IMAGE_DIRS)
    rows = []
    no_match = []

    for label_file in sorted(LABEL_DIR.glob("*.json")):
        if label_file.name.startswith("_"):
            continue

        car_id = label_file.stem.lower()
        report = json.loads(label_file.read_text(encoding="utf-8"))

        binary_label = 1 if report.get("damage_detected", False) else 0
        damage_items = report.get("damage_items", [])

        # Map damage types per view from VLM output
        view_damages: dict[str, list[str]] = {}
        for item in damage_items:
            sv = item.get("source_view", "unknown")
            dt = item.get("type", "Other")
            view_damages.setdefault(sv, []).append(dt)

        views = image_index.get(car_id)
        if not views:
            no_match.append(car_id)
            continue

        for view, img_path in views.items():
            damage_type = ", ".join(view_damages.get(view, []))
            rows.append({
                "image_path":  img_path,
                "car_id":      car_id,
                "view":        view,
                "label":       binary_label,
                "damage_type": damage_type if damage_type else ("damaged" if binary_label else "none"),
            })

    if no_match:
        print(f"⚠️  {len(no_match)} label(s) had no matching images: {no_match[:5]}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "car_id", "view", "label", "damage_type"])
        writer.writeheader()
        writer.writerows(rows)

    total     = len(rows)
    damaged_n = sum(1 for r in rows if r["label"] == 1)
    clean_n   = total - damaged_n

    print(f"{'─'*55}")
    print(f"  Total images : {total}")
    print(f"  Damaged (1)  : {damaged_n}  ({damaged_n/total*100:.1f}%)")
    print(f"  Clean   (0)  : {clean_n}  ({clean_n/total*100:.1f}%)")
    print(f"  Saved to     : {out_csv.resolve()}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", action="store_true",
                        help="Output to dataset_verified.csv using gemini consensus labels")
    args = parser.parse_args()

    if not LABEL_DIR.exists():
        print("❌  labels/ not found. Run batch_label.py first.")
    else:
        if args.verified:
            out = Path(CFG["paths"]["dataset_verified_csv"])
            print(f"Building verified dataset → {out}")
        else:
            out = OUT_CSV
        build_dataset(out_csv=out)
