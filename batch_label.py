"""
Batch VLM Labeler
-----------------
Groups images by car using filename pattern:
    <car_id>_Vehicle_<Front|Back|Left|Right>_View_<rest>.jpg

Sends each car's views to the VLM, saves one JSON per car to labels/.
Resumable — skips already-processed cars.

Outputs:
    labels/<car_id>.json    VLM damage report
    labels/_manifest.json   summary of all runs
"""

import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CFG
from damage_detect import detect

IMAGE_DIR = Path(CFG["paths"]["image_dir"])
LABEL_DIR = Path(CFG["paths"]["label_dir"])
LABEL_DIR.mkdir(exist_ok=True)

IMAGE_EXTS  = set(CFG["image"]["extensions"])
CAR_PATTERN = re.compile(
    r"^([0-9a-f]+)_Vehicle_(Front|Back|Left|Right)_View_",
    re.IGNORECASE,
)


def group_images(image_dir: Path) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    unmatched = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = CAR_PATTERN.match(img_path.name)
        if m:
            car_id = m.group(1).lower()
            view   = m.group(2).lower()
            groups[car_id][view] = str(img_path)
        else:
            unmatched.append(img_path)

    if unmatched:
        print(f"⚠️  {len(unmatched)} file(s) didn't match expected pattern — skipping:")
        for p in unmatched[:10]:
            print(f"   {p.name}")

    return dict(groups)


def process(groups: dict[str, dict[str, str]], workers: int = 4):
    total    = len(groups)
    manifest = {}
    counters = {"done": 0, "failed": 0, "skipped": 0}
    lock     = threading.Lock()

    # Separate already-done cars (skip instantly, no thread needed)
    pending = {}
    for car_id, views in groups.items():
        if (LABEL_DIR / f"{car_id}.json").exists():
            counters["skipped"] += 1
            manifest[car_id] = {"status": "skipped", "views": list(views.keys())}
        else:
            pending[car_id] = views

    print(f"⏭️   Skipping {counters['skipped']} already-done cars")
    print(f"🔄  Processing {len(pending)} remaining  |  workers={workers}\n")

    def process_car(car_id: str, views: dict[str, str]) -> tuple[str, dict]:
        try:
            result = detect(**views)
            (LABEL_DIR / f"{car_id}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            entry = {
                "status":          "ok",
                "views":           list(views.keys()),
                "damage_detected": result.get("damage_detected"),
                "damage_count":    len(result.get("damage_items", [])),
            }
            return car_id, entry, None
        except Exception as e:
            return car_id, {"status": "error", "error": str(e), "views": list(views.keys())}, e

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_car, cid, v): cid for cid, v in pending.items()}
        for future in as_completed(futures):
            car_id, entry, err = future.result()
            completed += 1
            with lock:
                manifest[car_id] = entry
                if err:
                    counters["failed"] += 1
                    print(f"[{counters['skipped']+completed}/{total}] ❌  {car_id}: {err}", file=sys.stderr)
                else:
                    counters["done"] += 1
                    dmg = entry.get("damage_detected")
                    cnt = entry.get("damage_count", 0)
                    print(f"[{counters['skipped']+completed}/{total}] ✅  {car_id}  damage={dmg}  items={cnt}")

    (LABEL_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'─'*55}")
    print(f"  Done: {counters['done']}  |  Skipped: {counters['skipped']}  |  Failed: {counters['failed']}")
    print(f"  Labels → {LABEL_DIR.resolve()}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=CFG["batch_label"]["workers"],
                        help="Parallel API calls (raise to 8 if no rate-limit errors)")
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR,
                        help="Override image directory (default: config.yaml paths.image_dir)")
    args = parser.parse_args()

    target_dir = args.image_dir
    if not target_dir.exists():
        print(f"❌  Folder not found: {target_dir.resolve()}")
        sys.exit(1)

    groups = group_images(target_dir)
    print(f"📂  {len(groups)} cars, {sum(len(v) for v in groups.values())} images\n")
    process(groups, workers=args.workers)
