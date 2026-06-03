"""
Model validation against dataset.csv labels.

Modes:
    python test_model.py              # per-image, 20 damaged + 20 clean
    python test_model.py --car        # car-level (all views → one verdict) — PRODUCTION MODE
    python test_model.py --car --n 50
    python test_model.py --car --all
"""

import argparse
import csv
import random
import requests
from collections import defaultdict
from pathlib import Path

API_URL = "http://localhost:8000/predict"


def print_results(results: dict, mode: str):
    hr = "─" * 45
    print(f"\n{hr}")
    print(f"  Mode         : {mode}")
    print(f"  Total tested : {results['total']}")
    print(f"  Accuracy     : {results['accuracy']:.1%}")
    print(f"  Precision    : {results['precision']:.1%}")
    print(f"  Recall       : {results['recall']:.1%}   (↑ critical for insurance)")
    print(f"  F1           : {results['f1']:.1%}")
    print(f"  TP={results['tp']} TN={results['tn']} FP={results['fp']} FN={results['fn']}")
    if results["errors"]:
        print(f"  Errors       : {len(results['errors'])}")
        for e in results["errors"][:5]:
            print(f"    {e}")
    print(hr)


def compute_metrics(tp, tn, fp, fn, errors):
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp)   if (tp + fp) else 0
    rec  = tp / (tp + fn)   if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "errors": errors}


def test_per_image(records: list[dict]) -> dict:
    tp = tn = fp = fn = 0
    errors = []

    for i, row in enumerate(records, 1):
        path = Path(row["image_path"])
        true_label = int(row["label"])
        view = row["view"]

        if not path.exists():
            errors.append(f"MISSING: {path.name}")
            continue

        with open(path, "rb") as f:
            try:
                resp = requests.post(API_URL, files={view: (path.name, f, "image/jpeg")}, timeout=10)
                resp.raise_for_status()
                pred_label = 1 if resp.json()["per_view"][view]["damage_detected"] else 0
            except Exception as e:
                errors.append(f"ERROR {path.name}: {e}")
                continue

        if   true_label == 1 and pred_label == 1: tp += 1
        elif true_label == 0 and pred_label == 0: tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
            print(f"  FP [{i}] {path.name}")
        else:
            fn += 1
            print(f"  FN [{i}] {path.name}")

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(records)}")

    return compute_metrics(tp, tn, fp, fn, errors)


def test_car_level(cars: dict[str, dict]) -> dict:
    """cars = {car_id: {"label": 0|1, "views": {view: path_str}}}"""
    tp = tn = fp = fn = 0
    errors = []

    items = list(cars.items())
    for i, (car_id, car) in enumerate(items, 1):
        true_label = car["label"]
        files = {}
        handles = []
        for view, path_str in car["views"].items():
            p = Path(path_str)
            if p.exists():
                h = open(p, "rb")
                handles.append(h)
                files[view] = (p.name, h, "image/jpeg")

        if not files:
            errors.append(f"NO FILES: {car_id}")
            continue

        try:
            resp = requests.post(API_URL, files=files, timeout=15)
            resp.raise_for_status()
            pred_label = 1 if resp.json()["damage_detected"] else 0
        except Exception as e:
            errors.append(f"ERROR {car_id}: {e}")
            for h in handles:
                h.close()
            continue
        finally:
            for h in handles:
                h.close()

        views_str = "+".join(car["views"].keys())
        if   true_label == 1 and pred_label == 1: tp += 1
        elif true_label == 0 and pred_label == 0: tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
            print(f"  FP [{i}/{len(items)}] {car_id} ({views_str})")
        else:
            fn += 1
            print(f"  FN [{i}/{len(items)}] {car_id} ({views_str})")

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(items)}")

    return compute_metrics(tp, tn, fp, fn, errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car",  action="store_true", help="Car-level evaluation (production mode)")
    parser.add_argument("--n",    type=int, default=20, help="Cars/images per class")
    parser.add_argument("--all",  action="store_true",  help="Test all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open("dataset.csv", newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if args.car:
        # Group by car_id
        car_map: dict[str, dict] = {}
        for row in records:
            cid = row["car_id"]
            if cid not in car_map:
                car_map[cid] = {"label": int(row["label"]), "views": {}}
            car_map[cid]["views"][row["view"]] = row["image_path"]

        damaged_cars = {cid: c for cid, c in car_map.items() if c["label"] == 1}
        clean_cars   = {cid: c for cid, c in car_map.items() if c["label"] == 0}
        print(f"Cars: {len(damaged_cars)} damaged | {len(clean_cars)} clean")

        if not args.all:
            rng = random.Random(args.seed)
            d_keys = rng.sample(list(damaged_cars), min(args.n, len(damaged_cars)))
            c_keys = rng.sample(list(clean_cars),   min(args.n, len(clean_cars)))
            sample = {k: damaged_cars[k] for k in d_keys} | {k: clean_cars[k] for k in c_keys}
            print(f"Sampling {len(d_keys)} damaged + {len(c_keys)} clean cars\n")
        else:
            sample = {**damaged_cars, **clean_cars}

        print(f"Testing {len(sample)} cars (all views per car) against {API_URL}...\n")
        results = test_car_level(sample)
        print_results(results, "CAR-LEVEL (production)")

    else:
        damaged = [r for r in records if int(r["label"]) == 1 and Path(r["image_path"]).exists()]
        clean   = [r for r in records if int(r["label"]) == 0 and Path(r["image_path"]).exists()]
        print(f"Images: {len(damaged)} damaged | {len(clean)} clean")

        if not args.all:
            rng = random.Random(args.seed)
            damaged = rng.sample(damaged, min(args.n, len(damaged)))
            clean   = rng.sample(clean,   min(args.n, len(clean)))
            print(f"Sampling {len(damaged)} damaged + {len(clean)} clean\n")

        sample = damaged + clean
        random.Random(args.seed).shuffle(sample)
        print(f"Testing {len(sample)} images against {API_URL}...\n")
        results = test_per_image(sample)
        print_results(results, "PER-IMAGE")


if __name__ == "__main__":
    main()
