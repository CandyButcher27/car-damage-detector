"""
Extract new zip files, run car+card classifiers, save filter_results.csv.

Usage:
    python filter_new_data.py              # extract + score (default thresholds)
    python filter_new_data.py --apply      # also copy passing images to images/
    python filter_new_data.py --car-thresh 0.7 --card-thresh 0.4 --apply

    Zips are auto-discovered from project root (excludes data_raw/).
    Drop any new zip into the project folder and re-run — no code edits needed.

Thresholds:
    car_score  >= car_thresh  → passes car filter
    card_score <= card_thresh → passes card filter (low score = not a card)
    image kept only if BOTH pass
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT     = Path(__file__).parent
DATA_RAW = ROOT / "data_raw"

CAR_MODEL_PATH  = "best_car_model (2).keras"
CARD_MODEL_PATH = "card_noncard_classifier_model (1).keras"
CAR_INPUT_SIZE  = (128, 128)
CARD_INPUT_SIZE = (224, 224)

EXTRACT_DIR  = Path("carstrain_new")
FILTERED_DIR = Path("images")
RESULTS_CSV  = Path("filter_results.csv")


def discover_zips() -> list[Path]:
    return sorted(
        p for p in ROOT.glob("*.zip")
        if not (DATA_RAW / p.name).exists()
    )


def extract_zips():
    EXTRACT_DIR.mkdir(exist_ok=True)
    zip_paths = discover_zips()
    if not zip_paths:
        print("  No new zips found in project root.")
        return 0
    print(f"  Found {len(zip_paths)} zip(s):")
    for z in zip_paths:
        print(f"    {z.name}")
    total = 0
    for zpath in zip_paths:
        if not zpath.exists():
            print(f"  MISSING: {zpath.name} — skipping")
            continue
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            print(f"  SKIPPED (bad/incomplete zip): {zpath.name}")
            continue
        with zf:
            entries = zf.namelist()
            skipped = 0
            extracted = 0
            for name in entries:
                dest = EXTRACT_DIR / name
                if dest.exists():
                    skipped += 1
                    continue
                zf.extract(name, EXTRACT_DIR)
                extracted += 1
            total += len(entries)
            print(f"  {zpath.name}: {extracted} extracted, {skipped} already present ({len(entries)} total)")
    return total


def load_image_for_model(path: Path, size: tuple) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(model, img_paths: list[Path], input_size: tuple, batch_size: int = 32) -> np.ndarray:
    scores = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch = np.concatenate([load_image_for_model(p, input_size) for p in batch_paths], axis=0)
        preds = model.predict(batch, verbose=0)
        scores.extend(preds.flatten().tolist())
        done = min(i + batch_size, len(img_paths))
        print(f"\r  {done}/{len(img_paths)}", end="", flush=True)
    print()
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car-thresh",  type=float, default=0.7)
    parser.add_argument("--card-thresh", type=float, default=0.4)
    parser.add_argument("--apply",       action="store_true",
                        help="Copy passing images to images/")
    parser.add_argument("--batch-size",  type=int, default=32)
    args = parser.parse_args()

    print("=== Step 1: Extracting zips ===")
    extract_zips()

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = sorted([p for p in EXTRACT_DIR.iterdir() if p.suffix.lower() in exts])
    print(f"\nFound {len(img_paths)} images in {EXTRACT_DIR}/")

    if RESULTS_CSV.exists():
        print(f"\nFound existing {RESULTS_CSV} — loading scores (delete file to re-run inference)")
        df = pd.read_csv(RESULTS_CSV)
    else:
        import tensorflow as tf

        print("\n=== Step 2: Loading models ===")
        print(f"  Car model:  {CAR_MODEL_PATH}")
        car_model = tf.keras.models.load_model(CAR_MODEL_PATH)
        print(f"  Card model: {CARD_MODEL_PATH}")
        card_model = tf.keras.models.load_model(CARD_MODEL_PATH)

        print("\n=== Step 3: Running car classifier ===")
        car_scores = run_inference(car_model, img_paths, CAR_INPUT_SIZE, args.batch_size)

        print("=== Step 4: Running card classifier ===")
        card_scores = run_inference(card_model, img_paths, CARD_INPUT_SIZE, args.batch_size)

        df = pd.DataFrame({
            "image_path": [str(p) for p in img_paths],
            "filename":   [p.name for p in img_paths],
            "car_score":  car_scores,
            "card_score": card_scores,
        })
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved scores → {RESULTS_CSV}")

    print(f"\n=== Step 5: Applying thresholds (car>={args.car_thresh}, card<={args.card_thresh}) ===")
    df["passes"] = (df["car_score"] >= args.car_thresh) & (df["card_score"] <= args.card_thresh)

    total    = len(df)
    passing  = df["passes"].sum()
    rejected = total - passing

    print(f"  Total:    {total}")
    print(f"  Passing:  {passing} ({100*passing/total:.1f}%)")
    print(f"  Rejected: {rejected} ({100*rejected/total:.1f}%)")

    print("\n  Score distribution (passing images):")
    passing_df = df[df["passes"]]
    if len(passing_df):
        print(f"    car_score  — mean: {passing_df['car_score'].mean():.3f}, min: {passing_df['car_score'].min():.3f}")
        print(f"    card_score — mean: {passing_df['card_score'].mean():.3f}, max: {passing_df['card_score'].max():.3f}")

    print("\n  Rejection breakdown:")
    failed_car  = ((df["car_score"]  <  args.car_thresh)).sum()
    failed_card = ((df["card_score"] >  args.card_thresh)).sum()
    failed_both = ((df["car_score"]  <  args.car_thresh) & (df["card_score"] > args.card_thresh)).sum()
    print(f"    Failed car only:  {failed_car - failed_both}")
    print(f"    Failed card only: {failed_card - failed_both}")
    print(f"    Failed both:      {failed_both}")

    df.to_csv(RESULTS_CSV, index=False)

    if args.apply:
        print(f"\n=== Step 6: Copying passing images → {FILTERED_DIR}/ ===")
        FILTERED_DIR.mkdir(exist_ok=True)
        copied = 0
        for _, row in df[df["passes"]].iterrows():
            src = Path(row["image_path"])
            dst = FILTERED_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        print(f"  Copied {copied} images to {FILTERED_DIR}/")

    print("\nDone. Next: run batch_label.py — new images already in images/.")


if __name__ == "__main__":
    main()
