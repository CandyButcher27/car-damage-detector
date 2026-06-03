"""
Gemini Cross-Validator
----------------------
Uses Gemini as a second opinion to clean up uncertain Ollama labels.

Strategy:
  - Ollama avg confidence >= HIGH_CONF_THRESHOLD  → trust Ollama, skip Gemini
  - Ollama avg confidence <  HIGH_CONF_THRESHOLD  → send to Gemini, use consensus
  - Ollama says clean (damage_detected=False)      → send CLEAN_SAMPLE_RATE % to Gemini
                                                     to catch false negatives

Consensus rules:
  Both say damaged  → label = 1  (confirmed_damaged)
  Both say clean    → label = 0  (confirmed_clean)
  Disagree          → label = uncertain (excluded from training CSV)

Outputs:
  labels/<car_id>.json updated with gemini_verdict field
  dataset_verified.csv  cleaned dataset with uncertain rows removed

Usage:
  python gemini_verify.py               # dry run: test on 1 high-conf car first
  python gemini_verify.py --run         # process all eligible cars
  python gemini_verify.py --run --workers 3
"""

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from config import CFG, PROMPTS

load_dotenv()

_G = CFG["gemini"]
_P = CFG["paths"]

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = _G["model"]
HIGH_CONF_THRESHOLD = _G["high_conf_threshold"]
CLEAN_SAMPLE_RATE   = _G["clean_sample_rate"]

IMAGE_DIRS = [Path(_P["image_dir"]), Path("carstrain_new_filtered")]
LABEL_DIR = Path(_P["label_dir"])
OUT_CSV   = Path(_P["dataset_verified_csv"])

IMAGE_EXTS  = set(CFG["image"]["extensions"])
CAR_PATTERN = re.compile(r"^([0-9a-f]+)_Vehicle_(Front|Back|Left|Right)_View_", re.IGNORECASE)


def avg_confidence(report: dict) -> float:
    items = report.get("damage_items", [])
    if not items:
        return 0.0
    return sum(i.get("confidence_score", 0) for i in items) / len(items)


def build_image_index(image_dirs: list[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            m = CAR_PATTERN.match(img_path.name)
            if m:
                index.setdefault(m.group(1).lower(), {})[m.group(2).lower()] = str(img_path)
    return index


def load_image_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > 1600:
        ratio = 1600 / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def call_gemini(views: dict[str, str], max_retries: int = 5) -> dict:
    """Send car views to Gemini, return parsed JSON report. Retries on 429/503."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    VIEW_LABELS = {"front": "FRONT VIEW", "back": "BACK VIEW",
                   "left": "LEFT SIDE VIEW", "right": "RIGHT SIDE VIEW"}

    parts = [
        types.Part.from_text(text=(
            f"I am providing {len(views)} image(s) of the same vehicle. "
            "Each image is labelled with its view direction. "
            "Analyze ALL images for physical damage and return the structured JSON report."
        ))
    ]

    for view_key, img_path in views.items():
        label = VIEW_LABELS.get(view_key, view_key.upper())
        parts.append(types.Part.from_text(text=f"--- {label} ---"))
        img_bytes = load_image_bytes(img_path)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=PROMPTS["damage_assessment_system"],
                    temperature=_G["temperature"],
                    max_output_tokens=_G["max_tokens"],
                ),
            )
            raw = response.text
            raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if not match:
                raise ValueError(f"Gemini returned no valid JSON.\nRaw: {raw[:300]}")
            return json.loads(match.group())
        except Exception as e:
            err_str = str(e)
            retry_delay = None
            delay_match = re.search(r"retryDelay.*?(\d+)s", err_str)
            if delay_match:
                retry_delay = int(delay_match.group(1)) + 2
            if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                wait = retry_delay or (2 ** attempt * 15)
                print(f"   ⏳ rate limit / unavailable — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def select_cars_for_verification(label_dir: Path, seed: int = 42) -> dict[str, str]:
    """
    Returns {car_id: reason} for cars that need Gemini verification.
    reason = 'low_confidence' | 'clean_sample'
    """
    rng = random.Random(seed)
    to_verify = {}

    for f in sorted(label_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        report = json.loads(f.read_text(encoding="utf-8"))
        already_verified = "gemini_verdict" in report

        if already_verified:
            continue

        damaged  = report.get("damage_detected", False)
        avg_conf = avg_confidence(report)

        if damaged and avg_conf < HIGH_CONF_THRESHOLD:
            to_verify[f.stem] = "low_confidence"
        elif not damaged and rng.random() < CLEAN_SAMPLE_RATE:
            to_verify[f.stem] = "clean_sample"

    return to_verify


def process_car(car_id: str, reason: str, views: dict[str, str], label_file: Path) -> dict:
    ollama_report  = json.loads(label_file.read_text(encoding="utf-8"))
    ollama_damaged = ollama_report.get("damage_detected", False)

    gemini_report  = call_gemini(views)
    gemini_damaged = gemini_report.get("damage_detected", False)

    if ollama_damaged and gemini_damaged:
        consensus = "confirmed_damaged"
        final_label = 1
    elif not ollama_damaged and not gemini_damaged:
        consensus = "confirmed_clean"
        final_label = 0
    else:
        consensus = "uncertain"
        final_label = None

    ollama_report["gemini_verdict"] = {
        "damage_detected": gemini_damaged,
        "damage_items":    gemini_report.get("damage_items", []),
        "consensus":       consensus,
        "final_label":     final_label,
        "reason_checked":  reason,
    }
    label_file.write_text(json.dumps(ollama_report, indent=2), encoding="utf-8")

    return {
        "car_id":      car_id,
        "ollama":      ollama_damaged,
        "gemini":      gemini_damaged,
        "consensus":   consensus,
        "final_label": final_label,
    }


def run_verification(workers: int = 3):
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set.\n"
            "1. Get free key: https://aistudio.google.com\n"
            "2. Add GEMINI_API_KEY=... to your .env file"
        )

    image_index = build_image_index(IMAGE_DIRS)
    to_verify   = select_cars_for_verification(LABEL_DIR)

    print(f"🔍  Cars to verify: {len(to_verify)}")
    breakdown = {}
    for r in to_verify.values():
        breakdown[r] = breakdown.get(r, 0) + 1
    print(f"   low_confidence: {breakdown.get('low_confidence', 0)}")
    print(f"   clean_sample:   {breakdown.get('clean_sample', 0)}\n")

    lock    = threading.Lock()
    results = []

    def job(car_id, reason):
        views      = image_index.get(car_id, {})
        label_file = LABEL_DIR / f"{car_id}.json"
        if not views:
            return {"car_id": car_id, "error": "no images found"}
        try:
            return process_car(car_id, reason, views, label_file)
        except Exception as e:
            return {"car_id": car_id, "error": str(e)}

    total = len(to_verify)
    done  = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(job, cid, r): cid for cid, r in to_verify.items()}
        for future in as_completed(futures):
            res = future.result()
            done += 1
            with lock:
                results.append(res)
                if "error" in res:
                    print(f"[{done}/{total}] ❌  {res['car_id']}: {res['error']}")
                else:
                    sym = {"confirmed_damaged": "✅", "confirmed_clean": "✅", "uncertain": "⚠️ "}
                    print(f"[{done}/{total}] {sym.get(res['consensus'], '?')}  {res['car_id']}  "
                          f"ollama={res['ollama']} gemini={res['gemini']}  → {res['consensus']}")

    agreed    = sum(1 for r in results if r.get("consensus") in ("confirmed_damaged", "confirmed_clean"))
    uncertain = sum(1 for r in results if r.get("consensus") == "uncertain")
    errors    = sum(1 for r in results if "error" in r)
    print(f"\n{'─'*55}")
    print(f"  Agreed: {agreed}  |  Uncertain (excluded): {uncertain}  |  Errors: {errors}")
    print(f"{'─'*55}\n")
    print("Next: run `python prepare_dataset.py --verified` to rebuild CSV using consensus labels.")


def dry_run_single():
    """Test Gemini on one high-confidence damaged car before running full verification."""
    image_index = build_image_index(IMAGE_DIRS)

    # Pick the highest avg-confidence damaged car
    best = None
    best_conf = 0
    for f in LABEL_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        report = json.loads(f.read_text(encoding="utf-8"))
        if not report.get("damage_detected"):
            continue
        ac = avg_confidence(report)
        if ac > best_conf:
            best_conf = ac
            best = f.stem

    if not best:
        print("No damaged cars found in labels/")
        return

    views = image_index.get(best, {})
    print(f"🧪  Dry run on: {best}  (Ollama avg_conf={best_conf:.2f})")
    print(f"   Views: {list(views.keys())}\n")

    result = call_gemini(views)
    print("Gemini response:")
    print(json.dumps(result, indent=2))

    ollama = json.loads((LABEL_DIR / f"{best}.json").read_text(encoding="utf-8"))
    print(f"\nOllama damage_detected: {ollama.get('damage_detected')}")
    print(f"Gemini damage_detected: {result.get('damage_detected')}")
    agree = ollama.get("damage_detected") == result.get("damage_detected")
    print(f"Agreement: {'✅ YES' if agree else '⚠️  NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",     action="store_true", help="Run full verification (default: dry run on 1 car)")
    parser.add_argument("--workers", type=int, default=CFG["gemini_verify"]["workers"], help="Parallel Gemini calls")
    args = parser.parse_args()

    if args.run:
        run_verification(workers=args.workers)
    else:
        dry_run_single()
