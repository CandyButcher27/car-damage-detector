"""
Car Damage Detector — Ollama Cloud VLM (Multi-View)
====================================================
Accepts up to 4 views of a car (front, back, left, right) and returns a
structured JSON damage report with type, location, and confidence score.

CLI Usage
---------
    python damage_detect.py --front front.jpg --back back.jpg --left left.jpg --right right.jpg
    python damage_detect.py --front front.jpg --left left.jpg
    python damage_detect.py --front front.jpg --back back.jpg --save report.json
    python damage_detect.py --front front.jpg --json-only

Import Usage
------------
    from damage_detect import detect
    result = detect(front="front.jpg", back="back.jpg", left="left.jpg", right="right.jpg")
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from config import CFG, PROMPTS

load_dotenv()

API_KEY  = os.getenv("OLLAMA_API_KEY", "")
MODEL    = os.getenv("OLLAMA_MODEL",   CFG["ollama"]["model"])
BASE_URL = os.getenv("OLLAMA_BASE_URL", CFG["ollama"]["base_url"])

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key=API_KEY)

SYSTEM      = PROMPTS["damage_assessment_system"]
_IMG_CFG    = CFG["image"]
_OLLAMA_CFG = CFG["ollama"]


def load_image_as_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path)
    if img.mode in ("RGBA", "P", "LA", "L"):
        img = img.convert("RGB")

    w, h = img.size
    upscale_target = _IMG_CFG["upscale_target"]
    if max(w, h) < upscale_target:
        scale = upscale_target / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"      ↑  Upscaled {w}×{h}  →  {new_w}×{new_h}")

    img = img.filter(ImageFilter.UnsharpMask(
        radius=_IMG_CFG["unsharp_radius"],
        percent=_IMG_CFG["unsharp_percent"],
        threshold=_IMG_CFG["unsharp_threshold"],
    ))
    img = ImageEnhance.Contrast(img).enhance(_IMG_CFG["contrast_factor"])

    max_px = _IMG_CFG["max_px"]
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=_IMG_CFG["jpeg_quality"])
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


VIEW_LABELS = {
    "front": "FRONT VIEW",
    "back":  "BACK VIEW",
    "left":  "LEFT SIDE VIEW",
    "right": "RIGHT SIDE VIEW",
}


def _build_user_content(views: dict[str, str]) -> list[dict]:
    content = [{
        "type": "text",
        "text": (
            f"I am providing {len(views)} image(s) of the same vehicle. "
            "Each image is labelled with its view direction. "
            "Analyze ALL images for physical damage and return the structured JSON report."
        ),
    }]
    for view_key, image_path in views.items():
        label = VIEW_LABELS.get(view_key, view_key.upper())
        print(f"   📷  Loading [{label}] : {image_path}")
        data_uri = load_image_as_base64(image_path)
        content.append({"type": "text", "text": f"--- {label} ---"})
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    return content


def detect(
    front: str | None = None,
    back:  str | None = None,
    left:  str | None = None,
    right: str | None = None,
) -> dict:
    if not API_KEY:
        raise ValueError(
            "OLLAMA_API_KEY is not set.\n"
            "1. Copy example.env → .env\n"
            "2. Paste your key from https://ollama.com/settings/keys"
        )

    views: dict[str, str] = {}
    for key, path in [("front", front), ("back", back), ("left", left), ("right", right)]:
        if path is not None:
            views[key] = path

    if not views:
        raise ValueError("At least one image must be provided (--front/--back/--left/--right).")

    print(f"\n🚗  Analyzing {len(views)} view(s): {', '.join(views.keys())}")
    print(f"🤖  Model : {MODEL}  |  Endpoint : {BASE_URL}\n")

    user_content = _build_user_content(views)

    print("\n⏳  Sending request to model…")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        temperature=_OLLAMA_CFG["temperature"],
        max_tokens=_OLLAMA_CFG["max_tokens"],
    )

    raw = response.choices[0].message.content
    print("✅  Response received.\n")

    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return valid JSON.\nRaw output:\n{raw}")

    report = json.loads(match.group())
    report.setdefault("total_views_analyzed", len(views))
    return report


def pretty_print(report: dict) -> None:
    hr    = "─" * 65
    items = report.get("damage_items", [])
    det   = report.get("damage_detected", False)
    views = report.get("total_views_analyzed", "?")

    print(hr)
    print(f"  DAMAGE DETECTED     : {'⚠️  YES' if det else '✅  None found'}")
    print(f"  VIEWS ANALYZED      : {views}")
    print(f"  TOTAL DAMAGE ITEMS  : {len(items)}")
    print(hr)

    for i, item in enumerate(items, 1):
        conf  = item.get("confidence_score", 0)
        bar   = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        print(f"\n  [{i}]  {item.get('type', 'Unknown damage').upper()}")
        print(f"       Location   : {item.get('location', 'Unknown')}")
        print(f"       View       : {item.get('source_view', 'Unknown')}")
        print(f"       Confidence : {bar}  {conf:.0%}")

    print(f"\n{hr}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect car damage from up to 4 views using Ollama cloud VLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python damage_detect.py --front f.jpg --back b.jpg\n"
            "  python damage_detect.py --front f.jpg --save report.json\n"
            "  python damage_detect.py --front f.jpg --json-only\n"
        ),
    )
    parser.add_argument("--front",     "-f", default=None, metavar="PATH")
    parser.add_argument("--back",      "-b", default=None, metavar="PATH")
    parser.add_argument("--left",      "-l", default=None, metavar="PATH")
    parser.add_argument("--right",     "-r", default=None, metavar="PATH")
    parser.add_argument("--model",     "-m", default=None, metavar="NAME")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--save",      "-s", default=None, metavar="FILE")
    args = parser.parse_args()

    if args.model:
        MODEL = args.model

    try:
        report = detect(front=args.front, back=args.back, left=args.left, right=args.right)
        if args.json_only:
            print(json.dumps(report, indent=2))
        else:
            pretty_print(report)
            print(json.dumps(report, indent=2))
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\n💾  Report saved to: {args.save}")
    except Exception as e:
        print(f"\n❌  Error: {e}", file=sys.stderr)
        sys.exit(1)
