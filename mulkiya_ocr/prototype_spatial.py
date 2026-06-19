"""Tier-2 prototype: box-aware spatial label->value binding.

The production extractor flattens OCR to a list of strings and keyword-scans,
so it mis-binds (color=label word, cc=weight). A Mulkiya is a fixed-layout card
where each field is "label : value" on one row (Arabic, right-to-left). This
prototype keeps RapidOCR boxes and binds each value to its label by ROW geometry:

  - cluster boxes into rows by y-center
  - within a row, order right->left (Arabic)
  - for a label keyword, the value is the remainder of the label's own box
    (label+value often land in one box, e.g. "رمادي اللون") OR the next box to
    its left on the same row.

Standalone — does not touch production. Compares spatial vs the current flat
extractor on the hand-verified spot images.

Usage:  python prototype_spatial.py vlm_check/spot_0.jpg
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

import ocr_simple_test as o  # reuse _fix_reversed_arabic_runs, digit helpers

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Arabic field labels (post reverse-fix forms) -> canonical field name.
LABELS = {
    "اللون": "color",
    "نوع اللوحة": "vehicle_type",   # value is خصوصي etc. (also a 'type' word)
    "نوع المركبة": "make_model",
    "المنشاء": "country_of_origin",
    "سعة المحرك": "engine_cc",
    "الوزن فارغ": "empty_weight_kg",
    "الحمولة القصوى": "max_load_kg",
    "عدد الركاب": "seats",
    "سنة الصنع": "year",
    "سنة الطراز": "model_year",
    "عدد المحاور": "axles",
    "رقم اللوحة": "plate_number",
}
# Looser keyword fallbacks (OCR drops the leading word often).
KEYWORDS = {
    "color": ["اللون"],
    "vehicle_type": ["نوع اللوحة", "اللوحة نوع"],
    "make_model": ["نوع المركبة", "المركبة نوع", "المركبة"],
    "country_of_origin": ["المنشاء", "المنشأ"],
    "engine_cc": ["سعة المحرك", "المحرك سعة", "سعة"],
    "empty_weight_kg": ["الوزن فارغ", "فارغ الوزن", "فارغ"],
    "max_load_kg": ["الحمولة القصوى", "القصوى الحمولة", "الحمولة"],
    "seats": ["عدد الركاب", "الركاب عدد", "الركاب"],
    "year": ["سنة الصنع", "الصنع سنة", "الصنع", "الصعنع"],
    "model_year": ["سنة الطراز", "الطراز سنة", "الطراز"],
}
KNOWN_COLORS = ["أبيض", "ابيض", "أسود", "اسود", "أحمر", "احمر", "أزرق", "ازرق",
                "أخضر", "اخضر", "رمادي", "فضي", "ذهبي", "بيج", "أصفر", "اصفر",
                "بني", "برتقالي", "بنفسجي", "وردي"]


_digit_engine = None


def _get_digit_engine():
    """Default (ch+en) RapidOCR — reads Western digits better than the Arabic
    recogniser, which fragments them. Used on upscaled numeric cells."""
    global _digit_engine
    if _digit_engine is None:
        _digit_engine = o._create_ocr_engine()
    return _digit_engine


def _ocr_boxes(img):
    """RapidOCR (Arabic) -> list of boxes as dicts with full bbox + text."""
    eng = o._create_arabic_ocr_engine()
    res = eng(img)
    if res is None or res.boxes is None:
        return []
    out = []
    for box, txt in zip(res.boxes, res.txts):
        b = np.asarray(box, dtype=float)
        xs, ys = b[:, 0], b[:, 1]
        out.append({
            "cx": xs.mean(), "cy": ys.mean(),
            "x0": xs.min(), "x1": xs.max(), "y0": ys.min(), "y1": ys.max(),
            "text": o._fix_reversed_arabic_runs(str(txt)),
        })
    return out


def _read_digits_upscaled(img, x0, x1, y0, y1, scale=3):
    """Crop a value cell, upscale, OCR with the digit engine, return first int.

    The cell's digits are too small for reliable recognition at native res; a
    3x bicubic upscale pushes them above the recogniser's resolution floor."""
    H, W = img.shape[:2]
    pad_y = (y1 - y0) * 0.4
    yy0 = max(0, int(y0 - pad_y)); yy1 = min(H, int(y1 + pad_y))
    xx0 = max(0, int(x0)); xx1 = min(W, int(x1))
    if xx1 - xx0 < 8 or yy1 - yy0 < 8:
        return None
    cell = img[yy0:yy1, xx0:xx1]
    big = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res = _get_digit_engine()(big)
    if res is None or not res.txts:
        return None
    txt = " ".join(res.txts)
    txt = o._convert_arabic_indic_digits_to_ascii(txt)
    nums = re.findall(r"\d{2,6}", txt)
    return int(nums[0]) if nums else None


def _rows(boxes, tol_frac=0.5):
    """Cluster boxes into rows by y-center; each row ordered right->left."""
    if not boxes:
        return []
    ys = sorted(b["cy"] for b in boxes)
    med_gap = np.median(np.diff(ys)) if len(ys) > 1 else 20
    tol = max(12.0, med_gap * tol_frac + 10)
    rows = []
    for b in sorted(boxes, key=lambda b: b["cy"]):
        placed = False
        for r in rows:
            if abs(b["cy"] - r["y"]) <= tol:
                r["items"].append(b)
                r["y"] = np.mean([it["cy"] for it in r["items"]])
                placed = True
                break
        if not placed:
            rows.append({"y": b["cy"], "items": [b]})
    for r in rows:
        r["items"].sort(key=lambda b: -b["cx"])  # right -> left
    return rows


def _digits(text):
    text = o._convert_arabic_indic_digits_to_ascii(text)
    return re.findall(r"\d{1,6}", text)


def extract_spatial(img):
    boxes = _ocr_boxes(img)
    rows = _rows(boxes)
    fields: dict[str, object] = {}

    def find_label(keys):
        """Return (row, idx_of_label_box) for the first matching label."""
        for r in rows:
            for idx, b in enumerate(r["items"]):
                if any(k in b["text"] for k in keys):
                    return r, idx
        return None, None

    def text_after_label(keys):
        r, idx = find_label(keys)
        if r is None:
            return None
        lbl = r["items"][idx]
        rest = lbl["text"]
        for k in keys:
            rest = rest.replace(k, " ")
        rest = rest.replace(":", " ").strip()
        for cand in [rest] + [it["text"] for it in r["items"][idx + 1:]]:
            cand = cand.strip()
            if cand:
                return cand
        return None

    def numeric_after_label(keys, lo, hi):
        """Locate the label cell; the value is the IMMEDIATE box to its left
        (RTL "value : label"). Upscale that one cell and OCR digits — cropping
        only the adjacent cell avoids the card's second column bleeding in."""
        r, idx = find_label(keys)
        if r is None:
            return None
        lbl = r["items"][idx]
        h = lbl["y1"] - lbl["y0"]
        neighbor = r["items"][idx + 1] if idx + 1 < len(r["items"]) else None
        if neighbor is not None and (lbl["x0"] - neighbor["x1"]) < 4 * h:
            # value is its own box just left of the label
            x0, x1 = neighbor["x0"], neighbor["x1"]
        else:
            # value shares the label box, or no neighbour: a narrow window left of label
            x0, x1 = lbl["x0"] - 3 * h, lbl["x0"]
        val = _read_digits_upscaled(img, x0, x1, lbl["y0"], lbl["y1"])
        if val is not None and lo <= val <= hi:
            return val
        # fallback: digits already present on the label box / its left neighbour
        cands = [lbl["text"]] + ([neighbor["text"]] if neighbor else [])
        for cand in cands:
            for d in _digits(cand):
                if lo <= int(d) <= hi:
                    return int(d)
        return None

    # color: a known color value anywhere (prefer its label row, else whole doc)
    for r in rows:
        j = " ".join(it["text"] for it in r["items"])
        hit = next((c for c in KNOWN_COLORS if c in j), None)
        if hit:
            fields["color"] = hit
            break
    fields.setdefault("color", None)

    fields["vehicle_type"] = "خصوصي" if any("خصوصي" in b["text"] for b in boxes) else None
    fields["engine_cc"] = numeric_after_label(KEYWORDS["engine_cc"], 200, 9999)
    fields["empty_weight_kg"] = numeric_after_label(KEYWORDS["empty_weight_kg"], 200, 9999)
    fields["max_load_kg"] = numeric_after_label(KEYWORDS["max_load_kg"], 100, 99999)
    fields["seats"] = numeric_after_label(KEYWORDS["seats"], 1, 80)
    fields["year"] = numeric_after_label(KEYWORDS["year"], 1980, 2030)
    return fields, rows


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [Path("vlm_check/spot_0.jpg")]
    for p in paths:
        img = cv2.imread(str(p))
        fields, rows = extract_spatial(img)
        print(f"=== {p.name} — spatial extraction ===")
        for k, v in fields.items():
            print(f"  {k:18}: {v}")
        print(f"  ({len(rows)} rows detected)")


if __name__ == "__main__":
    main()
