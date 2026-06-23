from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING


if sys.platform.startswith("win"):
    # Avoid UnicodeEncodeError when printing Arabic text on some Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


if TYPE_CHECKING:
    import numpy as np


def _normalize_lines(result: object) -> list:
    if not result:
        return []

    def from_mapping(item: object) -> list:
        if not hasattr(item, "get"):
            return []

        texts = item.get("rec_texts")
        if texts is None:
            texts = []
        scores = item.get("rec_scores")
        if scores is None:
            scores = []
        boxes = None
        for key in ("rec_polys", "dt_polys", "rec_boxes"):
            candidate = item.get(key)
            if candidate is not None and len(candidate) > 0:
                boxes = candidate
                break
        if boxes is None:
            boxes = []
        lines = []
        for box, text, score in zip(boxes, texts, scores):
            if hasattr(box, "tolist"):
                box = box.tolist()
            lines.append((box, (text, float(score))))
        return lines

    if isinstance(result, list) and result:
        first = result[0]
        # PaddleOCR 3.x returns one mapping-like OCRResult per input.
        mapped = from_mapping(first)
        if mapped:
            return mapped
        # Some versions may return a flat list of [box, (text, conf)] entries
        # where (text, conf) is a tuple (not a list). Narrowed to tuple to
        # avoid ambiguity with our RapidOCR wrapper output.
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and isinstance(first[1], tuple)
            and len(first[1]) == 2
        ):
            return result
        # Typical PaddleOCR output is list-per-image.
        if isinstance(first, list):
            return first

    return []


def _create_ocr_engine():
    from rapidocr import RapidOCR
    return RapidOCR()


_ENGINE = None


def _get_engine():
    """Process-wide singleton RapidOCR engine. Loading the det/cls/rec ONNX
    models costs ~4s; in-process callers (the API) reuse one engine across
    requests instead of paying that on every call."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _create_ocr_engine()
    return _ENGINE


def preload() -> None:
    """Warm the singleton engine (and its lazy recogniser) once at startup."""
    _get_engine()


def _run_ocr(engine, image, *, use_cls: bool = True):
    """Run RapidOCR v3 and return output in PaddleOCR 2.x-compatible format."""
    result = engine(image, use_cls=use_cls)
    if result is None or not result.txts:
        return [[]]
    lines = []
    boxes = getattr(result, "boxes", None)
    for i, (txt, conf) in enumerate(zip(result.txts, result.scores)):
        box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
        if box is not None:
            lines.append([box, (str(txt), float(conf))])
    return [lines]


def _get_annotation_font(font_size: int):
    """Return a Unicode-capable font for annotations, preferring Windows fonts."""
    from PIL import ImageFont

    font_candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for font_path in font_candidates:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _shape_arabic_for_display(text: str) -> str:
    if not text or not _contains_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return _reshape_arabic(text, enabled=True)


def _field_label_map() -> dict[str, str]:
    return {
        "plate_number": "رقم اللوحة",
        "vehicle_type": "نوع المركبة",
        "make": "الصانع",
        "model": "الطراز",
        "color": "اللون",
        "year": "سنة الصنع",
        "model_year": "سنة الطراز",
        "country_of_origin": "بلد المنشأ",
        "vin_or_chassis": "رقم الشاصي",
        "engine_number": "رقم المحرك",
        "engine_cc": "سعة المحرك",
        "empty_weight_kg": "الوزن فارغ",
        "max_load_kg": "الحمولة القصوى",
        "seats": "عدد المقاعد",
        "issue_date": "تاريخ الإصدار",
        "expiry_date": "تاريخ الانتهاء",
        "owner_name": "اسم المالك",
    }


def _normalize_field_text(value: object) -> str:
    if value is None:
        return ""
    text = _normalize_for_translation(str(value))
    text = text.replace("/", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u0600-\u06FF]", "", text)
    return text.lower()


def _annotation_label_for_line(text: str, data: dict[str, object]) -> str | None:
    field_labels = _field_label_map()
    normalized_text = _normalize_field_text(text)
    if not normalized_text:
        return None

    matches: list[tuple[int, str]] = []
    for field, label in field_labels.items():
        value = data.get(field)
        normalized_value = _normalize_field_text(value)
        if not normalized_value:
            continue

        if normalized_value == normalized_text:
            return label

        if normalized_value in normalized_text or normalized_text in normalized_value:
            matches.append((len(normalized_value), label))

        if field in {"plate_number", "year", "engine_cc", "empty_weight_kg", "max_load_kg", "seats"}:
            digits_text = re.sub(r"\D", "", normalized_text)
            digits_value = re.sub(r"\D", "", normalized_value)
            if digits_text and digits_value and digits_text == digits_value:
                return label

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]
    return None


def _annotation_field_for_line(text: str, data: dict[str, object]) -> tuple | None:
    """Return (field_key, label) for a line if it matches a known field value."""
    field_labels = _field_label_map()
    normalized_text = _normalize_field_text(text)
    if not normalized_text:
        return None

    matches: list[tuple[int, str]] = []
    for field, label in field_labels.items():
        value = data.get(field)
        normalized_value = _normalize_field_text(value)
        if not normalized_value:
            continue

        if normalized_value == normalized_text:
            return (field, label)

        if normalized_value in normalized_text or normalized_text in normalized_value:
            matches.append((len(normalized_value), field))

        if field in {"plate_number", "year", "engine_cc", "empty_weight_kg", "max_load_kg", "seats"}:
            digits_text = re.sub(r"\D", "", normalized_text)
            digits_value = re.sub(r"\D", "", normalized_value)
            if digits_text and digits_value and digits_text == digits_value:
                return (field, label)

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return (matches[0][1], field_labels.get(matches[0][1], matches[0][1]))
    return None


def _draw_boxes(image_bgr: "np.ndarray", lines: list, min_conf: float) -> "np.ndarray":
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    annotated = image_bgr.copy()
    for box, (text, confidence) in lines:
        if float(confidence) < min_conf:
            continue

        pts = np.array(box, dtype=np.float32)
        pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(16, int(round(min(pil_image.size) * 0.03)))
    font = _get_annotation_font(font_size)

    for box, (text, confidence) in lines:
        if float(confidence) < min_conf:
            continue

        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0 = int(min(xs))
        y0 = int(min(ys))
        # Keep the label text in the same postprocessed form used everywhere else.
        label_text = text
        if _contains_arabic(label_text):
            label_text = _reshape_arabic(label_text, enabled=True)
            label_text = _normalize_for_translation(label_text)
        label = f"{_shape_arabic_for_display(label_text)} ({float(confidence):.2f})"
        draw.text((x0, max(0, y0 - font_size - 2)), label, fill=(0, 0, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _draw_field_boxes(image_bgr: "np.ndarray", lines: list, data: dict[str, object], min_conf: float) -> "np.ndarray":
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    annotated = image_bgr.copy()
    for box, (_text, confidence) in lines:
        if float(confidence) < min_conf:
            continue

        pts = np.array(box, dtype=np.float32)
        pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(16, int(round(min(pil_image.size) * 0.03)))
    font = _get_annotation_font(font_size)

    for box, (text, confidence) in lines:
        if float(confidence) < min_conf:
            continue

        af = _annotation_field_for_line(text, data)
        if not af:
            continue
        field_key, label = af

        # Value for the field (may be None)
        value = data.get(field_key)

        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0 = int(min(xs))
        y0 = int(min(ys))
        # Draw Arabic label on the first line (shaped for display)
        arabic_label = _shape_arabic_for_display(label)
        eng_line = f"{field_key}: {value}" if value is not None else field_key

        # Draw two lines: Arabic label above, English key+value below it.
        y_label = max(0, y0 - font_size * 2 - 4)
        draw.text((x0, y_label), arabic_label, fill=(0, 0, 255), font=font)
        draw.text((x0, y_label + font_size + 2), eng_line, fill=(0, 128, 0), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _normalize_for_translation(text: str) -> str:
    # Make output stable for downstream translation / NLP.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0640", "")  # tatweel
    # Remove common bidi/control marks that can confuse downstream processing.
    text = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def _is_arabic_char(ch: str) -> bool:
    # Covers Arabic + Arabic Supplement + Arabic Extended-A + presentation forms.
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def _contains_arabic(text: str) -> bool:
    return any(_is_arabic_char(ch) for ch in text)


def _reshape_arabic(text: str, enabled: bool = True) -> str:
    """Reshape Arabic presentation-form glyphs into properly joined logical forms.

    arabic_reshaper reconnects isolated glyphs that OCR engines often emit as
    disconnected presentation forms (e.g. ي ر ا ك -> كاري).  Reshaping is applied
    once, right after OCR detection, so every downstream consumer (rule extractor,
    text files and translation output receives consistently joined Arabic text.
    """
    if not enabled or not text or not _contains_arabic(text):
        return text
    try:
        import arabic_reshaper

        return arabic_reshaper.reshape(text)
    except Exception:
        # If the dependency is not installed, pass through silently.
        return text


def _convert_arabic_indic_digits_to_ascii(s: str) -> str:
    """Convert Arabic-Indic digits to ASCII digits in a string."""
    if not s:
        return s
    mapping = {}
    # Arabic-Indic (U+0660..U+0669)
    for i, cp in enumerate(range(0x0660, 0x066A)):
        mapping[chr(cp)] = str(i)
    # Extended Arabic-Indic (U+06F0..U+06F9)
    for i, cp in enumerate(range(0x06F0, 0x06FA)):
        mapping[chr(cp)] = str(i)
    return "".join(mapping.get(ch, ch) for ch in s)


def _apply_ocr_postprocessing(lines: list, enable_reshape: bool = True) -> list:
    """Post-process all OCR lines: reshape Arabic, normalize, clean whitespace.

    This ensures every downstream step (extraction, translation, text files)
    work with consistently formatted Arabic text.
    """
    out = []
    for box, (text, conf) in lines:
        # Apply reshaping first (fixes disconnected glyphs).
        text = _reshape_arabic(text, enabled=enable_reshape)
        # Then apply general normalization.
        text = _normalize_for_translation(text)
        out.append((box, (text, conf)))
    return out


def _validate_and_note_data(data: dict) -> None:
    """In-place validation: add issues to 'validation_notes' if data looks wrong."""
    existing_notes = data.get("validation_notes")
    notes: list[str] = list(existing_notes) if isinstance(existing_notes, list) else []

    # VIN/Chassis: should be 11-20 chars when present.
    vin = data.get("vin_or_chassis")
    if vin and not isinstance(vin, str):
        notes.append(f"vin_or_chassis: not a string ({type(vin).__name__})")
    elif vin and (len(vin) < 11 or len(vin) > 20):
        notes.append(f"vin_or_chassis: length {len(vin)} outside typical VIN range 11-20")

    # Plate number: typically 3-7 digits.
    plate = data.get("plate_number")
    if plate and isinstance(plate, str):
        digits = re.sub(r"\D", "", plate)
        if len(digits) < 3 or len(digits) > 7:
            notes.append(f"plate_number: {len(digits)} digits (expected 3-7)")

    # Year: should be 4-digit or null.
    year = data.get("year")
    if year is not None:
        if not isinstance(year, int) or year < 1900 or year > 2100:
            notes.append(f"year: {year} outside valid range 1900-2100")

    # Engine CC: should be 200-10000.
    cc = data.get("engine_cc")
    if cc is not None and (not isinstance(cc, int) or cc < 200 or cc > 10000):
        notes.append(f"engine_cc: {cc} outside typical range 200-10000")

    # Weights: empty should be < max load.
    empty_w = data.get("empty_weight_kg")
    max_w = data.get("max_load_kg")
    if empty_w and max_w and isinstance(empty_w, int) and isinstance(max_w, int):
        if empty_w > max_w:
            notes.append(f"weights: empty_weight_kg ({empty_w}) > max_load_kg ({max_w})")

    # Seats: should be 1-80.
    seats = data.get("seats")
    if seats is not None and (not isinstance(seats, int) or seats < 1 or seats > 80):
        notes.append(f"seats: {seats} outside valid range 1-80")

    # Dates: should match DD/MM/YY or YYYY/MM/DD.
    for dk in ["issue_date", "expiry_date"]:
        dv = data.get(dk)
        if dv and isinstance(dv, str):
            if not re.search(r"\d{2,4}/\d{1,2}/\d{1,2}", dv):
                notes.append(f"{dk}: {dv} does not match date pattern")

    if notes:
        data["validation_notes"] = notes


def _assess_extraction_quality(data: dict) -> dict:
    """Decide whether the result is good enough to accept, or the user should
    re-capture the image.

    Pixel-blur (Laplacian variance) and OCR confidence both proved UNRELIABLE as
    quality signals on real Mulkiya photos: a clean scan scored the lowest blur
    variance of all, and RapidOCR reports high confidence even on rotated/garbled
    text. The only reliable signal is the EXTRACTION OUTCOME — how many critical
    fields came out in a valid form. This counts them.

    Returns {usable, valid_field_count, document_type, reason, message}.
    """
    doc_type = data.get("document_type", "other")

    def valid_vin(v) -> bool:
        if not isinstance(v, str):
            return False
        v = v.upper()
        return bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{11,20}", v)) and bool(re.search(r"[A-Z]", v)) and bool(re.search(r"\d", v))

    def valid_plate(v) -> bool:
        if v in (None, ""):
            return False
        d = re.sub(r"\D", "", str(v))
        return 3 <= len(d) <= 7

    def in_range(v, lo, hi) -> bool:
        try:
            return lo <= int(v) <= hi
        except (TypeError, ValueError):
            return False

    def valid_date(v) -> bool:
        return isinstance(v, str) and bool(re.fullmatch(r"\d{2,4}/\d{1,2}/\d{1,2}", v))

    checks = {
        "vin": valid_vin(data.get("vin_or_chassis")),
        "plate": valid_plate(data.get("plate_number")),
        "engine_cc": in_range(data.get("engine_cc"), 200, 10000),
        "weight": in_range(data.get("empty_weight_kg"), 100, 10000) or in_range(data.get("max_load_kg"), 100, 100000),
        "year": in_range(data.get("year"), 1980, 2030),
        "issue_date": valid_date(data.get("issue_date")),
        "expiry_date": valid_date(data.get("expiry_date")),
        "vehicle_type": data.get("vehicle_type") not in (None, ""),
    }
    valid_count = sum(1 for v in checks.values() if v)

    # Thresholds: a real readable Mulkiya front yields well over 4 valid fields.
    MIN_VALID = 4

    if doc_type == "driving_licence":
        usable, reason = False, "not_a_mulkiya"
        message = "This looks like a driving licence, not a Mulkiya. Please upload the front of the vehicle registration card."
    elif doc_type != "mulkiya":
        usable, reason = False, "not_a_mulkiya"
        message = "This does not appear to be a Mulkiya. Please upload a clear photo of the front of the vehicle registration card."
    elif valid_count < MIN_VALID:
        usable, reason = False, "low_quality"
        message = "The image is not clear enough to read the Mulkiya reliably. Please retake a sharp, well-lit, flat photo of the whole card."
    else:
        usable, reason, message = True, "ok", None

    return {
        "usable": usable,
        "valid_field_count": valid_count,
        "document_type": doc_type,
        "reason": reason,
        "message": message,
        "field_checks": checks,
    }


def _numeric_fields_complete(data: dict) -> bool:
    """True when the cheap (full-image + range) read already has every
    template-relevant field as a valid value, so the expensive positional-
    template crop/orient/refine pass (~6s) can be skipped. Conservative: any
    missing or out-of-range field returns False and the template runs.

    NOTE: trades a little binding accuracy for latency — on a clean card the
    range extractor can mis-bind `seats` (it may read the axle count). The
    template would correct that, but only fires here when something looks off."""
    def _vin_ok(v) -> bool:
        if not isinstance(v, str):
            return False
        s = re.sub(r"[^A-Za-z0-9]", "", v)
        return len(s) == 17 and any(c.isalpha() for c in s) and any(c.isdigit() for c in s)

    def _plate_ok(v) -> bool:
        return v not in (None, "") and 3 <= len(re.sub(r"\D", "", str(v))) <= 7

    def _rng(v, lo, hi) -> bool:
        try:
            return lo <= int(v) <= hi
        except (TypeError, ValueError):
            return False

    def _date_ok(v) -> bool:
        return isinstance(v, str) and bool(re.fullmatch(r"\d{2,4}/\d{1,2}/\d{1,2}", v))

    return (
        _plate_ok(data.get("plate_number"))
        and _vin_ok(data.get("vin_or_chassis"))
        and _rng(data.get("engine_cc"), 200, 10000)
        and _rng(data.get("empty_weight_kg"), 100, 10000)
        and _rng(data.get("max_load_kg"), 100, 100000)
        and _rng(data.get("seats"), 1, 9)
        and _rng(data.get("year"), 1980, 2030)
        and _date_ok(data.get("issue_date"))
        and _date_ok(data.get("expiry_date"))
    )


def _group_ocr_lines_by_field(lines: list[str]) -> dict[str, list[str]]:
    """Group OCR lines by detected field labels (Arabic labels often appear inline).

    Returns all lines in '_other' for direct use by downstream extractors.
    """
    return {"_other": lines}


def _fix_reversed_arabic_runs(text: str) -> str:
    # Some OCR engines output Arabic glyphs in reverse logical order.
    # Fix by reversing only Arabic-character runs; keep numbers/Latin intact.
    if not text:
        return text

    out: list[str] = []
    run: list[str] = []
    in_ar = None

    def flush() -> None:
        nonlocal run, in_ar
        if not run:
            return
        if in_ar:
            out.extend(reversed(run))
        else:
            out.extend(run)
        run = []

    for ch in text:
        ch_is_ar = _is_arabic_char(ch)
        if in_ar is None:
            in_ar = ch_is_ar
            run.append(ch)
            continue
        if ch_is_ar == in_ar:
            run.append(ch)
            continue
        flush()
        in_ar = ch_is_ar
        run.append(ch)
    flush()
    return "".join(out)


def _postprocess_ocr_line(
    text: str,
    lang: str,
    fix_arabic_reverse: bool,
    reshape_arabic: bool,
) -> str:
    """Full post-processing pipeline applied to every OCR line immediately after detection.

    Running this once at the source means the clean text propagates to all
    downstream consumers automatically: annotated image labels, text-file writers,
    and rule-based Mulkiya extractor.

    Optional pipeline steps:
      1. Reverse-run fix -- corrects glyphs emitted in wrong logical order by
         some OCR models. PaddleOCR's Arabic model already returns logical
         order, so this must stay off by default.
      2. Arabic reshaper -- useful for drawing connected glyphs on images, but
         it creates presentation-form characters that make rule extraction and
         JSON consumers harder to work with. Keep it off for stored OCR text.

    Both steps are no-ops for non-Arabic languages.
    """
    if not text:
        return text
    is_ar = lang.lower() in {"ar", "arabic"}
    if fix_arabic_reverse and is_ar:
        text = _fix_reversed_arabic_runs(text)
    text = _reshape_arabic(text, enabled=reshape_arabic and is_ar)
    return text


def _sort_lines(lines: list, rtl: bool) -> list:
    # Row-aware reading order:
    # 1) cluster boxes into rows using y-centers
    # 2) sort rows top-to-bottom
    # 3) sort within each row left-to-right or right-to-left
    def geom(item: object) -> tuple:
        box, (_text, _conf) = item
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        yc = (y0 + y1) / 2.0
        h = max(1.0, y1 - y0)
        return x0, x1, y0, y1, yc, h

    items = [(item, geom(item)) for item in lines]
    if not items:
        return []

    heights = sorted(g[-1] for _i, g in items)
    median_h = heights[len(heights) // 2]
    row_tol = max(8.0, 0.6 * median_h)

    rows: list[dict] = []
    for item, (x0, x1, y0, y1, yc, h) in sorted(items, key=lambda t: t[1][4]):
        placed = False
        for row in rows:
            row_yc = float(row["yc"])
            if abs(yc - row_yc) <= row_tol:
                row["items"].append((item, x0, x1, y0, y1, yc))
                n = float(row["n"])
                row["yc"] = (row_yc * n + yc) / (n + 1.0)
                row["n"] = n + 1.0
                placed = True
                break
        if not placed:
            rows.append({"yc": yc, "n": 1.0, "items": [(item, x0, x1, y0, y1, yc)]})

    sorted_items: list = []
    for row in sorted(rows, key=lambda r: float(r["yc"])):
        row_items = row["items"]
        if rtl:
            row_items = sorted(row_items, key=lambda t: t[2], reverse=True)  # x1 desc
        else:
            row_items = sorted(row_items, key=lambda t: t[1])  # x0 asc
        sorted_items.extend([it for (it, *_rest) in row_items])

    return sorted_items


def _detect_document_type(lines: list[str], has_vehicle_specs: bool = False) -> str:
    """Classify the OCR'd document by anchor strings + a field fingerprint.

    The card/non-card model passes driving licences (a licence IS a card) and
    the mulkiya model only tells front-from-back, so neither rejects a
    non-Mulkiya document. This anchor gate does: a genuine Omani Mulkiya front
    always carries "MOTOR VEHICLE LICENSE" / "رخصة مركبة" / "نوع اللوحة";
    a driving licence carries "DRIVING LICENCE" / "LICENCE NUMBER" / "رخصة سياقة".

    Anchor text alone is brittle — the English pass misses Arabic-only anchors,
    so a real Mulkiya can read as "other". `has_vehicle_specs` is the safety net:
    engine displacement / kerb weight / max load simply do not exist on a licence
    or ID card, so their presence means this IS a vehicle registration regardless
    of whether the header text was recognised.

    Returns "mulkiya", "driving_licence", or "other". When a frame contains both
    (e.g. a licence photographed next to a Mulkiya), Mulkiya wins so the
    registration is still processed.
    """
    joined = " ".join(lines)
    latin = re.sub(r"[^A-Za-z]", "", joined).upper()

    mulkiya_latin = "MOTORVEHICLELIC" in latin
    mulkiya_ar = any(a in joined for a in ("رخصة مركبة", "مركبة رخصة", "نوع اللوحة", "رقم اللوحة"))
    licence_latin = any(a in latin for a in ("DRIVINGLICENCE", "DRIVINGLICENSE", "VEHICLEDRIVING", "LICENCENUMBER", "LICENSENUMBER"))
    licence_ar = any(a in joined for a in ("رخصة سياقة", "سياقة رخصة", "رخصة قيادة"))

    is_mulkiya = mulkiya_latin or mulkiya_ar or has_vehicle_specs
    is_licence = licence_latin or licence_ar

    if is_mulkiya:
        return "mulkiya"
    if is_licence:
        return "driving_licence"
    return "other"


def _do_extract_vin(text: str) -> str | None:
    """Extract VIN/chassis number from a text string. Shared by all extractors."""
    single_artifacts = {
        "VEHICLE", "MOTOR", "ENGINE", "LICENSE", "LCENSE", "TIRAFIIC",
        "SULTANATE", "OMAN", "POLICE", "ROYA", "ROYAL", "KINGDOM",
        "GENERALOFTRAFFIC", "GENERAL", "TRAFFIC",
    }
    compound_artifacts = {
        "VEHICLEMOTOR", "MOTORVEHICLE", "ENGINEMOTOR", "MOTORENGINE",
        "POLICEOMATIC", "SULTANATEOMAN", "DIRGENERALOFTRAFFIC",
    }
    all_artifacts = single_artifacts | compound_artifacts
    candidates: list[tuple[str, int]] = []
    for m in re.finditer(r"(?:[A-Za-z0-9]{2,}[\s\-]*)+", text):
        raw = m.group(0)
        cand = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
        if 11 <= len(cand) <= 20:
            is_artifact = any(
                cand == a or cand.startswith(a) or cand.endswith(a)
                for a in single_artifacts
            ) or cand in all_artifacts
            if not is_artifact:
                candidates.append((cand, len(cand)))
    for m in re.finditer(r"[A-Za-z0-9]{11,20}", text):
        cand = m.group(0).upper()
        if cand not in all_artifacts and not any(
            cand.startswith(a) or cand.endswith(a) for a in single_artifacts
        ):
            candidates.append((cand, len(cand)))
    if not candidates:
        return None
    uniq = {c: l for c, l in candidates}
    exact_17 = [
        c for c, l in uniq.items()
        if l == 17 and re.search(r"[A-Z]", c) and re.search(r"\d", c)
    ]
    if exact_17:
        return exact_17[0]
    letter_start = {c: l for c, l in uniq.items() if c and c[0].isalpha()}
    pool = letter_start if letter_start else uniq
    return max(pool.items(), key=lambda x: x[1])[0]


def _extract_mulkya_rulebased(lines: list[str]) -> dict:
    # Lightweight heuristic extractor for Omani Mulkiya-like layouts.
    joined = "\n".join(lines)
    joined_ascii = _convert_arabic_indic_digits_to_ascii(joined)
    lines_ascii = [_convert_arabic_indic_digits_to_ascii(ln) for ln in lines]

    field_label_terms = [
        "اللوحة",
        "الوحة",
        "نوع",
        "المركبة",
        "اللون",
        "المنشاء",
        "سنة",
        "الطرار",
        "الصنع",
        "الصلع",
        "المحرك",
        "الوزن",
        "فارغ",
        "الحمولة",
        "الركاب",
        "المحاور",
        "الشاص",
        "الشاصي",
        "القاعدة",
        "الاعدة",
        "المحرة",
        "الرخصة",
        "صلاحية",
    ]

    def is_field_label(line: str) -> bool:
        return any(term in line for term in field_label_terms)

    def find_after(keyword: str) -> str | None:
        for i, ln in enumerate(lines):
            if keyword in ln and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                return nxt or None
        return None

    def find_number_near(
        keyword: str,
        max_lookahead: int = 4,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int | None:
        for i, ln in enumerate(lines):
            if keyword in ln:
                for j in range(i, min(i + max_lookahead, len(lines))):
                    candidate_line = lines[j]
                    if j > i and is_field_label(candidate_line):
                        break
                    for m in re.finditer(r"\b\d{1,5}\b", candidate_line):
                        try:
                            value = int(m.group(0))
                        except Exception:
                            continue
                        if min_value is not None and value < min_value:
                            continue
                        if max_value is not None and value > max_value:
                            continue
                        return value
        return None

    def find_date() -> list[str]:
        out: list[str] = []
        for ln in lines_ascii:
            for m in re.finditer(r"\b\d{2,4}/\d{1,2}/\d{1,2}\b", ln):
                value = m.group(0)
                if value not in out:
                    out.append(value)
        return out

    def date_key(value: str) -> tuple[int, int, int] | None:
        m = re.fullmatch(r"(\d{2,4})/(\d{1,2})/(\d{1,2})", value)
        if not m:
            return None
        y, mo, d = (int(part) for part in m.groups())
        if y < 100:
            y += 2000
        return (y, mo, d)

    def numeric_tokens(text: str) -> list[int]:
        text = _convert_arabic_indic_digits_to_ascii(text)
        text = re.sub(r"\b\d{2,4}/\d{1,2}/\d{1,2}\b", " ", text)
        values: list[int] = []
        for raw in re.findall(r"\d+", text):
            pieces = [raw]
            if len(raw) == 8:
                left, right = raw[:4], raw[4:]
                if 100 <= int(left) <= 100000 and 100 <= int(right) <= 100000:
                    pieces = [left, right]
            for piece in pieces:
                try:
                    values.append(int(piece))
                except Exception:
                    continue
        return values

    all_numbers = numeric_tokens(joined_ascii)

    plate_number = None
    plate_text = None
    # Try plate-specific label variations. Do not use generic "رقم" because
    # Mulkiya has many other numbered fields.
    plate_labels = ["اللوحة", "الوحة", "رقم اللوحة", "رقم الوحة", "ﺍﻟﻠﻮﺣﺔ"]
    plate_token_skip = {"رقم", "اللوحة", "الوحة", "نوع", "خصوصي"}
    for i, ln in enumerate(lines):
        if not any(label in ln for label in plate_labels) or "نوع" in ln:
            continue
        window = " ".join(lines[max(0, i - 2) : min(len(lines), i + 4)])
        for m in re.finditer(r"\b\d{3,7}\b", window):
            if re.search(r"\d{2,4}/\d{1,2}/\d{1,2}", window):
                continue
            plate_number = m.group(0)
            break
        if plate_number:
            break
        for token in re.findall(r"(?<![\u0600-\u06FF])[\u0621-\u064A]{1,3}(?![\u0600-\u06FF])", window):
            if token not in plate_token_skip:
                plate_text = token
                break
        break

    # Prefer explicit plate pattern: 5 digits followed by an Arabic letter
    try:
        pat = re.search(r"([\u0660-\u0669\u06F0-\u06F90-9]{5}\s*[\u0600-\u06FF])", joined_ascii)
        if pat:
            raw = pat.group(1)
            # Extract the 5-digit portion and the Arabic letter
            m2 = re.search(r"([\u0660-\u0669\u06F0-\u06F90-9]{5})", raw)
            m3 = re.search(r"([\u0600-\u06FF])", raw)
            if m2 and m3:
                digits = _convert_arabic_indic_digits_to_ascii(m2.group(1))
                arabic_letter = m3.group(1)
                plate_number = f"{digits}{arabic_letter}"
    except Exception:
        pass

    if not plate_number:
        # The Arabic recognizer often sees the Arabic plate letters while the
        # English recognizer sees the numeric portion. Pick the first plausible
        # 3-7 digit token that is not a date/year/vehicle-spec value.
        plate_candidates = [value for value in all_numbers if 10000 <= value <= 9999999]
        plate_candidates.extend(value for value in all_numbers if 100 <= value <= 9999)
        for value in plate_candidates:
            if 100 <= value <= 9999999:
                if 1900 <= value <= 2100:
                    continue
                plate_number = str(value)
                break

    vin_or_chassis = None
    label_keys = ["الشاص", "الشاصي", "شاصي", "chassis", "vin", "رقم القاعدة", "القاعدة", "الاعدة"]
    for i, ln in enumerate(lines):
        if any(k in ln.lower() for k in label_keys if isinstance(k, str)):
            window = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 8)])
            cand = _do_extract_vin(window)
            if cand:
                vin_or_chassis = cand
                break
    if not vin_or_chassis:
        vin_or_chassis = _do_extract_vin(joined_ascii)

    make = None
    for brand in ["تويوتا", "نيسان", "هيونداي", "كيا", "هوندا", "مرسيدس", "بي ام", "BMW", "LEXUS", "لكزس"]:
        if brand in joined:
            make = brand
            break
    model = None
    for mdl in ["كورولا", "كامري", "يارس", "ألتِيما", "التيما", "صني", "باترول",
                "لاندكروزر", "برادو", "اكورد", "سيفيك", "النترا", "سوناتا"]:
        if mdl in joined:
            model = mdl
            break
    # Color: match a known value as a substring of any line FIRST — the value
    # often shares a line with its label (e.g. "رمادي اللون"). find_after()
    # alone returns the *next* line, which is usually another field's label.
    _known_colors = ["أبيض", "ابيض", "أسود", "اسود", "أحمر", "احمر", "أزرق", "ازرق",
                     "أخضر", "اخضر", "رمادي", "فضي", "ذهبي", "بيج", "أصفر", "اصفر",
                     "بني", "برتقالي", "بنفسجي", "وردي"]
    color = None
    for ln in lines:
        for col in _known_colors:
            if col in ln:
                color = col
                break
        if color:
            break
    if color is None:
        cand = find_after("اللون")
        if cand and not is_field_label(cand):
            color = cand
    country_of_origin = None
    if "الولايات" in joined and ("المتحدة" in joined or "الامريكية" in joined):
        country_of_origin = "الولايات المتحدة الامريكية"

    vehicle_type = "خصوصي" if "خصوصي" in joined else None

    engine_cc = find_number_near("المحرك", min_value=200, max_value=10000)
    empty_weight_kg = find_number_near("فارغ", min_value=100, max_value=10000) or find_number_near("الوزن", min_value=100, max_value=10000)
    max_load_kg = find_number_near("الحمولة", min_value=100, max_value=100000)
    seats = find_number_near("الركاب", max_lookahead=3, min_value=1, max_value=80)
    engine_number = "NIL" if re.search(r"\bNIL\b", joined_ascii, flags=re.IGNORECASE) else None

    year = None
    model_year = None
    # "سنة الصنع" (year of manufacture). RapidOCR garbles the label
    # inconsistently (الصعنع / الصلع), so match several observed variants.
    _manufacture_labels = ("الصنع", "الصلع", "الصعنع", "لصنع", "صنع")
    for i, ln in enumerate(lines):
        if any(label in ln for label in _manufacture_labels):
            for j in range(max(0, i - 1), min(i + 5, len(lines))):
                m = re.search(r"\b\d{4}\b|\b0?\d{2,3}\b", lines_ascii[j])
                if not m:
                    continue
                try:
                    raw = m.group(0)
                    y = int(raw)
                except Exception:
                    continue
                if y >= 1000:
                    year = y
                    break
                if 0 <= y <= 30:
                    year = 2000 + y
                    break
                if 31 <= y <= 99:
                    year = 1900 + y
                    break
            if year is not None:
                break

    spec_numbers = [value for value in all_numbers if value != _coerce_int(plate_number)]
    spec_years = [value for value in spec_numbers if 1900 <= value <= 2100]
    if spec_years:
        model_year = spec_years[0]
        if year is None:
            year = spec_years[-1] if len(spec_years) > 1 else spec_years[0]

    if engine_cc is None:
        for value in spec_numbers:
            if 200 <= value <= 10000 and not (1900 <= value <= 2100):
                engine_cc = value
                break

    if empty_weight_kg is None or max_load_kg is None:
        weight_candidates = [
            value
            for value in spec_numbers
            if 100 <= value <= 100000 and value != engine_cc and value not in {model_year, year}
        ]
        if empty_weight_kg is None and weight_candidates:
            empty_weight_kg = weight_candidates[0]
        if max_load_kg is None:
            if len(weight_candidates) >= 2:
                max_load_kg = weight_candidates[1]
            elif len(weight_candidates) == 1 and empty_weight_kg == weight_candidates[0]:
                max_load_kg = weight_candidates[0]

    if seats is None:
        for value in spec_numbers:
            if 1 <= value <= 80 and value != 2:
                seats = value
                break

    dates = find_date()
    issue_date = None
    expiry_date = None
    if dates:
        dated = [(date_key(value), value) for value in dates]
        dated = [(key, value) for key, value in dated if key is not None]
        if len(dated) >= 2:
            dated.sort(key=lambda item: item[0])
            issue_date = dated[0][1]
            expiry_date = dated[-1][1]
        else:
            expiry_date = dates[-1]

    # Vehicle-spec fingerprint: these fields exist on a Mulkiya but not on a
    # driving licence / ID, so they rescue real Mulkiyas whose Arabic-only
    # header anchors the English OCR pass didn't read.
    has_vehicle_specs = any(v is not None for v in (engine_cc, empty_weight_kg, max_load_kg)) or (
        seats is not None and vehicle_type is not None
    )
    document_type = _detect_document_type(lines, has_vehicle_specs=has_vehicle_specs)

    return {
        "document_type": document_type,
        "is_mulkiya": document_type == "mulkiya",
        "plate_number": plate_number,
        "plate_text": plate_text,
        "vehicle_type": vehicle_type,
        "make": make,
        "model": model,
        "color": color,
        "year": year,
        "model_year": model_year,
        "country_of_origin": country_of_origin,
        "vin_or_chassis": vin_or_chassis,
        "engine_number": engine_number,
        "engine_cc": engine_cc,
        "empty_weight_kg": empty_weight_kg,
        "max_load_kg": max_load_kg,
        "seats": seats,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "owner_name": None,
        "notes": "Heuristic extraction.",
    }


def _split_merged_weight(n: int) -> tuple[int, int] | None:
    """Split a 7-8 digit number that is two weight values fused by the OCR detector.

    Mulkiya weight rows have two values (empty_weight + max_load) on the same
    horizontal band. When the OCR detector merges them into one token (e.g.
    5201060 or 20002000), we need to find the correct split point.

    Strategy: collect all valid splits, then prefer the one whose parts have
    equal digit length (symmetric split = more likely the two values have similar
    magnitudes, as on a real card). Fall back to the first valid split.
    """
    s = str(n)
    valid: list[tuple[int, int, int]] = []  # (left, right, |len_diff|)
    for i in range(3, len(s) - 2):
        left, right = int(s[:i]), int(s[i:])
        if 100 <= left <= 5000 and 100 <= right <= 50000:
            valid.append((left, right, abs(len(s[:i]) - len(s[i:]))))
    if not valid:
        return None
    valid.sort(key=lambda x: x[2])  # prefer most symmetric split
    return valid[0][0], valid[0][1]


def _assign_dates_sorted(dates: list[str]) -> tuple[str | None, str | None]:
    """Return (issue_date, expiry_date) sorted chronologically from a list of date strings."""
    def date_key(v: str) -> tuple:
        m = re.fullmatch(r"(\d{2,4})/(\d{1,2})/(\d{1,2})", v)
        if not m:
            return (9999, 0, 0)
        y, mo, d = (int(x) for x in m.groups())
        if y < 100:
            y += 2000
        return (y, mo, d)

    dated = [(date_key(d), d) for d in dates]
    dated = [(k, v) for k, v in dated if k[0] < 9999]
    dated.sort()
    if len(dated) >= 2:
        return dated[0][1], dated[-1][1]
    if len(dated) == 1:
        return None, dated[0][1]
    return None, None


def _weight_row_crop(image_bgr, token: dict, engine) -> tuple[int | None, int | None]:
    """Crop a horizontal band around a token and re-OCR to separate two weight values."""
    import cv2
    H, W = image_bgr.shape[:2]
    h = max(float(token['y1']) - float(token['y0']), 20.0)
    pad = h * 0.7
    y0 = max(0, int(token['y0'] - pad))
    y1 = min(H, int(token['y1'] + pad))
    row_crop = image_bgr[y0:y1, 0:W]
    row_crop = cv2.resize(row_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    result = engine(row_crop)
    if result is None or not result.txts:
        return None, None
    boxes = getattr(result, 'boxes', None)
    nums_with_x: list[tuple[int, float]] = []
    for i, (txt, conf) in enumerate(zip(result.txts, result.scores)):
        if float(conf) < 0.35:
            continue
        txt = _convert_arabic_indic_digits_to_ascii(str(txt))
        box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
        xc = float(sum(p[0] for p in box) / len(box)) if box else 0.0
        for m in re.finditer(r'\b(\d{2,5})\b', txt):
            n = int(m.group())
            if 100 <= n <= 50000:
                nums_with_x.append((n, xc))
    if len(nums_with_x) >= 2:
        nums_with_x.sort(key=lambda x: x[1])
        a, b_ = nums_with_x[0][0], nums_with_x[-1][0]
        return (min(a, b_), max(a, b_))
    return None, None


_RANGE_EXTRACTOR_ENABLED = os.getenv("UPSURE_RANGE_EXTRACTOR", "1") not in ("0", "false", "False")


def _extract_by_range_with_boxes(lines: list, image_bgr, engine=None) -> dict:
    """All-token numeric extractor: no keyword binding, uses value ranges + y-position.

    Designed for EN OCR output where Arabic labels are absent.  Handles the
    common detector cell-merge failure (e.g. 5201060 → 520 kg + 1060 kg) via
    a heuristic split and a row-crop re-OCR fallback.

    Returns a partial dict covering only numeric / structured fields.  Text
    fields (make, model, color, vehicle_type) come from the Arabic aux pass.
    """
    import numpy as np

    if not lines:
        return {}

    tokens: list[dict] = []
    for item in lines:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        box, rest = item
        if not (isinstance(rest, (list, tuple)) and len(rest) == 2):
            continue
        text, conf = rest
        if float(conf) < 0.25:
            continue
        try:
            b = np.asarray(box, dtype=float)
            yc = float(b[:, 1].mean())
            xc = float(b[:, 0].mean())
            y0_t = float(b[:, 1].min()); y1_t = float(b[:, 1].max())
            x0_t = float(b[:, 0].min()); x1_t = float(b[:, 0].max())
        except Exception:
            continue
        tokens.append({
            'text': _convert_arabic_indic_digits_to_ascii(str(text)),
            'raw': str(text),
            'yc': yc, 'xc': xc,
            'y0': y0_t, 'y1': y1_t, 'x0': x0_t, 'x1': x1_t,
        })

    if not tokens:
        return {}

    H = float(image_bgr.shape[0]) if image_bgr is not None else max(t['y1'] for t in tokens) + 1
    all_text = " ".join(t['text'] for t in tokens)
    date_re = re.compile(r'\b\d{2,4}/\d{1,2}/\d{1,2}\b')

    dates = date_re.findall(all_text)
    issue_date, expiry_date = _assign_dates_sorted(dates)

    all_raw = " ".join(t['raw'] for t in tokens)
    vin = _do_extract_vin(all_raw)

    # Build numeric token list; split merged weight tokens directly.
    # Merged-split results bypass the candidate pool — we know they are weights.
    num_tokens: list[dict] = []
    direct_weights: list[int] = []   # values from a successful merge split
    merged_weight_token: dict | None = None  # source token (for row-crop fallback)

    for t in tokens:
        nd = date_re.sub(' ', t['text'])
        for m in re.finditer(r'\b(\d{1,8})\b', nd):
            n = int(m.group())
            if 1_000_000 <= n <= 99_999_999:
                split = _split_merged_weight(n)
                if split:
                    direct_weights.extend(split)
                    if merged_weight_token is None:
                        merged_weight_token = t
                    continue  # do not add to num_tokens
            num_tokens.append({**t, 'val': n})

    # Years — 4-digit, 1990-2030
    # Mulkiya layout (y=0 at top): model_year row is ABOVE manufacture_year row.
    # Sort descending by y-center so largest yc (lowest on card) = manufacture year first.
    seen_years: set[int] = set()
    year_tokens = []
    for nt in sorted((x for x in num_tokens if 1990 <= x['val'] <= 2030 and len(str(x['val'])) == 4), key=lambda x: -x['yc']):
        if nt['val'] not in seen_years:
            year_tokens.append(nt)
            seen_years.add(nt['val'])

    # ALL year-range integers are reserved — prevents e.g. 2018 leaking into weight pool
    all_year_range: set[int] = {nt['val'] for nt in num_tokens if 1990 <= nt['val'] <= 2030}
    used: set[int] = set(all_year_range)
    year = year_tokens[0]['val'] if year_tokens else None
    model_year = year_tokens[1]['val'] if len(year_tokens) > 1 else None

    # Engine CC — 3-4 digit, 800-8999, not a year, in upper 65% of card
    cc_cut = H * 0.65
    cc_cands = [
        nt for nt in num_tokens
        if 800 <= nt['val'] <= 8999
        and nt['val'] not in used
        and not (1990 <= nt['val'] <= 2030)
        and nt['yc'] < cc_cut
    ]
    cc_cands.sort(key=lambda x: x['yc'])
    engine_cc = cc_cands[0]['val'] if cc_cands else None
    if engine_cc is not None:
        used.add(engine_cc)

    # Seats — 1-9, in lower 50% of card
    seat_cut = H * 0.5
    seat_cands = [
        nt for nt in num_tokens
        if 1 <= nt['val'] <= 9 and nt['val'] not in used and nt['yc'] > seat_cut
    ]
    seat_cands.sort(key=lambda x: -x['yc'])
    seats = seat_cands[0]['val'] if seat_cands else None
    if seats is not None:
        used.add(seats)

    # Weights — 100-5000, in lower 60% of card
    wt_cut = H * 0.4
    wt_cands = [
        nt for nt in num_tokens
        if 100 <= nt['val'] <= 5000 and nt['val'] not in used and nt['yc'] > wt_cut
    ]
    wt_cands.sort(key=lambda x: (x['yc'], x['xc']))

    empty_weight_kg: int | None = None
    max_load_kg: int | None = None

    if direct_weights:
        # Merge-split result: most reliable path — detector saw both values fused.
        direct_weights.sort()
        empty_weight_kg = direct_weights[0]
        max_load_kg = direct_weights[-1] if len(direct_weights) > 1 else None
    elif len(wt_cands) >= 2:
        vals = sorted(wt['val'] for wt in wt_cands[:2])
        empty_weight_kg, max_load_kg = vals[0], vals[1]
    elif len(wt_cands) == 1:
        empty_weight_kg = wt_cands[0]['val']
        if image_bgr is not None and engine is not None:
            src = merged_weight_token or wt_cands[0]
            w1, w2 = _weight_row_crop(image_bgr, src, engine)
            if w1 is not None and w2 is not None:
                empty_weight_kg, max_load_kg = min(w1, w2), max(w1, w2)
    elif image_bgr is not None and engine is not None and merged_weight_token is not None:
        w1, w2 = _weight_row_crop(image_bgr, merged_weight_token, engine)
        if w1 is not None and w2 is not None:
            empty_weight_kg, max_load_kg = min(w1, w2), max(w1, w2)

    # Plate number — 3-7 digit, in top 40% of card, not a year
    pl_cut = H * 0.4
    pl_cands = [
        nt for nt in num_tokens
        if 100 <= nt['val'] <= 9_999_999
        and not (1990 <= nt['val'] <= 2030)
        and nt['yc'] < pl_cut
    ]
    pl_cands.sort(key=lambda x: -len(str(x['val'])))
    plate_number = str(pl_cands[0]['val']) if pl_cands else None

    return {
        'plate_number': plate_number,
        'vin_or_chassis': vin,
        'year': year,
        'model_year': model_year,
        'engine_cc': engine_cc,
        'empty_weight_kg': empty_weight_kg,
        'max_load_kg': max_load_kg,
        'seats': seats,
        'issue_date': issue_date,
        'expiry_date': expiry_date,
    }


# ── Positional-template extractor ───────────────────────────────────────────
# Built empirically from 5 fully hand-labelled Omani Mulkiya fronts (labelImg /
# YOLO boxes), normalised to the field-cluster bbox and averaged. Field std-devs
# are tiny (cx≤0.086, cy≤0.019) → the front layout is fixed. Once the card is
# deskewed/oriented, each value sits at a known card-relative position, so we
# bind a value to a field by GEOMETRY, not by reading its Arabic label. See
# eval/yolo/build_template.py + template.json.
_MULKIYA_TEMPLATE = {
    "plate_number":       {"cx": 0.8568, "cy": 0.0474},
    "model_year":         {"cx": 0.2652, "cy": 0.3891},
    "engine_cc":          {"cx": 0.8975, "cy": 0.4752},
    "empty_weight_kg":    {"cx": 0.5421, "cy": 0.4888},
    "max_load_kg":        {"cx": 0.0512, "cy": 0.5027},
    "manufacturing_year": {"cx": 0.9165, "cy": 0.5852},
    "seats":              {"cx": 0.5828, "cy": 0.5986},
    "no_of_axles":        {"cx": 0.0726, "cy": 0.6192},
    "vin_or_chassis":     {"cx": 0.5978, "cy": 0.7052},
    "engine_number":      {"cx": 0.8702, "cy": 0.8074},
    "valid_from":         {"cx": 0.7506, "cy": 0.9218},
    "valid_until":        {"cx": 0.1999, "cy": 0.9428},
}

_TEMPLATE_EXTRACTOR_ENABLED = os.getenv("UPSURE_TEMPLATE_EXTRACTOR", "1") not in ("0", "false", "False")
# Max normalised distance (fraction of cluster diagonal) for a token to bind to
# a field slot. Loose enough to absorb residual skew, tight enough to reject the
# wrong column.
_TEMPLATE_MATCH_GATE = 0.28


def _loose_date(text: str) -> str | None:
    """Parse a date, tolerant of one OCR-dropped separator.

    Strict `YYYY/MM/DD` (any of / or -) first. Else, if the token still carries
    at least one separator (so we don't grab a bare plate/VIN number), recover an
    8-digit YYYYMMDD whose parts fall in valid date ranges — handles OCR reads
    like '202503-10' (the leading '-' was lost) → '2025/03/10'.
    """
    t = _convert_arabic_indic_digits_to_ascii(text)
    m = re.search(r"\b(\d{2,4})[/\-](\d{1,2})[/\-](\d{1,2})\b", t)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    if "-" in t or "/" in t:
        digits = re.sub(r"\D", "", t)
        if len(digits) == 8:
            y, mo, d = digits[:4], digits[4:6], digits[6:8]
            if 1990 <= int(y) <= 2035 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}/{mo}/{d}"
    return None


def _template_field_value(field: str, text: str):
    """Validate/parse a token's text for a given template field. Returns the
    typed value (int / str) or None if the token can't be that field."""
    t = _convert_arabic_indic_digits_to_ascii(text)
    loose = _loose_date(t)

    if field in ("valid_from", "valid_until"):
        return loose

    date_m = loose

    if field == "vin_or_chassis":
        return _do_extract_vin(t)

    if field == "engine_number":
        if re.search(r"\bNIL\b", t, re.IGNORECASE):
            return "NIL"
        m = re.search(r"\b[A-Za-z0-9]{5,20}\b", t)
        return m.group(0).upper() if m and re.search(r"\d", m.group(0)) else None

    # numeric fields — never read a digit out of a date token
    if date_m:
        return None
    # ...nor out of an alphanumeric token (VIN/plate-with-letter/engine-number):
    # plate, cc, weights, seats, axles, years are pure-digit cells, so a token
    # carrying Latin letters is the wrong field (prevents '4' leaking from a VIN).
    if re.search(r"[A-Za-z]", t):
        return None
    nums = [int(n) for n in re.findall(r"\d{1,8}", t)]
    if not nums:
        return None

    if field in ("model_year", "manufacturing_year"):
        for n in nums:
            if 1980 <= n <= 2030 and len(str(n)) == 4:
                return n
        return None
    if field == "plate_number":
        for n in nums:
            if 3 <= len(str(n)) <= 7 and not (1980 <= n <= 2030):
                return str(n)
        return None
    if field == "engine_cc":
        for n in nums:
            if 600 <= n <= 8999 and not (1980 <= n <= 2030):
                return n
        return None
    if field == "empty_weight_kg":
        # Kerb weight of an insured car/van — reject plate-sized garbage (37319).
        for n in nums:
            if 100 <= n <= 6000 and not (1980 <= n <= 2030):
                return n
        return None
    if field == "max_load_kg":
        for n in nums:
            if 100 <= n <= 50000 and not (1980 <= n <= 2030):
                return n
        return None
    if field == "seats":
        for n in nums:
            if 1 <= n <= 9:
                return n
        return None
    if field == "no_of_axles":
        for n in nums:
            if 1 <= n <= 6:
                return n
        return None
    return None


def _find_template_anchors(toks: list[dict]) -> dict:
    """Locate the format-UNIQUE fields we can trust regardless of position:
    VIN (17-ish alnum), the two validity dates (bottom row, split by x), and the
    plate (top-most short numeric). These pin the card's coordinate frame so the
    format-AMBIGUOUS fields (cc, weights, seats, years — all bare integers) can be
    placed by transform instead of guessed. Returns {field: (px, py)}."""
    anchors: dict[str, tuple[float, float]] = {}

    # VIN — unmistakable 14-18 char alnum with letters+digits.
    best_vin = None
    for t in toks:
        v = _do_extract_vin(_convert_arabic_indic_digits_to_ascii(t["raw"]))
        if v and 14 <= len(v) <= 18 and re.search(r"[A-Z]", v) and re.search(r"\d", v):
            if best_vin is None or len(v) > best_vin[1]:
                best_vin = (t, len(v))
    if best_vin:
        anchors["vin_or_chassis"] = (best_vin[0]["cx"], best_vin[0]["cy"])

    # Dates — take the two lowest date tokens; left = valid_until, right = valid_from.
    date_ts = [t for t in toks if _loose_date(t["raw"])]
    if len(date_ts) >= 2:
        bottom = sorted(date_ts, key=lambda t: -t["cy"])[:2]
        bottom.sort(key=lambda t: t["cx"])
        anchors["valid_until"] = (bottom[0]["cx"], bottom[0]["cy"])
        anchors["valid_from"] = (bottom[1]["cx"], bottom[1]["cy"])

    # Plate — top-most 3-7 digit non-year numeric token.
    plate_ts = []
    for t in toks:
        a = _convert_arabic_indic_digits_to_ascii(t["raw"])
        if _loose_date(a):
            continue
        for n in re.findall(r"\d+", a):
            if 3 <= len(n) <= 7 and not (1980 <= int(n) <= 2030):
                plate_ts.append(t)
                break
    if plate_ts:
        top = min(plate_ts, key=lambda t: t["cy"])
        anchors["plate_number"] = (top["cx"], top["cy"])

    return anchors


def _fit_template_affine(anchors: dict):
    """Fit template-space (cx,cy in [0,1]) → image-pixel affine from the anchor
    correspondences. >=3 anchors → full affine (RANSAC); 2 → similarity
    (rotation+scale+translation). Returns a 2x3 matrix M or None."""
    import cv2
    import numpy as np

    src, dst = [], []
    for field, (px, py) in anchors.items():
        slot = _MULKIYA_TEMPLATE.get(field)
        if slot:
            src.append([slot["cx"], slot["cy"]])
            dst.append([px, py])
    if len(src) < 2:
        return None
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if len(src) >= 3:
        M, _inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=12.0)
    else:
        M, _inl = cv2.estimateAffinePartial2D(src, dst)
    return M


# ── Tier-1 cell refinement: preprocess + upscale + digit re-OCR ─────────────
# Once the anchor frame is fit we know each numeric cell's pixel box. Small-font
# digits on WhatsApp-grade photos sit below the recogniser's resolution floor at
# full-page scale, so we crop the cell, boost contrast (CLAHE) + sharpen, upscale
# 4x, and re-read it digit-only. A confident in-range cell read overrides the
# full-page token bind — this attacks the residual digit MISREADS (1400->140),
# not just blanks.
_CELL_REFINE_ENABLED = os.getenv("UPSURE_CELL_REFINE", "1") not in ("0", "false", "False")
# Fields worth a cell re-read: small-font numerics prone to misreads. Years are
# excluded — a slightly-off year cell can grab an adjacent date.
_CELL_REFINE_RANGES = {
    "engine_cc": (600, 8999),
    "empty_weight_kg": (100, 6000),
    "max_load_kg": (100, 50000),
    "seats": (1, 9),
}


def _preprocess_cell(cell):
    """Grayscale + CLAHE contrast + unsharp mask, returned as 3-channel BGR.
    Lifts faded/compressed digits before the upscale + re-OCR."""
    import cv2
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _read_cell_number(image_bgr, cx, cy, hw, hh, engine, lo, hi):
    """Crop a numeric cell around (cx,cy), preprocess, 4x upscale, OCR, return
    the first in-range integer (digit-only via post-filter)."""
    import cv2
    H, W = image_bgr.shape[:2]
    x0 = max(0, int(cx - hw)); x1 = min(W, int(cx + hw))
    y0 = max(0, int(cy - hh)); y1 = min(H, int(cy + hh))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    cell = _preprocess_cell(image_bgr[y0:y1, x0:x1])
    big = cv2.resize(cell, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res = engine(big)
    if res is None or not res.txts:
        return None
    txt = _convert_arabic_indic_digits_to_ascii(" ".join(res.txts))
    for n in re.findall(r"\d{1,8}", txt):
        v = int(n)
        if lo <= v <= hi:
            return v
        split = _split_merged_weight(v) if lo == 100 else None  # weight cell merge
        if split and lo <= max(split) <= hi:
            return max(split)
    return None


def _extract_by_template(lines: list, image_bgr, engine=None) -> dict:
    """Bind values to fields by card-relative position (fixed Mulkiya layout).

    Framing: first try to fit an affine transform from the format-unique anchor
    fields (plate / VIN / the two dates) onto the template — this rides on the
    fields we trust and self-corrects scale/rotation/skew. If too few anchors are
    found, fall back to normalising against the value-token cluster bbox. Then
    greedy-match each field slot to the nearest type-compatible token.
    Returns only fields that matched a token.

    Requires a deskewed/oriented card — geometry is meaningless on a rotated
    frame, so the caller must crop/orient first.
    """
    import numpy as np

    if not lines:
        return {}

    toks: list[dict] = []
    for item in lines:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        box, rest = item
        if not (isinstance(rest, (list, tuple)) and len(rest) == 2):
            continue
        text, conf = rest
        if float(conf) < 0.25:
            continue
        ascii_t = _convert_arabic_indic_digits_to_ascii(str(text))
        # value-like: has a digit, or is the literal NIL engine-number
        if not (re.search(r"\d", ascii_t) or re.search(r"\bNIL\b", ascii_t, re.IGNORECASE)):
            continue
        try:
            b = np.asarray(box, dtype=float)
            xs, ys = b[:, 0], b[:, 1]
            toks.append({
                "raw": str(text),
                "cx": float(xs.mean()), "cy": float(ys.mean()),
                "x0": float(xs.min()), "x1": float(xs.max()),
                "y0": float(ys.min()), "y1": float(ys.max()),
            })
        except Exception:
            continue

    if len(toks) < 3:
        return {}

    cx0 = min(t["x0"] for t in toks)
    cy0 = min(t["y0"] for t in toks)
    cx1 = max(t["x1"] for t in toks)
    cy1 = max(t["y1"] for t in toks)
    cw = max(cx1 - cx0, 1.0)
    ch = max(cy1 - cy0, 1.0)
    diag = (cw ** 2 + ch ** 2) ** 0.5

    # Preferred frame: affine fit on the trusted anchors. Fall back to bbox.
    anchors = _find_template_anchors(toks)
    M = _fit_template_affine(anchors)

    def slot_xy(slot: dict) -> tuple[float, float]:
        if M is not None:
            import numpy as _np
            p = M @ _np.asarray([slot["cx"], slot["cy"], 1.0])
            return float(p[0]), float(p[1])
        return cx0 + slot["cx"] * cw, cy0 + slot["cy"] * ch

    # All compatible (distance, field, token, value) triples.
    pairs: list[tuple[float, str, int, object]] = []
    for field, slot in _MULKIYA_TEMPLATE.items():
        tx, ty = slot_xy(slot)
        for i, t in enumerate(toks):
            val = _template_field_value(field, t["raw"])
            if val is None:
                continue
            dist = (((t["cx"] - tx)) ** 2 + ((t["cy"] - ty)) ** 2) ** 0.5 / diag
            if dist <= _TEMPLATE_MATCH_GATE:
                pairs.append((dist, field, i, val))

    pairs.sort(key=lambda p: p[0])
    out: dict[str, object] = {}
    used_tokens: set[int] = set()
    bound_tok: dict[str, int] = {}  # field -> token index it bound to
    for dist, field, i, val in pairs:
        if field in out or i in used_tokens:
            continue
        out[field] = val
        bound_tok[field] = i
        used_tokens.add(i)

    # Merged-weight rescue. The two weight cells are adjacent, so the detector
    # often fuses them into one token (e.g. 5201060, 385260). A fused token fails
    # the per-field range check, so neither weight binds. Find a 5-8 digit token
    # in the weight band (~0.5 of cluster height) and split it. Observed on these
    # light vehicles: empty_weight > max_load, so larger half = empty.
    if not ("empty_weight_kg" in out and "max_load_kg" in out):
        ewx, ewy = slot_xy(_MULKIYA_TEMPLATE["empty_weight_kg"])
        mlx, mly = slot_xy(_MULKIYA_TEMPLATE["max_load_kg"])
        for i, t in enumerate(toks):
            if i in used_tokens:
                continue
            # near either weight cell (within ~25% of the frame diagonal)
            de = (((t["cx"] - ewx)) ** 2 + ((t["cy"] - ewy)) ** 2) ** 0.5 / diag
            dm = (((t["cx"] - mlx)) ** 2 + ((t["cy"] - mly)) ** 2) ** 0.5 / diag
            if min(de, dm) > 0.30:
                continue
            a = _convert_arabic_indic_digits_to_ascii(t["raw"])
            m = re.search(r"\b(\d{5,8})\b", a)
            if not m:
                continue
            split = _split_merged_weight(int(m.group(1)))
            if split:
                out["empty_weight_kg"], out["max_load_kg"] = max(split), min(split)
                used_tokens.add(i)
                break

    # Validity dates by x-position. Both dates sit on the bottom row; the card
    # prints "from" on the right, "to/until" on the left. Binding by x is more
    # robust than nearest-distance when a skewed crop drops stray dates higher up.
    date_toks: list[tuple[float, float, str]] = []
    for t in toks:
        d = _loose_date(t["raw"])
        if d:
            fy = (t["cy"] - cy0) / ch
            date_toks.append((fy, t["cx"], d))
    bottom = [d for d in date_toks if d[0] >= 0.78]
    use = bottom if len(bottom) >= 2 else date_toks
    if len(use) >= 2:
        use.sort(key=lambda d: d[1])  # left -> right
        out["valid_until"] = use[0][2]
        out["valid_from"] = use[-1][2]

    # Tier-1 cell refinement (targeted, not blanket — keeps the sync API fast).
    # A field that bound cleanly to a token is TRUSTED and skipped: re-reading it
    # would only risk a regression and cost an OCR. We re-read only the fields
    # that are blank or came from a merge-split guess (not in bound_tok) — crop
    # the affine-predicted cell at 4x with contrast/sharpen, digit-only. So a
    # healthy card pays ~zero extra OCR; only the weak cells are re-examined.
    # Needs an engine and a good affine frame (the cell position comes from it).
    if engine is not None and M is not None and _CELL_REFINE_ENABLED:
        heights = sorted(t["y1"] - t["y0"] for t in toks)
        cell_h = heights[len(heights) // 2] if heights else 20.0
        for field, (lo, hi) in _CELL_REFINE_RANGES.items():
            if field in bound_tok:
                continue  # cleanly bound → trust it, no re-OCR
            cx, cy = slot_xy(_MULKIYA_TEMPLATE[field])
            hw = (1.2 if field == "seats" else 2.4) * cell_h
            hh = 1.0 * cell_h
            v = _read_cell_number(image_bgr, cx, cy, hw, hh, engine, lo, hi)
            if v is not None:
                out[field] = v
    return out


def _template_yield_score(d: dict) -> float:
    """Score a template extraction for orientation selection. Rewards bound
    fields, with extra weight on the unmistakable ones (VIN pattern, dates) that
    only parse when the card is upright — so an upside-down garbage read scores
    near zero even if a few stray numbers happen to bind."""
    score = float(len(d))
    if "vin_or_chassis" in d:
        score += 3.0
    if "valid_from" in d or "valid_until" in d:
        score += 1.0
    if "plate_number" in d:
        score += 1.0
    return score


def _best_template_orientation(crop, engine, use_cls: bool):
    """The crop arrives already oriented by card_crop's cheap pixel header check
    (no OCR). OCR it once and extract. Only if that read is weak (no VIN and few
    fields — i.e. the header check may have mis-flipped) do we pay a single extra
    OCR on the 180° flip and keep whichever binds more. Max 2 OCR per card vs the
    old 4, and card_crop no longer OCRs for orientation either.

    Returns (oriented_crop, lines, template_dict)."""
    import cv2
    # Orientation probes are refine-free (engine omitted) — cheap binding only.
    # Cell refinement runs once on the chosen orientation.
    lines = _normalize_lines(_run_ocr(engine, crop, use_cls=use_cls))
    d = _extract_by_template(lines, crop)
    if "vin_or_chassis" in d and len(d) >= 6:
        return crop, lines, _extract_by_template(lines, crop, engine=engine)

    flip = cv2.rotate(crop, cv2.ROTATE_180)
    lines2 = _normalize_lines(_run_ocr(engine, flip, use_cls=use_cls))
    d2 = _extract_by_template(lines2, flip)
    if _template_yield_score(d2) > _template_yield_score(d):
        return flip, lines2, _extract_by_template(lines2, flip, engine=engine)
    return crop, lines, _extract_by_template(lines, crop, engine=engine)


def _coerce_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    if isinstance(value, str):
        m = re.search(r"\b\d{2,4}\b", value)
        if not m:
            return None
        try:
            y = int(m.group(0))
        except Exception:
            return None
        if y < 100:
            return 2000 + y
        return y if 1900 <= y <= 2100 else None
    return None


def _coerce_int(value: object, min_v: int | None = None, max_v: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            v = value
        elif isinstance(value, float):
            v = int(value)
        elif isinstance(value, str):
            m = re.search(r"\b\d+\b", value)
            if not m:
                return None
            v = int(m.group(0))
        else:
            return None
    except Exception:
        return None
    if min_v is not None and v < min_v:
        return None
    if max_v is not None and v > max_v:
        return None
    return v


def _preload_argos_translate() -> None:
    try:
        import importlib

        importlib.import_module("argostranslate.translate")
    except Exception as exc:
        raise SystemExit(
            "Translation requires Argos Translate. Install with: pip install argostranslate"
        ) from exc


def _translate_texts_argos(texts: list[str], from_code: str, to_code: str) -> list[str]:
    try:
        import importlib

        argos_translate = importlib.import_module("argostranslate.translate")
    except Exception as exc:
        raise SystemExit(
            "Translation requires Argos Translate. Install with: pip install argostranslate"
        ) from exc

    languages = argos_translate.get_installed_languages()
    from_lang = next((l for l in languages if l.code == from_code), None)
    to_lang = next((l for l in languages if l.code == to_code), None)
    if from_lang is None or to_lang is None:
        raise SystemExit(
            f"Argos language package missing for {from_code}->{to_code}. "
            "Install the appropriate Argos model (ar_en)."
        )

    translation = from_lang.get_translation(to_lang)
    if translation is None:
        raise SystemExit(
            f"Argos translation model missing for {from_code}->{to_code}. "
            "Install the appropriate Argos model (ar_en)."
        )

    cache: dict[str, str] = {}
    out: list[str] = []
    for t in texts:
        if t in cache:
            out.append(cache[t])
            continue
        tr = translation.translate(t)
        cache[t] = tr
        out.append(tr)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple PaddleOCR image/PDF test")
    parser.add_argument(
        "input",
        nargs="?",
        default="test_image.jpeg",
        help="Path to an image or PDF (default: test_image.jpeg)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to save output: for images a file path (default: <image>_ocr.png); "
            "for PDFs a directory (default: <pdf>_ocr_pages/)"
        ),
    )
    parser.add_argument(
        "--write_text",
        action="store_true",
        help="Write extracted data files in JSON format (PDF: results.json, fulltext.json; image: <image>_ocr.txt)",
    )
    parser.add_argument(
        "--write_translation_text",
        action="store_true",
        help=(
            "Write translation-friendly JSON data (normalized, whitespace-cleaned; Arabic uses RTL sorting). "
            "Image: <image>_ocr_translation.txt; PDF: fulltext_translation.json"
        ),
    )
    parser.add_argument(
        "--translate_to_en",
        action="store_true",
        help="Translate extracted Arabic text to English (requires argostranslate + ar_en model)",
    )
    parser.add_argument(
        "--fix_arabic_reverse",
        action="store_true",
        help=(
            "Apply a legacy heuristic that reverses Arabic character runs. "
            "Leave off for PaddleOCR Arabic models."
        ),
    )
    parser.add_argument(
        "--no_fix_arabic_reverse",
        action="store_true",
        help=(
            "Legacy no-op kept for compatibility; Arabic reverse fixing is off by default."
        ),
    )
    parser.add_argument(
        "--arabic_reshaper",
        action="store_true",
        help=(
            "Apply arabic-reshaper to stored OCR text. Useful for display experiments, "
            "but normally off so JSON contains logical Unicode text."
        ),
    )
    parser.add_argument(
        "--no_arabic_reshaper",
        action="store_true",
        help=(
            "Legacy no-op kept for compatibility; Arabic reshaping is off by default."
        ),
    )
    parser.add_argument(
        "--extract_mulkya",
        action="store_true",
        help="Image only: extract structured Mulkiya vehicle details into <image>_mulkya.json",
    )
    parser.add_argument(
        "--write_benchmark",
        action="store_true",
        help="Write benchmark CSV (timings/conf stats) into output folder",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="PDF only: do not write annotated page PNGs (faster full-text runs)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a window to preview the annotated image",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU (requires a CUDA-capable paddlepaddle build)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id to use when --use_gpu is set (default: 0)",
    )
    parser.add_argument(
        "--enable_mkldnn",
        action="store_true",
        help="Enable MKL-DNN acceleration on CPU (often faster on Intel CPUs)",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=10,
        help="CPU threads for math library (default: 10)",
    )
    parser.add_argument(
        "--rec_batch_num",
        type=int,
        default=6,
        help="Recognizer batch size (default: 6)",
    )
    parser.add_argument(
        "--cls_batch_num",
        type=int,
        default=6,
        help="Angle-classifier batch size (default: 6)",
    )
    parser.add_argument(
        "--det_limit_side_len",
        type=float,
        default=960,
        help="Detector max side length for resizing (default: 960; lower is faster)",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.0,
        help="Only draw boxes with confidence >= this value (default: 0.0)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language model (e.g. en, ch, french, german, korean, japan)",
    )
    parser.add_argument(
        "--use_angle_cls",
        action="store_true",
        help="Enable angle classifier (better for rotated text)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF render DPI (default: 200)",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=0,
        help="Max PDF pages to process (0 = all pages)",
    )
    parser.add_argument(
        "--prefer_pdf_text",
        action="store_true",
        help="PDF only: if a text layer exists on a page, use it instead of OCR (much faster)",
    )
    parser.add_argument(
        "--make_crop",
        action="store_true",
        help=(
            "Image only: detect/deskew/orient the Mulkiya card and write "
            "<stem>_cropped.jpg, then exit (no extraction). Used as a quality "
            "fallback: re-run extraction on the crop when the original is unusable."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    is_arabic = args.lang.lower() in {"ar", "arabic"}
    fix_arabic_reverse = is_arabic and args.fix_arabic_reverse and (not args.no_fix_arabic_reverse)
    reshape_arabic = is_arabic and args.arabic_reshaper and (not args.no_arabic_reshaper)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if args.make_crop:
        if input_path.suffix.lower() == ".pdf":
            raise SystemExit("--make_crop is image-only.")
        import cv2
        import card_crop
        img = cv2.imread(str(input_path))
        if img is None:
            raise SystemExit(f"Could not read image: {input_path}")
        crop, reason = card_crop.choose_mulkiya_crop(img)
        out_crop = input_path.with_name(f"{input_path.stem}_cropped.jpg")
        cv2.imwrite(str(out_crop), crop)
        print(f"Crop written to: {out_crop}  [{reason}]")
        return

    if args.translate_to_en:
        _preload_argos_translate()

    import cv2
    import numpy as np

    engine = _get_engine()

    # ------------------------------------------------------------------
    # postprocess() — call this on every raw OCR text string right after
    # PaddleOCR returns it.  Running once here means all downstream
    # consumers (file writers and rule extractor) share the same
    # clean, properly shaped Arabic text without any extra conversion steps.
    # ------------------------------------------------------------------
    def postprocess(text: str) -> str:
        return _postprocess_ocr_line(
            text,
            lang=args.lang,
            fix_arabic_reverse=fix_arabic_reverse,
            reshape_arabic=reshape_arabic,
        )

    # PaddleOCR returns a list per image. Each element is:
    #   [ [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ], (text, confidence) ]
    if input_path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception as exc:
            raise SystemExit(
                "PyMuPDF (fitz) is required for PDF input. It should already be installed via PaddleOCR."
            ) from exc

        # For PDFs we will write a single JSON next to the input PDF
        # and only create a page folder when images are requested.
        write_text = True
        write_benchmark = args.write_benchmark

        # JSON output file (single file per document)
        fulltext_json = input_path.with_name(f"{input_path.stem}_ocr.json")
        fulltext_translation_json = input_path.with_name(f"{input_path.stem}_ocr_translation.json")
        fulltext_translation_en_json = input_path.with_name(f"{input_path.stem}_ocr_translation_en.json")

        # Initialize data structures for JSON output
        fulltext_data = {"pages": []}
        fulltext_translation_data = {"pages": []}
        fulltext_translation_en_data = {"pages": []}

        # Decide where to store page images/benchmarks (only if needed)
        out_dir = None
        if not args.no_images:
            out_dir = Path(args.out) if args.out else input_path.with_name(f"{input_path.stem}_ocr_pages")
            out_dir.mkdir(parents=True, exist_ok=True)

        benchmark_csv = (out_dir / "benchmark.csv") if out_dir else input_path.with_name(f"{input_path.stem}_ocr_benchmark.csv")
        bench_f = benchmark_csv.open("w", encoding="utf-8", newline="") if write_benchmark else None
        bench_writer = None
        if bench_f is not None:
            bench_writer = csv.DictWriter(
                bench_f,
                fieldnames=[
                    "page",
                    "seconds",
                    "lines",
                    "conf_mean",
                    "conf_min",
                    "conf_max",
                    "error",
                ],
            )
            bench_writer.writeheader()

        doc = fitz.open(str(input_path))
        total_pages = doc.page_count
        max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else total_pages
        pages_to_process = min(total_pages, max_pages)

        try:
            for page_index in range(pages_to_process):
                t0 = time.perf_counter()
                page_num = page_index + 1
                error: str | None = None

                try:
                    page = doc.load_page(page_index)
                    zoom = float(args.dpi) / 72.0
                    if args.prefer_pdf_text:
                        pdf_text = (page.get_text("text") or "").strip()
                    else:
                        pdf_text = ""

                    if pdf_text:
                        # Fast path: digital PDF with an embedded text layer.
                        # Post-process each line exactly as we do with OCR output.
                        extracted_lines = [
                            postprocess(ln.rstrip())
                            for ln in pdf_text.splitlines()
                            if ln.strip()
                        ]
                        if write_text:
                            fulltext_data["pages"].append(
                                {
                                    "page": page_num,
                                    "lines": [
                                        {"text": t, "confidence": 1.0}
                                        for t in extracted_lines
                                    ],
                                }
                            )

                            translation_lines = []
                            for t in extracted_lines:
                                nt = _normalize_for_translation(t)
                                if nt:
                                    translation_lines.append({"text": nt, "confidence": 1.0})
                            fulltext_translation_data["pages"].append(
                                {"page": page_num, "lines": translation_lines}
                            )

                            if args.translate_to_en:
                                normalized = [
                                    _normalize_for_translation(t) for t in extracted_lines
                                ]
                                normalized = [t for t in normalized if t]
                                translated = _translate_texts_argos(normalized, "ar", "en") if normalized else []
                                fulltext_translation_en_data["pages"].append(
                                    {
                                        "page": page_num,
                                        "lines": [{"text": _normalize_for_translation(t)} for t in translated if _normalize_for_translation(t)],
                                    }
                                )

                        lines: list = []
                        conf_mean = 1.0
                        conf_min = 1.0
                        conf_max = 1.0
                        error = "pdf_text_layer"

                        if not args.no_images:
                            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                            img = np.frombuffer(pix.samples, dtype=np.uint8)
                            img = img.reshape((pix.height, pix.width, pix.n))
                            if pix.n == 4:
                                img = img[:, :, :3]
                            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            if out_dir:
                                page_out = out_dir / f"page_{page_num:03d}.png"
                            else:
                                page_out = input_path.with_name(f"{input_path.stem}_page_{page_num:03d}.png")
                            if not cv2.imwrite(str(page_out), image_bgr):
                                raise RuntimeError(f"Failed to write output image: {page_out}")

                        print(f"Page {page_num}/{pages_to_process}: PDF text layer ({len(extracted_lines)} lines)")

                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

                        img = np.frombuffer(pix.samples, dtype=np.uint8)
                        img = img.reshape((pix.height, pix.width, pix.n))
                        if pix.n == 4:
                            img = img[:, :, :3]

                        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        result = _run_ocr(engine, image_bgr, use_cls=args.use_angle_cls)
                        raw_lines = _normalize_lines(result)

                        # ── POST-PROCESS every OCR line right after detection ──
                        # arabic-reshaper runs here — once — so the reshaped text
                        # flows into file writers and rule extractor.
                        lines = [
                            (box, (postprocess(text), conf))
                            for box, (text, conf) in raw_lines
                        ]
                        # ─────────────────────────────────────────────────────

                        confs = [float(conf) for _box, (_text, conf) in lines]
                        conf_mean = (sum(confs) / len(confs)) if confs else 0.0
                        conf_min = min(confs) if confs else 0.0
                        conf_max = max(confs) if confs else 0.0

                        if write_text:
                            # Accumulate fulltext data
                            fulltext_lines = []
                            for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
                                if float(confidence) >= args.min_conf:
                                    fulltext_lines.append({
                                        "text": text,
                                        "confidence": float(confidence)
                                    })
                            fulltext_data["pages"].append({
                                "page": page_num,
                                "lines": fulltext_lines
                            })

                            # Accumulate translation data
                            translation_lines = []
                            for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
                                if float(confidence) >= args.min_conf:
                                    nt = _normalize_for_translation(text)
                                    if nt:
                                        translation_lines.append({
                                            "text": nt,
                                            "confidence": float(confidence)
                                        })
                            fulltext_translation_data["pages"].append({
                                "page": page_num,
                                "lines": translation_lines
                            })

                            # Accumulate English translation data
                            if args.translate_to_en:
                                kept: list[str] = []
                                for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
                                    if float(confidence) >= args.min_conf:
                                        nt = _normalize_for_translation(text)
                                        if nt:
                                            kept.append(nt)
                                translated = _translate_texts_argos(kept, "ar", "en") if kept else []
                                en_lines = []
                                for t in translated:
                                    t = _normalize_for_translation(t)
                                    if t:
                                        en_lines.append({"text": t})
                                fulltext_translation_en_data["pages"].append({
                                    "page": page_num,
                                    "lines": en_lines
                                })

                        annotated = None
                        if (not args.no_images) or args.show:
                            annotated = _draw_boxes(image_bgr, lines, args.min_conf)

                        if not args.no_images:
                            page_out = out_dir / f"page_{page_num:03d}.png"
                            if not cv2.imwrite(
                                str(page_out), annotated if annotated is not None else image_bgr
                            ):
                                raise RuntimeError(f"Failed to write output image: {page_out}")

                        if args.show and page_index == 0 and annotated is not None:
                            cv2.imshow("PaddleOCR - annotated (page 1)", annotated)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        print(
                            f"Page {page_num}/{pages_to_process}: {len(lines)} line(s) "
                            f"conf_mean={conf_mean:.3f}"
                        )

                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    conf_mean = 0.0
                    conf_min = 0.0
                    conf_max = 0.0
                    lines = []
                    print(f"Page {page_num}/{pages_to_process}: ERROR {error}")

                seconds = time.perf_counter() - t0
                if bench_writer is not None:
                    bench_writer.writerow(
                        {
                            "page": page_num,
                            "seconds": f"{seconds:.4f}",
                            "lines": len(lines),
                            "conf_mean": f"{conf_mean:.4f}",
                            "conf_min": f"{conf_min:.4f}",
                            "conf_max": f"{conf_max:.4f}",
                            "error": error or "",
                        }
                    )
                    bench_f.flush()

        finally:
            # Write JSON files: single fulltext JSON next to PDF
            if write_text:
                fulltext_json.write_text(json.dumps(fulltext_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.write_translation_text:
                fulltext_translation_json.write_text(json.dumps(fulltext_translation_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.translate_to_en:
                fulltext_translation_en_json.write_text(json.dumps(fulltext_translation_en_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if bench_f is not None:
                bench_f.close()

        if out_dir:
            print(f"PDF analysis saved to: {out_dir}")
        else:
            print(f"PDF analysis JSON saved to: {fulltext_json}")
        if write_text:
            print(f"Full text JSON: {fulltext_json}")
        if args.write_translation_text:
            print(f"Translation JSON: {fulltext_translation_json}")
        if args.translate_to_en:
            print(f"Translation (EN) JSON: {fulltext_translation_en_json}")
        if write_benchmark:
            print(f"Benchmark: {benchmark_csv}")
        return

    # ── Single image path ────────────────────────────────────────────────────
    image_path = input_path
    out_path = Path(args.out) if args.out else image_path.with_name(f"{image_path.stem}_ocr.png")
    write_text = args.write_text
    write_benchmark = args.write_benchmark

    t0 = time.perf_counter()
    result = _run_ocr(engine, str(image_path), use_cls=args.use_angle_cls)
    raw_lines = _normalize_lines(result)
    if not raw_lines:
        print("No text detected.")
        return

    # ── POST-PROCESS every OCR line right after detection ───────────────────
    # arabic-reshaper runs here — once — so the same reshaped text is used in
    # the annotated-image labels, text files, and rule extractor.
    lines = [
        (box, (postprocess(text), conf))
        for box, (text, conf) in raw_lines
    ]
    # ────────────────────────────────────────────────────────────────────────

    print(f"Detected {len(lines)} text line(s):")
    for i, line in enumerate(lines, start=1):
        _, (text, confidence) = line
        print(f"{i:02d}. {text}  (conf={float(confidence):.3f})")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    if write_text:
        json_out = image_path.with_name(f"{image_path.stem}_ocr.json")
        page = {"image": str(image_path), "lines": []}
        for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
            if float(confidence) >= args.min_conf:
                page["lines"].append({"text": text, "confidence": float(confidence)})
        json_out.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Full text JSON saved to: {json_out}")

    if args.write_translation_text:
        json_out = image_path.with_name(f"{image_path.stem}_ocr_translation.json")
        page = {"image": str(image_path), "lines": []}
        kept: list[str] = []
        for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
            if float(confidence) >= args.min_conf:
                nt = _normalize_for_translation(text)
                if nt:
                    kept.append(nt)
                    page["lines"].append({"text": nt, "confidence": float(confidence)})
        json_out.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Translation JSON saved to: {json_out}")

    if args.translate_to_en:
        kept = []
        for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
            if float(confidence) >= args.min_conf:
                nt = _normalize_for_translation(text)
                if nt:
                    kept.append(nt)
        translated = _translate_texts_argos(kept, "ar", "en") if kept else []
        json_out = image_path.with_name(f"{image_path.stem}_ocr_en.json")
        page = {"image": str(image_path), "lines": []}
        for t in translated:
            t = _normalize_for_translation(t)
            if t:
                page["lines"].append({"text": t})
        json_out.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"English translation saved to: {json_out}")

    data: dict[str, object] | None = None
    if args.extract_mulkya:
        ordered: list[str] = []
        for _box, (text, confidence) in _sort_lines(lines, rtl=is_arabic):
            if float(confidence) >= args.min_conf:
                # text is already post-processed (reshaped) — just normalise for NLP.
                nt = _normalize_for_translation(text)
                if nt:
                    ordered.append(nt)

        data = _extract_mulkya_rulebased(ordered)

        critical_fields = (
            "plate_number",
            "vin_or_chassis",
            "engine_cc",
            "empty_weight_kg",
            "max_load_kg",
            "seats",
            "issue_date",
            "year",
        )
        aux_ocr_lang = None
        if is_arabic and any(data.get(field) in (None, "") for field in critical_fields):
            try:
                aux_result = _run_ocr(engine, str(image_path), use_cls=args.use_angle_cls)
                aux_raw_lines = _normalize_lines(aux_result)
                aux_lines = [
                    (
                        box,
                        (
                            _postprocess_ocr_line(
                                text,
                                lang="en",
                                fix_arabic_reverse=False,
                                reshape_arabic=False,
                            ),
                            conf,
                        ),
                    )
                    for box, (text, conf) in aux_raw_lines
                ]
                aux_ordered: list[str] = []
                aux_min_conf = max(args.min_conf, 0.70)
                for _box, (text, confidence) in _sort_lines(aux_lines, rtl=False):
                    if float(confidence) >= aux_min_conf:
                        nt = _normalize_for_translation(text)
                        if nt:
                            aux_ordered.append(nt)
                if aux_ordered:
                    data = _extract_mulkya_rulebased(ordered + aux_ordered)
                    data["notes"] = "Heuristic extraction with auxiliary English OCR fallback."
                    aux_ocr_lang = "en"
            except Exception as exc:
                data.setdefault("validation_notes", [])
                if isinstance(data["validation_notes"], list):
                    data["validation_notes"].append(f"auxiliary_ocr_failed: {exc}")

        # Template field names → existing JSON schema names.
        _TEMPLATE_TO_SCHEMA = {
            "plate_number": "plate_number",
            "model_year": "model_year",
            "manufacturing_year": "year",
            "engine_cc": "engine_cc",
            "empty_weight_kg": "empty_weight_kg",
            "max_load_kg": "max_load_kg",
            "seats": "seats",
            "no_of_axles": "no_of_axles",
            "vin_or_chassis": "vin_or_chassis",
            "engine_number": "engine_number",
            "valid_from": "issue_date",
            "valid_until": "expiry_date",
        }
        # Position-AMBIGUOUS fields: bare integers with overlapping ranges that
        # can ONLY be told apart by card position. When the anchor frame is
        # confident we trust it exclusively for these — a blank beats a wrong
        # guess from the flat/range extractors.
        _AMBIGUOUS_FIELDS = ("engine_cc", "empty_weight_kg", "max_load_kg", "seats", "year", "model_year")
        template_set: set[str] = set()
        template_confident = False

        # Range-based numeric extractor runs FIRST — it reuses the full-image OCR
        # boxes (no extra OCR pass, ~free) and handles the detector cell-merge for
        # weights (5201060 → 520 + 1060). On a clean upright card it fills every
        # numeric, which lets us skip the expensive positional-template pass below.
        if _RANGE_EXTRACTOR_ENABLED:
            try:
                range_data = _extract_by_range_with_boxes(lines, image_bgr, engine)
                range_overrides: dict = {}
                _range_fields = (
                    'plate_number', 'vin_or_chassis', 'year', 'model_year',
                    'engine_cc', 'empty_weight_kg', 'max_load_kg', 'seats',
                    'issue_date', 'expiry_date',
                )
                for field in _range_fields:
                    if range_data.get(field) is not None:
                        if range_data[field] != data.get(field):
                            range_overrides[field] = range_data[field]
                        data[field] = range_data[field]
                # If range extractor found no seats, invalidate flat extractor's
                # out-of-range value (flat extractor has no geometry and often
                # picks arbitrary numbers in 1-80).
                if range_data.get('seats') is None and isinstance(data.get('seats'), int) and not (1 <= data['seats'] <= 9):
                    data['seats'] = None
                if range_overrides:
                    data.setdefault("validation_notes", [])
                    if isinstance(data["validation_notes"], list):
                        changed = ", ".join(f"{k}={v}" for k, v in range_overrides.items())
                        data["validation_notes"].append(f"range_extractor_override: {changed}")
            except Exception as exc:
                data.setdefault("validation_notes", [])
                if isinstance(data["validation_notes"], list):
                    data["validation_notes"].append(f"range_extractor_failed: {exc}")

        # Positional-template extractor (AUTHORITATIVE for numerics/dates) — it
        # deskews/orients the card and re-OCRs a crop plus each field cell, which
        # costs ~6s. The cheap full-image + range read above is already correct on
        # clean upright cards, so only pay the template when that read is missing
        # or has an out-of-range field (rotated / multi-doc / noisy frames). When
        # it runs it OVERRIDES the range values (template > range > flat).
        if _TEMPLATE_EXTRACTOR_ENABLED and not is_arabic and not _numeric_fields_complete(data):
            try:
                import card_crop
                crop, crop_reason = card_crop.choose_mulkiya_crop(image_bgr)
                # Resolve residual 180°/90° ambiguity by extraction yield.
                _crop_oriented, crop_lines, tmpl = _best_template_orientation(
                    crop, engine, args.use_angle_cls
                )
                # Confident = the trusted anchors (plate+VIN) were found and the
                # affine frame bound a healthy number of fields.
                template_confident = (
                    "vin_or_chassis" in tmpl and "plate_number" in tmpl and len(tmpl) >= 6
                )
                tmpl_overrides: dict = {}
                for tfield, val in tmpl.items():
                    sfield = _TEMPLATE_TO_SCHEMA.get(tfield, tfield)
                    if val != data.get(sfield):
                        tmpl_overrides[sfield] = val
                    data[sfield] = val
                    template_set.add(sfield)
                # Confident frame → drop the range/flat guesses for any ambiguous
                # field the template left blank (prefer blank to wrong).
                if template_confident:
                    for f in _AMBIGUOUS_FIELDS:
                        if f not in template_set:
                            data[f] = None
                if tmpl_overrides:
                    data.setdefault("validation_notes", [])
                    if isinstance(data["validation_notes"], list):
                        changed = ", ".join(f"{k}={v}" for k, v in tmpl_overrides.items())
                        data["validation_notes"].append(f"template_override[{crop_reason}]: {changed}")
            except Exception as exc:
                data.setdefault("validation_notes", [])
                if isinstance(data["validation_notes"], list):
                    data["validation_notes"].append(f"template_extractor_failed: {exc}")

        # Validate extracted data and add any inconsistency notes.
        _validate_and_note_data(data)
        data["quality"] = _assess_extraction_quality(data)

        data.setdefault("source", {})
        if isinstance(data["source"], dict):
            data["source"].update(
                {
                    "input": str(image_path),
                    "lang": args.lang,
                    "auxiliary_ocr_lang": aux_ocr_lang,
                    "fix_arabic_reverse": fix_arabic_reverse,
                    "reshape_arabic": reshape_arabic,
                    "template_extractor": _TEMPLATE_EXTRACTOR_ENABLED,
                    "range_extractor": _RANGE_EXTRACTOR_ENABLED,
                }
            )
        out_json = image_path.with_name(f"{image_path.stem}_mulkya.json")
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Mulkiya JSON saved to: {out_json}")
        if data.get("validation_notes"):
            print(f"  ⚠ Validation notes: {'; '.join(data['validation_notes'][:3])}")

    if (not args.no_images) or args.show:
        annotated = (
            _draw_field_boxes(image_bgr, lines, data, args.min_conf)
            if data is not None
            else _draw_boxes(image_bgr, lines, args.min_conf)
        )
    else:
        annotated = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), annotated if annotated is not None else image_bgr):
        raise SystemExit(f"Failed to write output image: {out_path}")

    print(f"Annotated image saved to: {out_path}")

    if write_benchmark:
        confs = [float(conf) for _box, (_text, conf) in lines]
        conf_mean = (sum(confs) / len(confs)) if confs else 0.0
        bench_out = image_path.with_name(f"{image_path.stem}_ocr_benchmark.csv")
        with bench_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["seconds", "lines", "conf_mean", "conf_min", "conf_max"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "seconds": f"{(time.perf_counter() - t0):.4f}",
                    "lines": len(lines),
                    "conf_mean": f"{conf_mean:.4f}",
                    "conf_min": f"{(min(confs) if confs else 0.0):.4f}",
                    "conf_max": f"{(max(confs) if confs else 0.0):.4f}",
                }
            )
        print(f"Benchmark saved to: {bench_out}")

    if args.show:
        cv2.imshow("PaddleOCR - annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
