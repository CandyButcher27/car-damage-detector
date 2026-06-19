# NOTE ON AUTHORSHIP
# This module started from a teammate's multi-filetype/Arabic OCR helper on the
# UpsureAI data-ingestion service. The current file is my substantial rework:
# the RapidOCR (PP-OCR) dual-pass Arabic+English path, the reversed-Arabic-run
# repair, the digit/field normalization helpers, and the spatial extraction used
# by prototype_spatial.py. Kept here as reference for my mulkiya OCR work.

from __future__ import annotations

import argparse
import csv
import json
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


def _create_arabic_ocr_engine():
    from rapidocr import RapidOCR, LangRec
    return RapidOCR(params={"Rec.lang_type": LangRec.ARABIC})


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

    def extract_vin(text: str) -> str | None:
        """Extract VIN/chassis from text, prioritizing alphanumeric sequences of 11-20 chars.

        Robust to splitting across lines, spaces, and dashes. Prefers 17-char VINs.
        Filters out common OCR artifacts and header/footer terms.
        """
        # Single words or common phrase artifacts to reject
        single_artifacts = {
            "VEHICLE", "MOTOR", "ENGINE", "LICENSE", "LCENSE", "TIRAFIIC",
            "SULTANATE", "OMAN", "POLICE", "ROYA", "ROYAL", "KINGDOM",
            "GENERALOFTRAFFIC", "GENERAL", "TRAFFIC",
        }
        # Compound artifacts (concatenations of single artifacts)
        compound_artifacts = {
            "VEHICLEMOTOR", "MOTORVEHICLE", "ENGINEMOTOR", "MOTORENGINE",
            "POLICEOMATIC", "SULTANATEOMAN", "DIRGENERALOFTRAFFIC",
        }
        all_artifacts = single_artifacts | compound_artifacts

        candidates: list[tuple[str, int]] = []

        # Pattern 1: chunked (space/dash separated alphanumeric runs totaling 11-20 chars)
        for m in re.finditer(r"(?:[A-Za-z0-9]{2,}[\s\-]*)+", text):
            raw = m.group(0)
            cand = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
            # Reject if it's a known artifact, starts with a single artifact, or is too short.
            if 11 <= len(cand) <= 20:
                # Check if it starts/ends with known single artifacts
                is_artifact = False
                for artifact in single_artifacts:
                    if cand == artifact or cand.startswith(artifact) or cand.endswith(artifact):
                        is_artifact = True
                        break
                if cand in all_artifacts:
                    is_artifact = True
                if not is_artifact:
                    candidates.append((cand, len(cand)))

        # Pattern 2: simple contiguous alphanumeric (11-20 chars).
        for m in re.finditer(r"[A-Za-z0-9]{11,20}", text):
            cand = m.group(0).upper()
            if cand not in all_artifacts and not any(
                cand.startswith(a) or cand.endswith(a) for a in single_artifacts
            ):
                candidates.append((cand, len(cand)))

        if not candidates:
            return None

        # De-duplicate and rank by length (prefer 17-char, then longest).
        uniq_dict = {c: l for c, l in candidates}
        exact_17 = [
            c
            for c, l in uniq_dict.items()
            if l == 17 and re.search(r"[A-Z]", c) and re.search(r"\d", c)
        ]
        if exact_17:
            return exact_17[0]
        
        # Prefer candidates starting with letters only after exact VIN-length
        # candidates have been considered. Some real VINs start with digits.
        letter_start = {c: l for c, l in uniq_dict.items() if c and c[0].isalpha()}
        pool = letter_start if letter_start else uniq_dict
        
        # Fall back to longest from available pool
        return max(pool.items(), key=lambda x: x[1])[0]

    vin_or_chassis = None
    label_keys = ["الشاص", "الشاصي", "شاصي", "chassis", "vin", "رقم القاعدة", "القاعدة", "الاعدة"]
    for i, ln in enumerate(lines):
        if any(k in ln.lower() for k in label_keys if isinstance(k, str)):
            window = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 8)])
            cand = extract_vin(window)
            if cand:
                vin_or_chassis = cand
                break
    if not vin_or_chassis:
        vin_or_chassis = extract_vin(joined_ascii)

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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    engine = _create_ocr_engine()

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

        # Fields the Arabic pass extracts better because their VALUE sits next
        # to an Arabic LABEL (the English pass sees only bare numbers/text with
        # no label to bind to).
        # Fields the Arabic pass extracts better because their VALUE sits next
        # to an Arabic LABEL (the English pass sees only bare text with no label
        # to bind to). Numeric fields are intentionally NOT merged from Arabic:
        # the Arabic recogniser fragments digits worse than the English one, so
        # an Arabic-bound number is usually less accurate than the English read.
        arabic_text_fields = ("vehicle_type", "make", "model", "color", "country_of_origin")
        if not is_arabic and any(data.get(field) in (None, "") for field in arabic_text_fields):
            try:
                ara_engine = _create_arabic_ocr_engine()
                ara_result = _run_ocr(ara_engine, str(image_path), use_cls=args.use_angle_cls)
                ara_raw_lines = _normalize_lines(ara_result)
                ara_lines = [
                    (box, (_fix_reversed_arabic_runs(text), conf))
                    for box, (text, conf) in ara_raw_lines
                ]
                ara_ordered: list[str] = []
                for _box, (text, confidence) in _sort_lines(ara_lines, rtl=True):
                    if float(confidence) >= args.min_conf:
                        nt = _normalize_for_translation(text)
                        if nt:
                            ara_ordered.append(nt)
                if ara_ordered:
                    ara_data = _extract_mulkya_rulebased(ara_ordered)
                    for field in arabic_text_fields:
                        if data.get(field) in (None, "") and ara_data.get(field) not in (None, ""):
                            data[field] = ara_data[field]
                    if any(data.get(f) not in (None, "") for f in arabic_text_fields):
                        aux_ocr_lang = (aux_ocr_lang + "+ar") if aux_ocr_lang else "ar"
            except Exception as exc:
                data.setdefault("validation_notes", [])
                if isinstance(data["validation_notes"], list):
                    data["validation_notes"].append(f"auxiliary_arabic_ocr_failed: {exc}")

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
