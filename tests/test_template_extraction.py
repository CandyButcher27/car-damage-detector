"""Unit tests for the positional-template Mulkiya extractor.

Synthetic OCR tokens are placed at the template's card-relative positions on a
notional 1000x500 card, so the cluster-normalisation + greedy geometric binding
can be tested without running RapidOCR or card_crop.
"""
import ocr_simple_test as o

W, H = 1000, 500


def _line(text, fx, fy, w=60, h=24):
    """An OCR line `[box, (text, conf)]` centred at (fx,fy) fractions of WxH."""
    cx, cy = fx * W, fy * H
    box = [
        [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2],
    ]
    return [box, (text, 0.95)]


def _full_card_lines():
    t = o._MULKIYA_TEMPLATE
    vals = {
        "plate_number": "63888",
        "model_year": "2013",
        "engine_cc": "3500",
        "empty_weight_kg": "1550",
        "max_load_kg": "350",
        "manufacturing_year": "2012",
        "seats": "4",
        "no_of_axles": "2",
        "vin_or_chassis": "4T1BK1FK9DU531385",
        "engine_number": "NIL",
        "valid_from": "2024/03/10",
        "valid_until": "2025/12/06",
    }
    return [_line(vals[f], t[f]["cx"], t[f]["cy"]) for f in vals]


def test_template_binds_all_fields():
    d = o._extract_by_template(_full_card_lines(), image_bgr=None)
    assert d["plate_number"] == "63888"
    assert d["engine_cc"] == 3500
    assert d["empty_weight_kg"] == 1550
    assert d["max_load_kg"] == 350
    assert d["seats"] == 4
    assert d["no_of_axles"] == 2
    assert d["vin_or_chassis"] == "4T1BK1FK9DU531385"
    assert d["engine_number"] == "NIL"


def test_template_distinguishes_two_years_by_position():
    d = o._extract_by_template(_full_card_lines(), image_bgr=None)
    # model_year sits upper-left, manufacturing_year mid-right — bound by geometry
    assert d["model_year"] == 2013
    assert d["manufacturing_year"] == 2012


def test_template_dates_by_x_position():
    # valid_from prints on the right, valid_until on the left of the bottom row.
    d = o._extract_by_template(_full_card_lines(), image_bgr=None)
    assert d["valid_from"] == "2024/03/10"
    assert d["valid_until"] == "2025/12/06"


def test_template_recovers_dropped_date_separator():
    # OCR drops the first '-': '202503-10' must still parse to 2025/03/10.
    assert o._loose_date("202503-10") == "2025/03/10"
    assert o._loose_date("2025-12-06") == "2025/12/06"
    # a bare 8-digit number with no separator is NOT a date (could be a plate/VIN)
    assert o._loose_date("20250310") is None


def test_template_splits_merged_weight_token():
    # The two weight cells fuse into one token (e.g. 5201060) → split across both.
    t = o._MULKIYA_TEMPLATE
    lines = _full_card_lines()
    # replace the two separate weight tokens with one fused token in the weight band
    lines = [
        ln for ln in lines
        if ln[1][0] not in ("1550", "350")
    ]
    lines.append(_line("5201060", t["empty_weight_kg"]["cx"], t["empty_weight_kg"]["cy"]))
    d = o._extract_by_template(lines, image_bgr=None)
    assert d["empty_weight_kg"] == 1060  # larger half
    assert d["max_load_kg"] == 520        # smaller half


def test_template_rejects_plate_sized_number_as_weight():
    # A 5-digit plate-like number must not bind as empty_weight (range ≤ 6000).
    assert o._template_field_value("empty_weight_kg", "37319") is None
    assert o._template_field_value("empty_weight_kg", "1550") == 1550


def test_template_empty_on_no_tokens():
    assert o._extract_by_template([], image_bgr=None) == {}
