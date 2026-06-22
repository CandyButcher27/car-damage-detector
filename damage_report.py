from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_TYPE_COLOR_PIL: dict[str, tuple[int, int, int]] = {
    "deformation":    (255, 140,   0),
    "scratches":      (255, 215,   0),
    "car-part-crack": (220,  20,  60),
    "glass-crack":    ( 30, 144, 255),
    "lamp-crack":     (148,   0, 211),
    "flat-tire":      (255,  20, 147),
}

_TYPE_COLOR_RL: dict[str, colors.Color] = {
    "deformation":    colors.HexColor("#FF8C00"),
    "scratches":      colors.HexColor("#FFD700"),
    "car-part-crack": colors.HexColor("#DC143C"),
    "glass-crack":    colors.HexColor("#1E90FF"),
    "lamp-crack":     colors.HexColor("#9400D3"),
    "flat-tire":      colors.HexColor("#FF1493"),
}

_SEVERITY_LW = {"minor": 2, "moderate": 4, "severe": 7}
_FALLBACK_COLOR = (255, 0, 0)
_PAGE_IMG_WIDTH = 16.0 * cm


def _annotate_image(img_bytes: bytes, damages: list[dict[str, Any]]) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    try:
        font = ImageFont.load_default(size=14)
        font_small = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
        font_small = font

    for dmg in damages:
        bbox = dmg.get("bbox") or []
        if len(bbox) != 4:
            continue
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw / 2) * iw)
        y1 = int((cy - bh / 2) * ih)
        x2 = int((cx + bw / 2) * iw)
        y2 = int((cy + bh / 2) * ih)
        x1, x2 = max(0, x1), min(iw, x2)
        y1, y2 = max(0, y1), min(ih, y2)

        dtype = dmg.get("type", "deformation")
        color = _TYPE_COLOR_PIL.get(dtype, _FALLBACK_COLOR)
        lw = _SEVERITY_LW.get(dmg.get("severity", "minor"), 2)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = f"{dtype} {dmg.get('severity', '')} {dmg.get('confidence', 0):.0%}"
        tb = draw.textbbox((0, 0), label, font=font)
        lbl_w = tb[2] - tb[0] + 8
        lbl_h = tb[3] - tb[1] + 6

        # x: clamp so label stays within image
        lbl_x = max(0, min(x1, iw - lbl_w))

        # y: prefer above box, else below, else inside top of box
        if y1 >= lbl_h:
            lbl_y = y1 - lbl_h
        elif y2 + lbl_h <= ih:
            lbl_y = y2
        else:
            lbl_y = max(0, y1)
        lbl_y = max(0, min(lbl_y, ih - lbl_h))

        draw.rectangle([lbl_x, lbl_y, lbl_x + lbl_w, lbl_y + lbl_h], fill=color)
        # tb[1] is ascender offset from anchor — subtract it so text sits inside rect
        draw.text((lbl_x + 4, lbl_y + 3 - tb[1]), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _rl_image(img_bytes: bytes) -> RLImage:
    buf = io.BytesIO(img_bytes)
    pil = Image.open(io.BytesIO(img_bytes))
    iw, ih = pil.size
    aspect = ih / iw
    return RLImage(buf, width=_PAGE_IMG_WIDTH, height=_PAGE_IMG_WIDTH * aspect)


def _damage_table(damages: list[dict[str, Any]], styles: Any) -> Table:
    header = ["Type", "Severity", "Confidence", "Parts at Risk", "Repair Action"]
    rows = [header]
    for dmg in damages:
        rows.append([
            dmg.get("type", "—"),
            dmg.get("severity", "—"),
            f"{dmg.get('confidence', 0):.1%}",
            ", ".join(dmg.get("parts_at_risk") or []) or "—",
            dmg.get("repair_action", "—"),
        ])

    col_widths = [3.2 * cm, 2.2 * cm, 2.4 * cm, 4.5 * cm, 5.7 * cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

    ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8F8F8"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])

    for row_idx, dmg in enumerate(damages, start=1):
        dtype = dmg.get("type", "deformation")
        rl_color = _TYPE_COLOR_RL.get(dtype, colors.red)
        ts.add("LEFTPADDING",  (0, row_idx), (0, row_idx), 0)
        ts.add("BACKGROUND",   (0, row_idx), (0, row_idx), colors.white)
        ts.add("LINEAFTER",    (0, row_idx), (0, row_idx), 4, rl_color)

    tbl.setStyle(ts)
    return tbl


def _legend_table() -> Table:
    items = list(_TYPE_COLOR_RL.items())
    row = []
    for dtype, rl_color in items:
        swatch = Table([[""]], colWidths=[0.4 * cm], rowHeights=[0.4 * cm])
        swatch.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), rl_color),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        row.append(swatch)
        row.append(Paragraph(dtype, ParagraphStyle("leg", fontSize=8, leading=10)))

    tbl = Table([row], colWidths=[0.5 * cm, 2.8 * cm] * len(items))
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def generate_damage_report(
    view_img_bytes: dict[str, bytes],
    per_view: dict[str, Any],
    payload: dict[str, Any],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title", fontSize=18, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A1A2E"), spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "subtitle", fontSize=9, textColor=colors.grey, spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "heading", fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A1A2E"), spaceBefore=10, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "body", fontSize=9, leading=13, spaceAfter=4,
    )

    policy = payload.get("policy_decision", {})
    decision = policy.get("decision", "UNKNOWN")
    decision_color = {
        "GRANT":              colors.HexColor("#28A745"),
        "GRANT_WITH_WARNING": colors.HexColor("#FFC107"),
        "DENY":               colors.HexColor("#DC3545"),
    }.get(decision, colors.grey)

    decision_style = ParagraphStyle(
        "decision", fontSize=14, fontName="Helvetica-Bold",
        textColor=decision_color, spaceAfter=6,
    )

    flowables: list[Any] = []

    flowables.append(Paragraph("UpsureAI — Damage Assessment Report", title_style))
    flowables.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"Views analyzed: {payload.get('total_views_analyzed', 0)}  ·  "
        f"Overall confidence: {payload.get('overall_confidence', 0):.1%}",
        subtitle_style,
    ))
    flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A1A2E")))
    flowables.append(Spacer(1, 0.3 * cm))

    damage_flag = "DAMAGE DETECTED" if payload.get("damage_detected") else "NO DAMAGE DETECTED"
    flowables.append(Paragraph(f"Result: {damage_flag}", decision_style))
    flowables.append(Paragraph(f"Policy Decision: {decision}", decision_style))

    deny_reasons = policy.get("deny_reasons") or []
    if deny_reasons:
        for r in deny_reasons:
            flowables.append(Paragraph(
                f"⚠ Deny reason — {r.get('type', '?')} ({r.get('severity', '?')}) "
                f"on {r.get('view', '?')} view",
                ParagraphStyle("warn", fontSize=9, textColor=colors.HexColor("#DC3545")),
            ))

    flowables.append(Spacer(1, 0.4 * cm))
    flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    flowables.append(Spacer(1, 0.2 * cm))

    for view_name, img_bytes in view_img_bytes.items():
        view_result = per_view.get(view_name, {})
        damages = view_result.get("damages") or []

        flowables.append(Paragraph(f"{view_name.capitalize()} View", heading_style))

        annotated = _annotate_image(img_bytes, damages)
        flowables.append(_rl_image(annotated))
        flowables.append(Spacer(1, 0.25 * cm))

        if damages:
            flowables.append(_damage_table(damages, styles))
        else:
            detected = view_result.get("damage_detected", False)
            msg = (
                "Binary model flagged damage but no localisation boxes returned."
                if detected else "No damage detected."
            )
            flowables.append(Paragraph(msg, body_style))

        flowables.append(Spacer(1, 0.5 * cm))
        flowables.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
        flowables.append(Spacer(1, 0.2 * cm))

    skipped = payload.get("skipped_views") or {}
    if skipped:
        flowables.append(Paragraph("Skipped Views (not a car)", heading_style))
        for vname, info in skipped.items():
            flowables.append(Paragraph(
                f"• {vname}: car confidence {info.get('car_confidence', 0):.1%}",
                body_style,
            ))
        flowables.append(Spacer(1, 0.3 * cm))

    flowables.append(Paragraph("Damage Type Legend", heading_style))
    flowables.append(_legend_table())
    flowables.append(Spacer(1, 0.3 * cm))
    flowables.append(Paragraph(
        "Line width indicates severity: thin = minor, medium = moderate, thick = severe.",
        ParagraphStyle("note", fontSize=8, textColor=colors.grey),
    ))

    doc.build(flowables)
    return buf.getvalue()
