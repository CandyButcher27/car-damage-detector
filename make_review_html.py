"""Generate a self-contained HTML viewer for mulkiya batch results.

Usage:
    python make_review_html.py <zip> <results_dir> [--out review.html]
"""
import argparse
import base64
import csv
import json
import sys
import zipfile
from pathlib import Path

FIELDS = [
    "document_type", "plate_number", "vin_or_chassis",
    "engine_cc", "empty_weight_kg", "max_load_kg", "seats",
    "year", "model_year", "issue_date", "expiry_date",
    "vehicle_type", "color",
]

VALID_FIELDS = {
    "plate_number", "vin_or_chassis", "engine_cc",
    "empty_weight_kg", "max_load_kg", "seats",
    "year", "model_year", "issue_date", "expiry_date",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip")
    p.add_argument("results_dir")
    p.add_argument("--out", default="review.html")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        sys.exit(f"results.csv not found in {results_dir}")

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Build {filename: row} map
    doc_map = {r["doc"]: r for r in rows if r.get("doc")}

    # Extract images from zip as base64
    img_b64 = {}
    try:
        with zipfile.ZipFile(args.zip) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base in doc_map:
                    ext = Path(name).suffix.lower()
                    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                    data = zf.read(name)
                    img_b64[base] = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Warning: could not read zip: {e}", file=sys.stderr)

    cards_html = []
    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        error = row.get("error", "")
        doc_type = row.get("document_type", "")

        # Header color
        if error:
            hdr_color = "#b0b0b0"
            badge = "ERROR"
        elif doc_type == "mulkiya":
            hdr_color = "#2e7d32"
            badge = "MULKIYA"
        elif doc_type == "driving_licence":
            hdr_color = "#c62828"
            badge = "LICENCE"
        else:
            hdr_color = "#e65100"
            badge = doc_type.upper() or "UNKNOWN"

        # Image
        b64 = img_b64.get(doc, "")
        if b64:
            img_tag = f'<img src="{b64}" style="max-width:100%;max-height:480px;object-fit:contain;border-radius:4px;">'
        else:
            img_tag = '<div style="color:#999;padding:40px;text-align:center;">image not available</div>'

        # Fields table
        rows_html = []
        for field in FIELDS:
            val = row.get(field, "") or ""
            valid_key = f"{field}__valid"
            valid = row.get(valid_key, "") if field in VALID_FIELDS else ""

            if valid == "BAD":
                val_style = "color:#c62828;font-weight:600;"
                badge_html = ' <span style="font-size:10px;background:#c62828;color:#fff;border-radius:3px;padding:1px 4px;">BAD</span>'
            elif valid == "ok":
                val_style = "color:#2e7d32;"
                badge_html = ""
            else:
                val_style = "color:#333;"
                badge_html = ""

            val_display = val if val else '<span style="color:#aaa">—</span>'
            rows_html.append(
                f'<tr>'
                f'<td style="padding:4px 8px 4px 0;color:#666;font-size:12px;white-space:nowrap;vertical-align:top;">{field}</td>'
                f'<td style="padding:4px 0;font-size:13px;{val_style}">{val_display}{badge_html}</td>'
                f'</tr>'
            )

        fields_table = f'<table style="border-collapse:collapse;width:100%">{"".join(rows_html)}</table>'

        short_name = doc[:40] + ("…" if len(doc) > 40 else "")

        card = f'''
<div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.08);">
  <div style="background:{hdr_color};color:#fff;padding:8px 12px;display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:12px;opacity:.85;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%;" title="{doc}">{i+1}. {short_name}</span>
    <span style="font-size:11px;font-weight:700;letter-spacing:.5px;">{badge}</span>
  </div>
  <div style="display:flex;gap:0;">
    <div style="flex:0 0 55%;padding:12px;background:#fafafa;border-right:1px solid #eee;display:flex;align-items:center;justify-content:center;">
      {img_tag}
    </div>
    <div style="flex:1;padding:12px;overflow-y:auto;">
      {fields_table}
      {"<div style='margin-top:8px;padding:6px;background:#fff3e0;border-radius:4px;font-size:11px;color:#bf360c;'>" + error + "</div>" if error else ""}
    </div>
  </div>
</div>'''
        cards_html.append(card)

    total = len(rows)
    mulkiyas = sum(1 for r in rows if r.get("document_type") == "mulkiya")
    errors = sum(1 for r in rows if r.get("error"))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mulkiya OCR Review</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f5f5f5; padding:20px; }}
  h1 {{ font-size:20px; color:#222; margin-bottom:6px; }}
  .summary {{ color:#555; font-size:13px; margin-bottom:20px; }}
</style>
</head>
<body>
<h1>Mulkiya OCR Review</h1>
<p class="summary">{total} docs processed &nbsp;·&nbsp; {mulkiyas} mulkiya &nbsp;·&nbsp; {errors} errors &nbsp;·&nbsp; Results: {results_dir}</p>
{"".join(cards_html)}
</body>
</html>'''

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
