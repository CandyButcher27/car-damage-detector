"""Card crop + deskew + orientation for the Mulkiya OCR pipeline.

Finds card-shaped quadrilaterals in a photo, perspective-warps each to a flat
rectangle, orientation-corrects (0/90/180/270), and (when several cards are in
frame) picks the one whose OCR carries Mulkiya anchor strings. Strips background,
deskews rotated cards, and isolates the Mulkiya from multi-document frames.

Imported by `ocr_simple_test.py --make_crop` (the quality-triggered crop fallback
in the Mulkiya pipeline). Also runnable standalone for inspection.

Usage:
    python card_crop.py vlm_check/spot_0.jpg [spot_1.jpg ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MULKIYA_ANCHORS_LATIN = ("MOTORVEHICLELIC",)
MULKIYA_ANCHORS_AR = ("رخصة مركبة", "مركبة رخصة", "نوع اللوحة", "رقم اللوحة", "اللوحة")

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]      # BR: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # TR: smallest y-x
    rect[3] = pts[np.argmax(diff)]   # BL: largest y-x
    return rect


def _warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect
    wa = np.linalg.norm(br - bl)
    wb = np.linalg.norm(tr - tl)
    ha = np.linalg.norm(tr - br)
    hb = np.linalg.norm(tl - bl)
    w = int(max(wa, wb))
    h = int(max(ha, hb))
    if w < 10 or h < 10:
        return img
    # Mulkiya is landscape; if the detected quad came out portrait, rotate.
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    if h > w:  # rotated card → make landscape
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _quad_from_contour(c: np.ndarray) -> np.ndarray | None:
    """Prefer a clean 4-point approx; fall back to the min-area rotated rect."""
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
    # Rounded corners / rotation / imperfect edges → rotated bounding box.
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # Only trust the rect if the contour actually fills most of it (a real card,
    # not an L-shaped edge fragment).
    if cv2.contourArea(c) < 0.7 * (rect[1][0] * rect[1][1]):
        return None
    return box.astype("float32")


def find_card_candidates(img: np.ndarray) -> list[np.ndarray]:
    """Return warped crops of card-like quadrilaterals, largest first."""
    scale = 1000.0 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    small_area = small.shape[0] * small.shape[1]
    inv = 1.0 / scale if scale < 1 else 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Two detectors merged: Canny edges (works on textured/photographed cards)
    # and Otsu threshold (works on a card against a plain background).
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours: list = []
    for mask in (edges, thr, cv2.bitwise_not(thr)):
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    crops: list[np.ndarray] = []
    seen_centers: list[tuple[float, float]] = []
    for c in contours:
        if cv2.contourArea(c) < 0.05 * small_area:
            continue
        quad = _quad_from_contour(c)
        if quad is None:
            continue
        rect = _order_points(quad)
        wd = np.linalg.norm(rect[1] - rect[0])
        ht = np.linalg.norm(rect[3] - rect[0])
        if min(wd, ht) < 1:
            continue
        ar = max(wd, ht) / min(wd, ht)
        if not (1.2 <= ar <= 2.8):
            continue
        # De-dup near-identical detections from the different masks.
        cx, cy = rect[:, 0].mean(), rect[:, 1].mean()
        if any(abs(cx - sx) < 20 and abs(cy - sy) < 20 for sx, sy in seen_centers):
            continue
        seen_centers.append((cx, cy))
        crops.append(_warp(img, rect * inv))
    return crops


def _ocr(img: np.ndarray) -> tuple[str, float]:
    res = _get_engine()(img)
    if res is None or not res.txts:
        return "", 0.0
    return " ".join(res.txts), float(np.mean(res.scores)) if res.scores else 0.0


def _has_mulkiya_anchor(text: str) -> bool:
    import re
    latin = re.sub(r"[^A-Za-z]", "", text).upper()
    if any(a in latin for a in MULKIYA_ANCHORS_LATIN):
        return True
    return any(a in text for a in MULKIYA_ANCHORS_AR)


_ROTATIONS = (
    (0, lambda im: im),
    (90, lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)),
    (180, lambda im: cv2.rotate(im, cv2.ROTATE_180)),
    (270, lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)),
)


def _header_score(crop: np.ndarray) -> float:
    """How 'header-like' is the top strip? The Mulkiya front carries a light-blue
    ROP header band + emblem across the top; the body is mostly white. The band
    is more SATURATED and a touch DARKER than the body, so the correct (upright)
    orientation maximises top-strip saturation. Pure pixel math — no OCR."""
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return -1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    strip = max(1, int(h * 0.16))
    top = hsv[:strip]
    sat = float(top[:, :, 1].mean())            # higher on the coloured band
    val_pen = float(top[:, :, 2].mean()) / 255.0  # white body is bright → penalise
    return sat - 12.0 * val_pen


def _orient_cheap(crop: np.ndarray) -> np.ndarray:
    """Pick the upright orientation by header position — NO OCR.

    Tries all 4 rotations and keeps the one whose top strip looks most like the
    Mulkiya's coloured header band. Replaces the old 4x-OCR `_best_orientation`
    for the orientation decision (aspect is already landscape-corrected in _warp,
    so this mostly resolves the 0-vs-180 flip, but evaluating all 4 also rescues
    an un-warped full-image fallback)."""
    best = (crop, -1e9)
    for _deg, rot in _ROTATIONS:
        r = rot(crop)
        sc = _header_score(r)
        if sc > best[1]:
            best = (r, sc)
    return best[0]


def _best_orientation(crop: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """A warped card can land at 0/90/180/270. OCR all four; score = anchor
    present (dominant) + mean confidence. Return (oriented_crop, has_anchor, score).

    Fast path: if the upright (0°) read already has an anchor with decent
    confidence, the card is upright — skip the other three rotations. Only
    genuinely rotated/ambiguous cards pay the full 4-way cost."""
    best = (crop, False, -1.0)
    for _deg, rot in _ROTATIONS:
        r = rot(crop)
        text, conf = _ocr(r)
        anchor = _has_mulkiya_anchor(text)
        score = (1.0 if anchor else 0.0) + conf
        if score > best[2]:
            best = (r, anchor, score)
        if _deg == 0 and anchor and conf >= 0.6:
            break  # upright and confident → done
    return best


def choose_mulkiya_crop(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (best_crop, reason). Orientation is resolved by the cheap pixel-based
    header check (NO 4x-OCR). A single detected card costs zero OCR; only a
    multi-card frame pays one OCR per candidate to pick the Mulkiya by anchor."""
    frame_area = img.shape[0] * img.shape[1]
    candidates = find_card_candidates(img)
    if not candidates:
        return _orient_cheap(img), "no_quad->full_image(cheap)"
    if len(candidates) == 1:
        return _orient_cheap(candidates[0]), "single_card(cheap)"

    # Multiple candidates: orient each cheaply, then a single OCR per candidate to
    # find the one carrying Mulkiya anchors (isolates it from a second document).
    anchor_matches = []  # (oriented_crop, area)
    for c in sorted(candidates, key=lambda c: c.shape[0] * c.shape[1])[:6]:
        oriented = _orient_cheap(c)
        text, _conf = _ocr(oriented)
        if _has_mulkiya_anchor(text):
            anchor_matches.append((oriented, c.shape[0] * c.shape[1]))
    if anchor_matches:
        isolated = [(o, a) for (o, a) in anchor_matches if a < 0.88 * frame_area]
        pick = min(isolated or anchor_matches, key=lambda t: t[1])
        tag = "isolated" if isolated else "near_full"
        return pick[0], f"{tag}_card_of_{len(anchor_matches)}_anchored(cheap)"
    # No anchor anywhere — largest quad, cheap-oriented.
    return _orient_cheap(max(candidates, key=lambda c: c.shape[0] * c.shape[1])), "largest_quad_no_anchor(cheap)"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("vlm_check").glob("spot_*.jpg"))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: could not read")
            continue
        crop, reason = choose_mulkiya_crop(img)
        out = p.with_name(f"{p.stem}_crop.jpg")
        cv2.imwrite(str(out), crop)
        print(f"{p.name}: {img.shape[1]}x{img.shape[0]} -> {crop.shape[1]}x{crop.shape[0]}  [{reason}]  -> {out.name}")


if __name__ == "__main__":
    main()
