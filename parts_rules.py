from pathlib import Path
import yaml

CFG = yaml.safe_load(Path("config.yaml").read_text())
YOLO_CLASSES = CFG["yolo"]["classes"]

# (damage_class, image_region) → parts at risk
# image_region: top_left | top_right | bottom_left | bottom_right | center
# region derived from bbox center relative to car bbox (not full image)

PARTS_RULES: dict[tuple[str, str], list[str]] = {
    # ── car-part-crack ────────────────────────────────────────────────────────
    ("car-part-crack", "top_left"):     ["hood", "left_fender", "windshield_frame"],
    ("car-part-crack", "top_right"):    ["hood", "right_fender", "windshield_frame"],
    ("car-part-crack", "bottom_left"):  ["front_bumper", "radiator_grille", "left_rocker_panel"],
    ("car-part-crack", "bottom_right"): ["front_bumper", "radiator_grille", "right_rocker_panel"],
    ("car-part-crack", "center"):       ["door_panel", "body_frame", "sill"],

    # ── detachment ────────────────────────────────────────────────────────────
    ("detachment", "top_left"):     ["hood_latch", "left_fender_trim", "roof_trim"],
    ("detachment", "top_right"):    ["hood_latch", "right_fender_trim", "roof_trim"],
    ("detachment", "bottom_left"):  ["front_bumper", "left_side_skirt", "undertray"],
    ("detachment", "bottom_right"): ["front_bumper", "right_side_skirt", "undertray"],
    ("detachment", "center"):       ["door_trim", "body_kit", "running_board"],

    # ── flat-tire ─────────────────────────────────────────────────────────────
    ("flat-tire", "top_left"):     ["left_front_tire", "left_front_rim", "left_front_brake_caliper"],
    ("flat-tire", "top_right"):    ["right_front_tire", "right_front_rim", "right_front_brake_caliper"],
    ("flat-tire", "bottom_left"):  ["left_rear_tire", "left_rear_rim", "left_rear_suspension"],
    ("flat-tire", "bottom_right"): ["right_rear_tire", "right_rear_rim", "right_rear_suspension"],
    ("flat-tire", "center"):       ["tire", "rim", "suspension"],

    # ── glass-crack ───────────────────────────────────────────────────────────
    ("glass-crack", "top_left"):     ["windshield", "left_a_pillar", "wiper_linkage"],
    ("glass-crack", "top_right"):    ["windshield", "right_a_pillar", "wiper_linkage"],
    ("glass-crack", "bottom_left"):  ["rear_windshield", "left_c_pillar", "rear_wiper"],
    ("glass-crack", "bottom_right"): ["rear_windshield", "right_c_pillar", "rear_wiper"],
    ("glass-crack", "center"):       ["side_window", "door_seal", "window_regulator"],

    # ── lamp-crack ────────────────────────────────────────────────────────────
    ("lamp-crack", "top_left"):     ["left_headlight_assembly", "left_indicator", "left_daytime_running_light"],
    ("lamp-crack", "top_right"):    ["right_headlight_assembly", "right_indicator", "right_daytime_running_light"],
    ("lamp-crack", "bottom_left"):  ["left_tail_light", "left_reverse_light", "left_brake_light"],
    ("lamp-crack", "bottom_right"): ["right_tail_light", "right_reverse_light", "right_brake_light"],
    ("lamp-crack", "center"):       ["lamp_assembly", "indicator"],

    # ── minor-deformation ─────────────────────────────────────────────────────
    ("minor-deformation", "top_left"):     ["hood", "left_fender"],
    ("minor-deformation", "top_right"):    ["hood", "right_fender"],
    ("minor-deformation", "bottom_left"):  ["front_bumper", "left_quarter_panel"],
    ("minor-deformation", "bottom_right"): ["front_bumper", "right_quarter_panel"],
    ("minor-deformation", "center"):       ["door_panel"],

    # ── moderate-deformation ──────────────────────────────────────────────────
    ("moderate-deformation", "top_left"):     ["hood", "left_fender", "left_a_pillar"],
    ("moderate-deformation", "top_right"):    ["hood", "right_fender", "right_a_pillar"],
    ("moderate-deformation", "bottom_left"):  ["front_bumper", "radiator_support", "left_frame_rail"],
    ("moderate-deformation", "bottom_right"): ["front_bumper", "radiator_support", "right_frame_rail"],
    ("moderate-deformation", "center"):       ["door_panel", "b_pillar", "body_frame"],

    # ── paint-chips ───────────────────────────────────────────────────────────
    ("paint-chips", "top_left"):     ["hood", "left_fender"],
    ("paint-chips", "top_right"):    ["hood", "right_fender"],
    ("paint-chips", "bottom_left"):  ["front_bumper", "left_rocker_panel"],
    ("paint-chips", "bottom_right"): ["front_bumper", "right_rocker_panel"],
    ("paint-chips", "center"):       ["door_panel"],

    # ── scratches ─────────────────────────────────────────────────────────────
    ("scratches", "top_left"):     ["hood", "left_fender"],
    ("scratches", "top_right"):    ["hood", "right_fender"],
    ("scratches", "bottom_left"):  ["front_bumper", "left_rocker_panel"],
    ("scratches", "bottom_right"): ["rear_bumper", "right_rocker_panel"],
    ("scratches", "center"):       ["door_panel"],

    # ── severe-deformation ────────────────────────────────────────────────────
    ("severe-deformation", "top_left"):     ["hood", "left_fender", "left_frame_rail", "radiator_support"],
    ("severe-deformation", "top_right"):    ["hood", "right_fender", "right_frame_rail", "radiator_support"],
    ("severe-deformation", "bottom_left"):  ["front_bumper", "left_frame_rail", "radiator_support", "crash_beam"],
    ("severe-deformation", "bottom_right"): ["front_bumper", "right_frame_rail", "radiator_support", "crash_beam"],
    ("severe-deformation", "center"):       ["door_panel", "b_pillar", "body_frame", "floor_pan"],

    # ── side-mirror-crack ─────────────────────────────────────────────────────
    ("side-mirror-crack", "top_left"):     ["left_side_mirror_assembly", "left_mirror_glass", "left_mirror_motor"],
    ("side-mirror-crack", "top_right"):    ["right_side_mirror_assembly", "right_mirror_glass", "right_mirror_motor"],
    ("side-mirror-crack", "bottom_left"):  ["left_side_mirror_assembly", "left_mirror_glass"],
    ("side-mirror-crack", "bottom_right"): ["right_side_mirror_assembly", "right_mirror_glass"],
    ("side-mirror-crack", "center"):       ["side_mirror_assembly", "mirror_glass"],
}


def get_image_region(bbox: list[float], car_bbox: list[float]) -> str:
    """
    Returns image region of damage relative to car bbox.
    bbox and car_bbox: [x_center, y_center, width, height] normalized.
    """
    dx = bbox[0] - car_bbox[0]
    dy = bbox[1] - car_bbox[1]
    car_w = car_bbox[2]
    car_h = car_bbox[3]

    rel_x = dx / (car_w / 2) if car_w > 0 else 0
    rel_y = dy / (car_h / 2) if car_h > 0 else 0

    if abs(rel_x) < 0.3 and abs(rel_y) < 0.3:
        return "center"
    if rel_y <= 0:
        return "top_left" if rel_x <= 0 else "top_right"
    return "bottom_left" if rel_x <= 0 else "bottom_right"


def lookup(damage_class: str, image_region: str) -> list[str]:
    return PARTS_RULES.get((damage_class, image_region), [])
