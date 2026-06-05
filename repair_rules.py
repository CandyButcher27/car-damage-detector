REPAIR_RULES: dict[tuple[str, str], dict] = {
    # ── car-part-crack ────────────────────────────────────────────────────────
    ("car-part-crack", "minor"):    {"action": "Repair — filler + repaint",              "replace": False},
    ("car-part-crack", "moderate"): {"action": "Replace cracked part",                   "replace": True},
    ("car-part-crack", "severe"):   {"action": "Replace part + inspect structural frame", "replace": True},

    # ── detachment ────────────────────────────────────────────────────────────
    ("detachment", "minor"):    {"action": "Reattach — adhesive/clip replacement",        "replace": False},
    ("detachment", "moderate"): {"action": "Replace part + mounting bracket",             "replace": True},
    ("detachment", "severe"):   {"action": "Replace part + structural integrity check",   "replace": True},

    # ── flat-tire ─────────────────────────────────────────────────────────────
    ("flat-tire", "minor"):    {"action": "Repair — patch tire",                          "replace": False},
    ("flat-tire", "moderate"): {"action": "Replace tire",                                 "replace": True},
    ("flat-tire", "severe"):   {"action": "Replace tire + inspect rim and suspension",    "replace": True},

    # ── glass-crack ───────────────────────────────────────────────────────────
    ("glass-crack", "minor"):    {"action": "Repair — resin injection (if single crack)", "replace": False},
    ("glass-crack", "moderate"): {"action": "Replace glass panel",                        "replace": True},
    ("glass-crack", "severe"):   {"action": "Replace glass + inspect frame seals",        "replace": True},

    # ── lamp-crack ────────────────────────────────────────────────────────────
    ("lamp-crack", "minor"):    {"action": "Replace lamp lens",                           "replace": True},
    ("lamp-crack", "moderate"): {"action": "Replace full lamp assembly",                  "replace": True},
    ("lamp-crack", "severe"):   {"action": "Replace lamp assembly + inspect mount",       "replace": True},

    # ── minor-deformation ─────────────────────────────────────────────────────
    ("minor-deformation", "minor"):    {"action": "Repair — paintless dent repair (PDR)",     "replace": False},
    ("minor-deformation", "moderate"): {"action": "Repair — PDR + repaint",                   "replace": False},
    ("minor-deformation", "severe"):   {"action": "Replace panel",                            "replace": True},

    # ── moderate-deformation ──────────────────────────────────────────────────
    ("moderate-deformation", "minor"):    {"action": "Repair — PDR + repaint",                "replace": False},
    ("moderate-deformation", "moderate"): {"action": "Replace panel",                         "replace": True},
    ("moderate-deformation", "severe"):   {"action": "Replace panel + inspect frame rails",   "replace": True},

    # ── paint-chips ───────────────────────────────────────────────────────────
    ("paint-chips", "minor"):    {"action": "Repair — touch-up paint",                    "replace": False},
    ("paint-chips", "moderate"): {"action": "Repair — repaint panel",                     "replace": False},
    ("paint-chips", "severe"):   {"action": "Repair — full repaint with primer coat",     "replace": False},

    # ── scratches ─────────────────────────────────────────────────────────────
    ("scratches", "minor"):    {"action": "Repair — machine polish + touch-up paint",     "replace": False},
    ("scratches", "moderate"): {"action": "Repair — repaint panel",                       "replace": False},
    ("scratches", "severe"):   {"action": "Repair — filler + full panel repaint",         "replace": False},

    # ── severe-deformation ────────────────────────────────────────────────────
    ("severe-deformation", "minor"):    {"action": "Replace panel + structural check",    "replace": True},
    ("severe-deformation", "moderate"): {"action": "Replace panel + frame realignment",   "replace": True},
    ("severe-deformation", "severe"):   {"action": "Replace panel + frame repair + structural integrity inspection", "replace": True},

    # ── side-mirror-crack ─────────────────────────────────────────────────────
    ("side-mirror-crack", "minor"):    {"action": "Replace mirror glass",                 "replace": True},
    ("side-mirror-crack", "moderate"): {"action": "Replace full mirror assembly",         "replace": True},
    ("side-mirror-crack", "severe"):   {"action": "Replace mirror assembly + inspect mount", "replace": True},
}


def get_severity(damage_area: float, car_area: float) -> str:
    """
    damage_area, car_area: pixel areas (w * h from bbox).
    Returns: 'minor' | 'moderate' | 'severe'
    """
    if car_area <= 0:
        return "minor"
    ratio = damage_area / car_area
    if ratio < 0.05:
        return "minor"
    if ratio < 0.15:
        return "moderate"
    return "severe"


def lookup(damage_class: str, severity: str) -> dict:
    return REPAIR_RULES.get(
        (damage_class, severity),
        {"action": "Manual inspection required", "replace": False}
    )
