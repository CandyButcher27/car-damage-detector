REPAIR_RULES: dict[tuple[str, str], dict] = {
    # ── car-part-crack ────────────────────────────────────────────────────────
    ("car-part-crack", "minor"):    {"action": "Repair — filler + repaint",              "replace": False},
    ("car-part-crack", "moderate"): {"action": "Replace cracked part",                   "replace": True},
    ("car-part-crack", "severe"):   {"action": "Replace part + inspect structural frame", "replace": True},

    # ── deformation (severity from bbox area) ─────────────────────────────────
    ("deformation", "minor"):    {"action": "Repair — paintless dent repair (PDR)",          "replace": False},
    ("deformation", "moderate"): {"action": "Repair — PDR + repaint panel",                  "replace": False},
    ("deformation", "severe"):   {"action": "Replace panel + inspect frame rails",            "replace": True},

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

    # ── scratches (absorbs paint-chips) ───────────────────────────────────────
    ("scratches", "minor"):    {"action": "Repair — machine polish + touch-up paint",     "replace": False},
    ("scratches", "moderate"): {"action": "Repair — repaint panel",                       "replace": False},
    ("scratches", "severe"):   {"action": "Repair — filler + full panel repaint",         "replace": False},
}


def get_severity(damage_area: float, car_area: float) -> str:
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
