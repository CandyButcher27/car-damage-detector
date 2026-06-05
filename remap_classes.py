"""
Remap 11 Roboflow CarDD classes → 6 merged classes.

Merges:
  minor-deformation + moderate-deformation + severe-deformation + detachment → deformation
  paint-chips → scratches
  side-mirror-crack → car-part-crack

New classes (6):
  0: car-part-crack
  1: deformation
  2: flat-tire
  3: glass-crack
  4: lamp-crack
  5: scratches

Usage:
    python remap_classes.py
Output: cardd_remapped.zip  — upload this to Drive instead of cardd_roboflow.zip
"""

import shutil, zipfile
from pathlib import Path

SRC_ZIP  = Path("cardd_roboflow.zip")
WORK_DIR = Path("cardd_remapped")
OUT_ZIP  = Path("cardd_remapped.zip")

OLD_CLASSES = [
    "car-part-crack",     # 0
    "detachment",         # 1
    "flat-tire",          # 2
    "glass-crack",        # 3
    "lamp-crack",         # 4
    "minor-deformation",  # 5
    "moderate-deformation", # 6
    "paint-chips",        # 7
    "scratches",          # 8
    "severe-deformation", # 9
    "side-mirror-crack",  # 10
]

NEW_CLASSES = [
    "car-part-crack",  # 0
    "deformation",     # 1
    "flat-tire",       # 2
    "glass-crack",     # 3
    "lamp-crack",      # 4
    "scratches",       # 5
]

# old_id → new_id
REMAP = {
    0:  0,  # car-part-crack → car-part-crack
    1:  1,  # detachment → deformation
    2:  2,  # flat-tire → flat-tire
    3:  3,  # glass-crack → glass-crack
    4:  4,  # lamp-crack → lamp-crack
    5:  1,  # minor-deformation → deformation
    6:  1,  # moderate-deformation → deformation
    7:  5,  # paint-chips → scratches
    8:  5,  # scratches → scratches
    9:  1,  # severe-deformation → deformation
    10: 0,  # side-mirror-crack → car-part-crack
}


def remap_label_file(src: Path, dst: Path):
    lines = src.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_id = int(parts[0])
        new_id = REMAP.get(old_id, old_id)
        new_lines.append(f"{new_id} {' '.join(parts[1:])}")
    dst.write_text("\n".join(new_lines))


def main():
    if OUT_ZIP.exists():
        print(f"{OUT_ZIP} already exists — delete to re-run.")
        return

    print("Extracting source zip...")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall(WORK_DIR)

    # Find CarDD-3 root
    candidates = list(WORK_DIR.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError("data.yaml not found inside zip")
    data_root = candidates[0].parent
    print(f"Dataset root: {data_root}")

    # Remap all label files
    total = 0
    for split in ["train", "valid", "test"]:
        label_dir = data_root / split / "labels"
        if not label_dir.exists():
            continue
        for lbl in label_dir.glob("*.txt"):
            remap_label_file(lbl, lbl)
            total += 1
        print(f"  {split}: {total} label files remapped")
        total = 0

    # Write new data.yaml
    yaml_text = (
        f"path: .\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(NEW_CLASSES)}\n"
        f"names:\n"
    )
    for name in NEW_CLASSES:
        yaml_text += f"  - {name}\n"
    (data_root / "data.yaml").write_text(yaml_text)
    print("data.yaml updated.")

    # Re-zip
    print("Zipping remapped dataset...")
    shutil.make_archive("cardd_remapped", "zip", WORK_DIR)
    shutil.rmtree(WORK_DIR)
    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"Done: {OUT_ZIP}  ({size_mb:.0f} MB)")
    print(f"\nUpload {OUT_ZIP} to: MyDrive/UpSureAI/cardd_remapped.zip")
    print("Then update DRIVE_DATASET_ZIP in cell-4 of train_yolo_colab.ipynb to 'cardd_remapped.zip'")


if __name__ == "__main__":
    main()
