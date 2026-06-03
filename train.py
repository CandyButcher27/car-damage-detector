"""
Car Damage Classifier — Training Script
----------------------------------------
Fine-tunes EfficientNet-B2 (pretrained on ImageNet) on the car damage dataset.
Trains binary classification: damaged (1) vs undamaged (0).

Usage:
    python train.py                        # train with defaults
    python train.py --epochs 30 --lr 3e-4
    python train.py --csv custom.csv --out my_model.pth

Outputs:
    damage_model.pth   — best model weights (PyTorch)
    damage_model.onnx  — ONNX export for CPU inference in production
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

from config import CFG

_T = CFG["training"]
DEFAULTS = dict(
    model_name   = _T["model_name"],
    img_size     = _T["img_size"],
    batch_size   = _T["batch_size"],
    epochs       = _T["epochs"],
    lr           = _T["lr"],
    weight_decay = _T["weight_decay"],
    num_workers  = _T["num_workers"],
    seed         = _T["seed"],
)


# ── Dataset ────────────────────────────────────────────────────────────────────
class CarDamageDataset(Dataset):
    def __init__(self, records: list[dict], transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records[idx]
        img   = Image.open(row["image_path"]).convert("RGB")
        img   = np.array(img)
        label = int(row["label"])

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


def get_transforms(img_size: int, train: bool):
    mean = _T["mean"]
    std  = _T["std"]
    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc, all_probs, all_labels


# ── Export ─────────────────────────────────────────────────────────────────────
def export_onnx(model, img_size: int, out_path: str, device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"✅  ONNX model saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="dataset.csv")
    parser.add_argument("--out",    default="damage_model.pth")
    parser.add_argument("--model",  default=DEFAULTS["model_name"])
    parser.add_argument("--epochs", type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",     type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch",  type=int,   default=DEFAULTS["batch_size"])
    args = parser.parse_args()

    torch.manual_seed(DEFAULTS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️   Device: {device}")

    # Load CSV
    records = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if Path(row["image_path"]).exists():
                records.append(row)
            else:
                print(f"⚠️  Missing image: {row['image_path']}")

    print(f"📊  Loaded {len(records)} records from {args.csv}")

    labels_arr = [int(r["label"]) for r in records]
    damaged    = sum(labels_arr)
    clean      = len(labels_arr) - damaged
    print(f"    Damaged: {damaged}  |  Clean: {clean}")

    # Train / val / test split — stratified so class balance is preserved
    train_rec, temp_rec = train_test_split(
        records, test_size=0.3, stratify=labels_arr, random_state=DEFAULTS["seed"]
    )
    temp_labels = [int(r["label"]) for r in temp_rec]
    val_rec, test_rec = train_test_split(
        temp_rec, test_size=0.5, stratify=temp_labels, random_state=DEFAULTS["seed"]
    )
    print(f"    Train: {len(train_rec)}  |  Val: {len(val_rec)}  |  Test: {len(test_rec)}")

    img_size = DEFAULTS["img_size"]
    train_ds = CarDamageDataset(train_rec, get_transforms(img_size, train=True))
    val_ds   = CarDamageDataset(val_rec,   get_transforms(img_size, train=False))
    test_ds  = CarDamageDataset(test_rec,  get_transforms(img_size, train=False))

    # Weighted sampler to handle class imbalance
    train_labels = [int(r["label"]) for r in train_rec]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights      = [1.0 / class_counts[l] for l in train_labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,  num_workers=DEFAULTS["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,     num_workers=DEFAULTS["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,     num_workers=DEFAULTS["num_workers"])

    model     = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=DEFAULTS["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    hr = "─" * 65

    print(f"\n{hr}")
    print(f"  Model: {args.model}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{hr}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_auc={vl_auc:.3f}  "
              f"[{elapsed:.1f}s]")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), args.out)
            print(f"  💾  New best AUC={best_auc:.4f} — saved {args.out}")

    # Test evaluation
    print(f"\n{hr}")
    print("  TEST SET EVALUATION")
    print(f"{hr}")
    model.load_state_dict(torch.load(args.out, map_location=device))
    _, te_acc, te_auc, probs, true_labels = eval_epoch(model, test_loader, criterion, device)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    print(f"  Accuracy : {te_acc:.4f}")
    print(f"  AUC-ROC  : {te_auc:.4f}")
    print(classification_report(true_labels, preds, target_names=["Clean", "Damaged"]))

    # ONNX export
    onnx_path = args.out.replace(".pth", ".onnx")
    export_onnx(model, img_size, onnx_path, device)


if __name__ == "__main__":
    main()
