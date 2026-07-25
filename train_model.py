"""
NatureDex AI — Model Training (Mac / Apple Silicon version)
===========================================================
Fine-tunes EfficientNetV2-S on the downloaded NC wildlife dataset.
This version runs on a Mac using Apple's MPS (Metal) GPU backend.

DIFFERENCES FROM THE WINDOWS/CUDA VERSION:
  - Uses MPS if available (Apple Silicon), else CPU. No CUDA.
  - Removes CUDA-only mixed precision (autocast + GradScaler) — Macs train
    in normal fp32.
  - DataLoader tuned for macOS (workers=0, no pin_memory) to avoid crashes.
  - ONNX export runs on CPU so it works regardless of MPS.

Usage:
    pip install torch torchvision torchaudio
    pip install Pillow tqdm scikit-learn matplotlib
    python train_model.py

Output (same as before):
    models/
    ├── naturedex_nc_v1.pth
    ├── naturedex_nc_v1.onnx
    ├── label_map.json
    └── training_log.csv
"""

import os
# Safety net: if any op isn't supported on MPS, fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import csv
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_DIR  = Path("dataset")
MODELS_DIR   = Path("models")
MODEL_NAME   = "naturedex_nc_v1"

IMG_SIZE     = 224          # must match download_dataset.py
BATCH_SIZE   = 32           # lower to 16 if you hit memory pressure
NUM_EPOCHS   = 20           # phase 1: train head (raise to 30 if you have time)
FINETUNE_EPOCHS = 10        # phase 2: fine-tune whole model (raise to 15 if time)
LR_HEAD      = 1e-3
LR_FINETUNE  = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0            # macOS: 0 is safest. Try 2-4 only if stable.

# Device: MPS (Apple Silicon GPU) > CPU. (No CUDA on Mac.)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ── Data Transforms ────────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(20),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_datasets():
    train_dir = DATASET_DIR / "train"
    val_dir   = DATASET_DIR / "val"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR.resolve()}\n"
            "Run download_dataset.py first."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Classes:            {len(train_dataset.classes)}")
    return train_loader, val_loader, train_dataset.classes


# ── Model Setup ────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


# ── Training Loop (no AMP — plain fp32, works on MPS/CPU) ────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for inputs, labels in tqdm(loader, desc="  train", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    for inputs, labels in tqdm(loader, desc="  val  ", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, 100.0 * correct / total


# ── Label Map ──────────────────────────────────────────────────────────────────

def save_label_map(classes, source_label_map):
    label_map_out = {}
    for idx, slug in enumerate(classes):
        meta = source_label_map.get(slug, {})
        label_map_out[str(idx)] = {
            "slug":            slug,
            "common_name":     meta.get("common_name", slug.replace("_", " ").title()),
            "scientific_name": meta.get("scientific_name", ""),
            "taxon_id":        meta.get("taxon_id", None),
            "iconic_group":    meta.get("iconic_group", ""),
        }
    out_path = MODELS_DIR / "label_map.json"
    with open(out_path, "w") as f:
        json.dump(label_map_out, f, indent=2)
    print(f"  Label map saved -> {out_path}")
    return label_map_out


# ── ONNX Export (on CPU) ─────────────────────────────────────────────────────────

def export_onnx(model, num_classes):
    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    out_path = MODELS_DIR / f"{MODEL_NAME}.onnx"
    torch.onnx.export(
        model_cpu, dummy, out_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"  ONNX model saved -> {out_path}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_history(history):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train")
    ax1.plot(epochs, [h["val_loss"]   for h in history], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="Train")
    ax2.plot(epochs, [h["val_acc"]   for h in history], label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    out = MODELS_DIR / "training_curves.png"
    plt.savefig(out, dpi=120)
    print(f"  Training curves -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NatureDex AI — Custom NC Model Training (Mac)")
    print(f"Device: {DEVICE.type}")
    if DEVICE.type == "cpu":
        print("  WARNING: no Apple-Silicon GPU (MPS) detected — training on CPU.")
        print("  This will be VERY slow. Consider fewer species or the PC's GPU.")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    source_label_map = {}
    label_map_src = DATASET_DIR / "label_map.json"
    if label_map_src.exists():
        with open(label_map_src) as f:
            source_label_map = json.load(f)

    print("\nLoading dataset...")
    train_loader, val_loader, classes = load_datasets()
    num_classes = len(classes)

    print(f"\nBuilding EfficientNetV2-S for {num_classes} classes...")
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    best_val_acc = 0.0
    best_weights = None

    # ── Phase 1: Train head only ──
    print(f"\n-- Phase 1: Training classification head ({NUM_EPOCHS} epochs) --")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "phase": 1,
                        "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 2)})
        marker = " <- best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train {train_acc:.1f}% | "
              f"val {val_acc:.1f}% | {elapsed:.0f}s{marker}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # ── Phase 2: Fine-tune entire model ──
    print(f"\n-- Phase 2: Fine-tuning full model ({FINETUNE_EPOCHS} epochs) --")
    unfreeze_model(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        total_epoch = NUM_EPOCHS + epoch
        history.append({"epoch": total_epoch, "phase": 2,
                        "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 2)})
        marker = " <- best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:3d}/{FINETUNE_EPOCHS} | train {train_acc:.1f}% | "
              f"val {val_acc:.1f}% | {elapsed:.0f}s{marker}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # ── Save outputs ──
    print(f"\n-- Saving outputs (best val accuracy: {best_val_acc:.1f}%) --")
    model.load_state_dict(best_weights)

    pth_path = MODELS_DIR / f"{MODEL_NAME}.pth"
    torch.save({
        "model_state_dict": best_weights,
        "classes": classes,
        "num_classes": num_classes,
        "val_accuracy": best_val_acc,
        "img_size": IMG_SIZE,
        "architecture": "efficientnet_v2_s",
    }, pth_path)
    print(f"  Model saved -> {pth_path}")

    save_label_map(classes, source_label_map)

    try:
        export_onnx(model, num_classes)
    except Exception as e:
        print(f"  ONNX export failed (non-critical): {e}")

    log_path = MODELS_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "phase", "train_loss",
                                               "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    try:
        plot_history(history)
    except Exception as e:
        print(f"  Plot failed (non-critical): {e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model: {pth_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()