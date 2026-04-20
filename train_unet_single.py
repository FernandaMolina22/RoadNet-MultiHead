from __future__ import annotations

"""
train_unet_single.py

Clean training script for the baseline single-head U-Net.

Features
--------
- Config section at the top
- Train / validation / test phases
- BCEWithLogits loss for binary mask prediction
- Dice and IoU reporting
- Best-checkpoint saving based on validation Dice
- Final test evaluation using the best checkpoint

Notes
-----
- Uses get_dataloaders(..., return_topology=False).
"""

from pathlib import Path
from typing import Dict
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn

from dataset_awr_seg_mosaic_split import get_dataloaders
#from dataset_awr_seg import get_dataloaders
from model_unet_single import build_unet_single


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"
OUTPUT_DIR = Path("runs_unet_single")

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.10
SEED = 42
NUM_WORKERS = 4
THRESHOLD = 0.5
PIN_MEMORY = True
SAVE_BEST_BY = "val_dice"   # options: "val_dice", "val_loss"


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def dice_coeff_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()



def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()



def save_history_csv(rows: list[Dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        dice_scores.append(dice_coeff_from_logits(logits, masks, threshold=threshold))
        iou_scores.append(iou_from_logits(logits, masks, threshold=threshold))
        n_batches += 1

    return {
        "loss": running_loss / max(1, n_batches),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = PIN_MEMORY and device.type == "cuda"

    print(f"Using device: {device}")
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        return_topology=False,
    )

    print("Building model...")
    model = build_unet_single(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if SAVE_BEST_BY == "val_dice" else "min",
        factor=0.5,
        patience=3,
    )

    best_metric = -float("inf") if SAVE_BEST_BY == "val_dice" else float("inf")
    ckpt_path = OUTPUT_DIR / "best_model_unet_single.pth"
    history: list[Dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=THRESHOLD,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "epoch_seconds": elapsed,
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  lr         = {current_lr:.6f}")
        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_loss   = {val_metrics['loss']:.4f}")
        print(f"  val_dice   = {val_metrics['dice']:.4f}")
        print(f"  val_iou    = {val_metrics['iou']:.4f}")
        print(f"  time       = {elapsed:.1f}s")

        if SAVE_BEST_BY == "val_dice":
            scheduler.step(val_metrics["dice"])
            is_best = val_metrics["dice"] > best_metric
            if is_best:
                best_metric = val_metrics["dice"]
        else:
            scheduler.step(val_metrics["loss"])
            is_best = val_metrics["loss"] < best_metric
            if is_best:
                best_metric = val_metrics["loss"]

        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "data_dir": DATA_DIR,
                        "num_epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "val_fraction": VAL_FRACTION,
                        "seed": SEED,
                        "threshold": THRESHOLD,
                        "save_best_by": SAVE_BEST_BY,
                    },
                    "best_metric_name": SAVE_BEST_BY,
                    "best_metric_value": best_metric,
                },
                ckpt_path,
            )
            print(f"  >> New best checkpoint saved to: {ckpt_path}")

        save_history_csv(history, OUTPUT_DIR / "training_history.csv")

    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=THRESHOLD,
    )

    print("\n===== FINAL TEST RESULTS (best checkpoint) =====")
    print(f"test_loss = {test_metrics['loss']:.4f}")
    print(f"test_dice = {test_metrics['dice']:.4f}")
    print(f"test_iou  = {test_metrics['iou']:.4f}")

    with (OUTPUT_DIR / "test_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("UNetSingle final test metrics\n")
        f.write(f"test_loss={test_metrics['loss']:.6f}\n")
        f.write(f"test_dice={test_metrics['dice']:.6f}\n")
        f.write(f"test_iou={test_metrics['iou']:.6f}\n")


if __name__ == "__main__":
    main()
