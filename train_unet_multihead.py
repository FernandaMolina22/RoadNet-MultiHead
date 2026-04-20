from __future__ import annotations

"""
train_unet_multihead.py

Clean training script for the topology-aware multi-head U-Net.

Features
--------
- Config section at the top
- Train / validation / test phases
- Joint loss over mask, centerline, and junction heads
- Dice and IoU reporting for the mask head (main comparison target)
- Optional validation metrics for centerline and junction heads
- Best-checkpoint saving based on validation Dice of the mask head
- Final test evaluation using the best checkpoint

Notes
-----
- Uses get_dataloaders(..., return_topology=True).
- Dataset returns the fourth tensor as `junction`, while model output key is `junctions`.
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
# from dataset_awr_seg import get_dataloaders
from model_unet_multihead import build_unet_multihead


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"
OUTPUT_DIR = Path("runs_unet_multihead")

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.10
SEED = 42
NUM_WORKERS = 4
THRESHOLD = 0.5
PIN_MEMORY = True

# Loss weights for the three tasks
W_MASK = 1.0
W_CENTERLINE = 1.0
W_JUNCTION = 1.0


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



def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    masks: torch.Tensor,
    centers: torch.Tensor,
    junctions: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_mask = criterion(outputs["mask"], masks)
    loss_center = criterion(outputs["centerline"], centers)
    loss_junction = criterion(outputs["junctions"], junctions)

    total_loss = (
        W_MASK * loss_mask
        + W_CENTERLINE * loss_center
        + W_JUNCTION * loss_junction
    )

    return total_loss, {
        "loss_mask": loss_mask.item(),
        "loss_centerline": loss_center.item(),
        "loss_junction": loss_junction.item(),
    }


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss_sum = 0.0
    loss_mask_sum = 0.0
    loss_center_sum = 0.0
    loss_junction_sum = 0.0
    n_batches = 0

    for images, masks, centers, junctions in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)
        junctions = junctions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        total_loss, parts = compute_total_loss(outputs, masks, centers, junctions, criterion)
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        loss_mask_sum += parts["loss_mask"]
        loss_center_sum += parts["loss_centerline"]
        loss_junction_sum += parts["loss_junction"]
        n_batches += 1

    return {
        "loss": total_loss_sum / max(1, n_batches),
        "loss_mask": loss_mask_sum / max(1, n_batches),
        "loss_centerline": loss_center_sum / max(1, n_batches),
        "loss_junction": loss_junction_sum / max(1, n_batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    total_loss_sum = 0.0
    loss_mask_sum = 0.0
    loss_center_sum = 0.0
    loss_junction_sum = 0.0
    mask_dice_scores = []
    mask_iou_scores = []
    center_dice_scores = []
    junction_dice_scores = []
    n_batches = 0

    for images, masks, centers, junctions in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)
        junctions = junctions.to(device, non_blocking=True)

        outputs = model(images)
        total_loss, parts = compute_total_loss(outputs, masks, centers, junctions, criterion)

        total_loss_sum += total_loss.item()
        loss_mask_sum += parts["loss_mask"]
        loss_center_sum += parts["loss_centerline"]
        loss_junction_sum += parts["loss_junction"]

        mask_dice_scores.append(dice_coeff_from_logits(outputs["mask"], masks, threshold=threshold))
        mask_iou_scores.append(iou_from_logits(outputs["mask"], masks, threshold=threshold))
        center_dice_scores.append(dice_coeff_from_logits(outputs["centerline"], centers, threshold=threshold))
        junction_dice_scores.append(dice_coeff_from_logits(outputs["junctions"], junctions, threshold=threshold))
        n_batches += 1

    return {
        "loss": total_loss_sum / max(1, n_batches),
        "loss_mask": loss_mask_sum / max(1, n_batches),
        "loss_centerline": loss_center_sum / max(1, n_batches),
        "loss_junction": loss_junction_sum / max(1, n_batches),
        "mask_dice": float(np.mean(mask_dice_scores)) if mask_dice_scores else 0.0,
        "mask_iou": float(np.mean(mask_iou_scores)) if mask_iou_scores else 0.0,
        "centerline_dice": float(np.mean(center_dice_scores)) if center_dice_scores else 0.0,
        "junction_dice": float(np.mean(junction_dice_scores)) if junction_dice_scores else 0.0,
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
        return_topology=True,
    )

    print("Building model...")
    model = build_unet_multihead(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_dice = -float("inf")
    ckpt_path = OUTPUT_DIR / "best_model_unet_multihead.pth"
    history: list[Dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
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
            "train_loss": train_metrics["loss"],
            "train_loss_mask": train_metrics["loss_mask"],
            "train_loss_centerline": train_metrics["loss_centerline"],
            "train_loss_junction": train_metrics["loss_junction"],
            "val_loss": val_metrics["loss"],
            "val_loss_mask": val_metrics["loss_mask"],
            "val_loss_centerline": val_metrics["loss_centerline"],
            "val_loss_junction": val_metrics["loss_junction"],
            "val_mask_dice": val_metrics["mask_dice"],
            "val_mask_iou": val_metrics["mask_iou"],
            "val_centerline_dice": val_metrics["centerline_dice"],
            "val_junction_dice": val_metrics["junction_dice"],
            "epoch_seconds": elapsed,
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  lr                   = {current_lr:.6f}")
        print(f"  train_total_loss     = {train_metrics['loss']:.4f}")
        print(f"  train_mask_loss      = {train_metrics['loss_mask']:.4f}")
        print(f"  train_center_loss    = {train_metrics['loss_centerline']:.4f}")
        print(f"  train_junction_loss  = {train_metrics['loss_junction']:.4f}")
        print(f"  val_total_loss       = {val_metrics['loss']:.4f}")
        print(f"  val_mask_loss        = {val_metrics['loss_mask']:.4f}")
        print(f"  val_center_loss      = {val_metrics['loss_centerline']:.4f}")
        print(f"  val_junction_loss    = {val_metrics['loss_junction']:.4f}")
        print(f"  val_mask_dice        = {val_metrics['mask_dice']:.4f}")
        print(f"  val_mask_iou         = {val_metrics['mask_iou']:.4f}")
        print(f"  val_centerline_dice  = {val_metrics['centerline_dice']:.4f}")
        print(f"  val_junction_dice    = {val_metrics['junction_dice']:.4f}")
        print(f"  time                 = {elapsed:.1f}s")

        scheduler.step(val_metrics["mask_dice"])

        if val_metrics["mask_dice"] > best_val_dice:
            best_val_dice = val_metrics["mask_dice"]
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
                        "loss_weights": {
                            "mask": W_MASK,
                            "centerline": W_CENTERLINE,
                            "junction": W_JUNCTION,
                        },
                    },
                    "best_metric_name": "val_mask_dice",
                    "best_metric_value": best_val_dice,
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
    print(f"test_total_loss      = {test_metrics['loss']:.4f}")
    print(f"test_mask_loss       = {test_metrics['loss_mask']:.4f}")
    print(f"test_centerline_loss = {test_metrics['loss_centerline']:.4f}")
    print(f"test_junction_loss   = {test_metrics['loss_junction']:.4f}")
    print(f"test_mask_dice       = {test_metrics['mask_dice']:.4f}")
    print(f"test_mask_iou        = {test_metrics['mask_iou']:.4f}")
    print(f"test_centerline_dice = {test_metrics['centerline_dice']:.4f}")
    print(f"test_junction_dice   = {test_metrics['junction_dice']:.4f}")

    with (OUTPUT_DIR / "test_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("UNetMultiHead final test metrics\n")
        for key, value in test_metrics.items():
            f.write(f"{key}={value:.6f}\n")


if __name__ == "__main__":
    main()
