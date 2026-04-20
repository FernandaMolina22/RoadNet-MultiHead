# ============================================
# train_topology_unet.py
#
# Mask-only training for the multi-head U-Net.
# - Uses the multi-head architecture (mask + centerline + junctions)
# - BUT only the "mask" head is supervised for now.
# - This is just to make sure:
#     * Dataset + DataLoader work
#     * Forward pass works (no shape mismatch)
#     * Loss goes down
#     * Dice improves a bit
# ============================================

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import time 

from dataset_awr_seg import get_dataloaders
from model_unet_multihead import build_unet_multihead


# ------------------------------
# 1. Simple Dice metric (for mask evaluation)
# ------------------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute the Dice coefficient for binary segmentation.

    Arguments
    ---------
    pred : torch.Tensor
        Predicted mask after thresholding. Shape: (B, 1, H, W), values {0,1}.
    target : torch.Tensor
        Ground truth mask. Shape: (B, 1, H, W), values {0,1}.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean Dice coefficient over the batch, as a Python float.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    # Sum over spatial dimensions (H, W) and channel dim (1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)

    # Average over batch dimension (B) and convert to float
    return dice.mean().item()


# ------------------------------
# 2. Train / validation epoch helpers
# ------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_loss: nn.Module,
) -> float:
    """
    Run ONE training epoch using ONLY the mask head and mask GT.

    For Step 0:
    - We ignore the centerline and junction heads.
    - Objective is just BCEWithLogitsLoss on the mask output.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        # Move tensors to GPU (or CPU) for faster computation
        images = images.to(device, non_blocking=True)  # (B, 3, H, W)
        masks  = masks.to(device, non_blocking=True)   # (B, 1, H, W)

        optimizer.zero_grad()

        # Forward pass through multi-head model
        outputs: Dict[str, torch.Tensor] = model(images)

        # We only use the "mask" head for now
        logits_mask = outputs["mask"]  # (B, 1, H, W), raw logits

        # BCEWithLogitsLoss combines Sigmoid + BCE in a stable way
        loss = bce_loss(logits_mask, masks)

        # Backprop + parameter update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    bce_loss: nn.Module,
):
    """
    Run ONE validation epoch.

    We compute:
    - Validation loss (BCE on mask head)
    - Dice coefficient for the mask head

    Returns
    -------
    (val_loss, val_dice) : (float, float)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_dice = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        outputs: Dict[str, torch.Tensor] = model(images)
        logits_mask = outputs["mask"]

        loss = bce_loss(logits_mask, masks)

        total_loss += loss.item()
        n_batches  += 1

        # --- Dice computation ---
        # 1) Convert logits → probabilities with Sigmoid
        probs_mask = torch.sigmoid(logits_mask)
        # 2) Threshold to get binary prediction
        preds_mask = (probs_mask > 0.5).float()
        # 3) Compute Dice between prediction and GT
        dice = dice_coeff(preds_mask, masks)
        all_dice.append(dice)

    mean_loss = total_loss / max(1, n_batches)
    mean_dice = sum(all_dice) / max(1, len(all_dice))

    return mean_loss, mean_dice


# ------------------------------
# 3. Main training function
# ------------------------------
def main():
    # ====== CONFIGURATION ======
    # IMPORTANT: set this to your actual AWR root folder.
    # It must contain subfolders like: image_png/ and mask/
    DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"


    # Where to store logs and best checkpoint
    OUTPUT_DIR = Path("runs_topology_unet_mask_only")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Basic hyperparameters (we keep it simple for Step 0)
    num_epochs   = 5     # a few epochs just to test pipeline
    batch_size   = 8
    lr           = 1e-4  # learning rate
    val_fraction = 0.1   # 10% of data used for validation

    # Choose device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== DATA LOADERS ======
    # Reuse your existing function from dataset_awr_seg.py
    print("Building DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=batch_size,
        val_fraction=val_fraction,
        num_workers=4,     # adjust if needed
        pin_memory=True,   # good practice when using GPU
    )

    # ====== MODEL, LOSS, OPTIMIZER, SCHEDULER ======
    print("Building model...")
    # This already moves the model to 'device' internally (as we defined it)
    model = build_unet_multihead(device)

    # Binary Cross-Entropy with logits (Sigmoid inside)
    bce_loss = nn.BCEWithLogitsLoss()

    # Adam optimizer on all model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler that reduces LR when val_loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,   # multiply LR by 0.5 when triggered
        patience=3,   # wait 3 epochs without improvement
        verbose=True,
    )

    # Track best model according to validation loss
    best_val_loss = float("inf")
    ckpt_path = OUTPUT_DIR / "best_model_mask_only.pth"

    # ====== TRAINING LOOP ======
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        start_time = time.time()  # Start Timer
        # --- Training step ---
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            bce_loss=bce_loss,
        )

        # --- Validation step ---
        val_loss, val_dice = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            bce_loss=bce_loss,
        )

        # Tell scheduler about the new validation loss
        scheduler.step(val_loss)

        # Compute elapsed time
        epoch_time = time.time() - start_time #stop timer

        # Print logs
        print(f"  train_loss      = {train_loss:.4f}")
        print(f"  val_loss        = {val_loss:.4f}")
        print(f"  val_dice(mask)  = {val_dice:.4f}")
        print(f"  epoch_time      = {epoch_time:.1f} seconds")
        # --- Save best model so far (by validation loss) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  >> New best model saved to {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
