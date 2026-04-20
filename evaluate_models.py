from __future__ import annotations

"""
Evaluation script for road segmentation models.

This script performs a FAIR and GLOBAL evaluation of:

- UNetSingle (baseline)
- UNetMultiHead (topology-aware model)

It computes:
1. Standard mask metrics on the TEST set
   - Precision
   - Recall
   - Dice / F1
   - IoU

2. Structural skeleton-based metrics
   - skeleton(predicted final mask) vs ground-truth centerline

3. Extra topology-head metrics for UNetMultiHead
   - predicted centerline head vs ground-truth centerline

4. Connectivity / fragmentation metrics
   - number of connected components
   - largest connected component ratio (LCC ratio)

IMPORTANT:
- Metrics are computed globally over ALL pixels for mask/skeleton metrics
- Connectivity metrics are computed patch by patch and then averaged
- Predictions are binarized using a fixed threshold (default = 0.5)
- Test evaluation uses mosaic-level splitting (no data leakage)
"""

from pathlib import Path
import json
import torch

from dataset_awr_seg_mosaic_split import get_dataloaders
from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead

from evaluation_utils import (
    RunningMaskMetrics,
    logits_to_binary,
    tensor_to_numpy_binary,
)
from skeleton_utils import RunningSkeletonMetrics
from connectivity_utils import RunningConnectivityMetrics


# ============================================================
# CONFIGURATION (⚠️ EDIT HERE WHEN NEEDED)
# ============================================================

# ----------------------------------------------------------------
# 📂 DATASET PATH
# ----------------------------------------------------------------
# 👉 CHANGE THIS if you move your dataset
# Must point to the ROOT folder of Amazon Wild Roads dataset
DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"


# ----------------------------------------------------------------
# 💾 MODEL CHECKPOINTS
# ----------------------------------------------------------------
# 👉 CHANGE THESE if:
# - you retrain models
# - you use different experiment folders
# - you want to evaluate other checkpoints
UNET_SINGLE_CKPT = Path("runs_unet_single/best_model_unet_single.pth")
UNET_MULTIHEAD_CKPT = Path("runs_unet_multihead/best_model_unet_multihead.pth")


# ----------------------------------------------------------------
# 📊 OUTPUT DIRECTORY
# ----------------------------------------------------------------
# Where evaluation results (JSON) will be saved
OUTPUT_DIR = Path("evaluation_outputs")


# ----------------------------------------------------------------
# ⚙️ EVALUATION PARAMETERS
# ----------------------------------------------------------------
BATCH_SIZE = 8
VAL_FRACTION = 0.10   # not used for reporting, but required by get_dataloaders
SEED = 42
NUM_WORKERS = 4

# 👉 CRITICAL: threshold for converting probabilities → binary mask
THRESHOLD = 0.5

# Connectivity convention:
# connectivity=2 means 8-connectivity in a 2D raster, which is appropriate
# for road extraction because diagonally touching road pixels are treated as connected.
CONNECTIVITY = 2

PIN_MEMORY = True


# ============================================================
# MODEL LOADING
# ============================================================

def load_unet_single(device: torch.device):
    """
    Load trained UNetSingle model from checkpoint.

    Steps:
    1. Build model architecture
    2. Load trained weights
    3. Set to evaluation mode

    Returns
    -------
    torch.nn.Module
        Trained UNetSingle ready for inference
    """
    model = build_unet_single(device)

    checkpoint = torch.load(UNET_SINGLE_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def load_unet_multihead(device: torch.device):
    """
    Load trained UNetMultiHead model from checkpoint.

    Note
    ----
    This model predicts three outputs:
    - mask
    - centerline
    - junctions

    In this evaluation script:
    - the mask output is used for common mask metrics
    - the mask output is also skeletonized for common structural metrics
    - the centerline output is evaluated separately as an extra analysis
    """
    model = build_unet_multihead(device)

    checkpoint = torch.load(UNET_MULTIHEAD_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


# ============================================================
# EVALUATION: UNetSingle
# ============================================================

@torch.no_grad()
def evaluate_unet_single(device: torch.device):
    """
    Evaluate UNetSingle on the TEST set.

    This function computes:
    1. mask-level metrics:
       predicted mask vs ground-truth mask

    2. skeleton-level metrics:
       skeleton(predicted mask) vs ground-truth centerline

    3. connectivity / fragmentation metrics:
       computed on the predicted binary mask and on the ground-truth binary mask

    Note
    ----
    Even though UNetSingle does not predict topology outputs, we still load
    the centerline ground-truth labels from the dataset because they are needed
    for the common structural comparison.
    """

    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        return_topology=True,   # needed to access GT centerlines
    )

    model = load_unet_single(device)

    mask_meter = RunningMaskMetrics()
    skeleton_meter = RunningSkeletonMetrics()

    pred_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)
    gt_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)

    for images, masks, centers, junctions in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass → raw logits for road mask
        logits = model(images)

        # Convert logits to binary predictions
        pred_mask = logits_to_binary(logits, threshold=THRESHOLD)

        # Convert tensors to NumPy binary arrays
        pred_mask_np = tensor_to_numpy_binary(pred_mask)   # shape: (B,1,H,W)
        gt_mask_np = tensor_to_numpy_binary(masks)         # shape: (B,1,H,W)
        gt_center_np = tensor_to_numpy_binary(centers)     # shape: (B,1,H,W)

        # 1) Global mask metrics
        mask_meter.update(pred_mask_np, gt_mask_np)

        # 2) Global structural metrics:
        #    skeleton(predicted mask) vs GT centerline
        #
        # 3) Connectivity metrics:
        #    computed patch by patch and averaged later
        for i in range(pred_mask_np.shape[0]):
            skeleton_meter.update(pred_mask_np[i], gt_center_np[i])
            pred_connectivity_meter.update(pred_mask_np[i])
            gt_connectivity_meter.update(gt_mask_np[i])

    return {
        "mask": mask_meter.compute(),
        "skeleton": skeleton_meter.compute(),
        "connectivity": {
            "prediction": pred_connectivity_meter.compute(),
            "ground_truth": gt_connectivity_meter.compute(),
        },
    }


# ============================================================
# EVALUATION: UNetMultiHead
# ============================================================

@torch.no_grad()
def evaluate_unet_multihead(device: torch.device):
    """
    Evaluate UNetMultiHead on the TEST set.

    This function computes:
    1. mask-level metrics:
       predicted mask vs ground-truth mask

    2. skeleton-level metrics:
       skeleton(predicted final mask) vs ground-truth centerline

    3. centerline-head metrics:
       predicted centerline head vs ground-truth centerline

    4. connectivity / fragmentation metrics:
       computed on the predicted binary mask and on the ground-truth binary mask

    This gives both:
    - a fair common comparison with the baseline
    - an extra model-specific analysis of the topology branch
    """

    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        return_topology=True,
    )

    model = load_unet_multihead(device)

    mask_meter = RunningMaskMetrics()
    skeleton_meter = RunningSkeletonMetrics()
    centerline_meter = RunningMaskMetrics()

    pred_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)
    gt_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)

    for images, masks, centers, junctions in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)

        # Extract relevant heads
        logits_mask = outputs["mask"]
        logits_center = outputs["centerline"]

        # Convert logits to binary predictions
        pred_mask = logits_to_binary(logits_mask, threshold=THRESHOLD)
        pred_center = logits_to_binary(logits_center, threshold=THRESHOLD)

        # Convert tensors to NumPy binary arrays
        pred_mask_np = tensor_to_numpy_binary(pred_mask)
        gt_mask_np = tensor_to_numpy_binary(masks)

        pred_center_np = tensor_to_numpy_binary(pred_center)
        gt_center_np = tensor_to_numpy_binary(centers)

        # 1) Global common mask metrics
        mask_meter.update(pred_mask_np, gt_mask_np)

        # 2) Extra centerline-head metrics
        centerline_meter.update(pred_center_np, gt_center_np)

        # 3) Global common structural metrics:
        #    skeleton(predicted final mask) vs GT centerline
        #
        # 4) Connectivity metrics:
        #    computed patch by patch and averaged later
        for i in range(pred_mask_np.shape[0]):
            skeleton_meter.update(pred_mask_np[i], gt_center_np[i])
            pred_connectivity_meter.update(pred_mask_np[i])
            gt_connectivity_meter.update(gt_mask_np[i])

    return {
        "mask": mask_meter.compute(),
        "skeleton": skeleton_meter.compute(),
        "centerline_head": centerline_meter.compute(),
        "connectivity": {
            "prediction": pred_connectivity_meter.compute(),
            "ground_truth": gt_connectivity_meter.compute(),
        },
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main evaluation entry point.

    Steps
    -----
    1. Evaluate UNetSingle
    2. Evaluate UNetMultiHead
    3. Print all results
    4. Save results to JSON

    Output structure
    ----------------
    - unet_single:
        - mask
        - skeleton
        - connectivity

    - unet_multihead:
        - mask
        - skeleton
        - centerline_head
        - connectivity
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------
    print("\nEvaluating UNetSingle...")
    single_metrics = evaluate_unet_single(device)

    print("\nEvaluating UNetMultiHead...")
    multi_metrics = evaluate_unet_multihead(device)

    # ------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------
    results = {
        "unet_single": single_metrics,
        "unet_multihead": multi_metrics,
    }

    # ------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------
    print("\n===== FINAL EVALUATION RESULTS =====")
    print(json.dumps(results, indent=2))

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    out_path = OUTPUT_DIR / "full_evaluation_results.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()