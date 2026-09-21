#evaluate_models.py
from __future__ import annotations

"""
Final evaluation script for road segmentation models.

This script performs a FAIR and GLOBAL evaluation of:

1. UNetSingle baseline
2. UNetMultiHead equal-weighting model
3. UNetMultiHead loss-weighting experiments:
   - aux05
   - aux025
   - center_focus
   - junction_focus

For each model/configuration, it computes:
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
- Metrics are computed globally over ALL pixels for mask/skeleton metrics.
- Connectivity metrics are computed patch by patch and then averaged.
- Predictions are binarized using a fixed threshold.
- Test evaluation uses mosaic-level splitting.
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
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"

SEEDS = [42, 123, 2025]

OUTPUT_DIR = Path("evaluation_outputs_final_comparison")

BATCH_SIZE = 8
VAL_FRACTION = 0.10
NUM_WORKERS = 4
THRESHOLD = 0.5
CONNECTIVITY = 2
PIN_MEMORY = True


# ============================================================
# CHECKPOINT PATHS
# ============================================================

SINGLE_CKPTS = {
    42: Path("runs_unet_single/best_model_unet_single.pth"),
    123: Path("runs_unet_single_seed123/best_model_unet_single.pth"),
    2025: Path("runs_unet_single_seed2025/best_model_unet_single.pth"),
}

MULTIHEAD_EQUAL_CKPTS = {
    42: Path("runs_unet_multihead/best_model_unet_multihead.pth"),
    123: Path("runs_unet_multihead_seed123/best_model_unet_multihead.pth"),
    2025: Path("runs_unet_multihead_seed2025/best_model_unet_multihead.pth"),
}

WEIGHTED_EXPERIMENTS = [
    "aux05",
    "aux025",
    "center_focus",
    "junction_focus",
]


def weighted_multihead_ckpt(seed: int, experiment_name: str) -> Path:
    return Path(
        f"runs_unet_multihead_seed{seed}_{experiment_name}/best_model_unet_multihead.pth"
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_unet_single(device: torch.device, checkpoint_path: Path):
    model = build_unet_single(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def load_unet_multihead(device: torch.device, checkpoint_path: Path):
    model = build_unet_multihead(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


# ============================================================
# EVALUATION: UNetSingle
# ============================================================

@torch.no_grad()
def evaluate_unet_single(
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
):
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=seed,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        return_topology=True,
    )

    model = load_unet_single(device, checkpoint_path)

    mask_meter = RunningMaskMetrics()
    skeleton_meter = RunningSkeletonMetrics()

    pred_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)
    gt_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)

    for images, masks, centers, junctions in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        pred_mask = logits_to_binary(logits, threshold=THRESHOLD)

        pred_mask_np = tensor_to_numpy_binary(pred_mask)
        gt_mask_np = tensor_to_numpy_binary(masks)
        gt_center_np = tensor_to_numpy_binary(centers)

        mask_meter.update(pred_mask_np, gt_mask_np)

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
def evaluate_unet_multihead(
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
):
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=seed,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        return_topology=True,
    )

    model = load_unet_multihead(device, checkpoint_path)

    mask_meter = RunningMaskMetrics()
    skeleton_meter = RunningSkeletonMetrics()
    centerline_meter = RunningMaskMetrics()

    pred_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)
    gt_connectivity_meter = RunningConnectivityMetrics(connectivity=CONNECTIVITY)

    for images, masks, centers, junctions in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)

        outputs = model(images)

        logits_mask = outputs["mask"]
        logits_center = outputs["centerline"]

        pred_mask = logits_to_binary(logits_mask, threshold=THRESHOLD)
        pred_center = logits_to_binary(logits_center, threshold=THRESHOLD)

        pred_mask_np = tensor_to_numpy_binary(pred_mask)
        gt_mask_np = tensor_to_numpy_binary(masks)

        pred_center_np = tensor_to_numpy_binary(pred_center)
        gt_center_np = tensor_to_numpy_binary(centers)

        mask_meter.update(pred_mask_np, gt_mask_np)
        centerline_meter.update(pred_center_np, gt_center_np)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {
        "metadata": {
            "seeds": SEEDS,
            "threshold": THRESHOLD,
            "connectivity": CONNECTIVITY,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
        },
        "results_by_seed": {},
    }

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"Evaluating seed {seed}")
        print("=" * 60)

        results["results_by_seed"][str(seed)] = {}

        # --------------------------------------------------------
        # UNetSingle baseline
        # --------------------------------------------------------
        single_ckpt = SINGLE_CKPTS[seed]

        print("\nEvaluating UNetSingle")
        print(f"Checkpoint: {single_ckpt}")

        if single_ckpt.exists():
            results["results_by_seed"][str(seed)]["unet_single"] = evaluate_unet_single(
                device=device,
                checkpoint_path=single_ckpt,
                seed=seed,
            )
        else:
            print(f"WARNING: checkpoint not found, skipping: {single_ckpt}")

        # --------------------------------------------------------
        # UNetMultiHead equal weighting
        # --------------------------------------------------------
        equal_ckpt = MULTIHEAD_EQUAL_CKPTS[seed]

        print("\nEvaluating UNetMultiHead equal")
        print(f"Checkpoint: {equal_ckpt}")

        if equal_ckpt.exists():
            results["results_by_seed"][str(seed)]["multihead_equal"] = evaluate_unet_multihead(
                device=device,
                checkpoint_path=equal_ckpt,
                seed=seed,
            )
        else:
            print(f"WARNING: checkpoint not found, skipping: {equal_ckpt}")

        # --------------------------------------------------------
        # UNetMultiHead weighted experiments
        # --------------------------------------------------------
        for experiment_name in WEIGHTED_EXPERIMENTS:
            ckpt = weighted_multihead_ckpt(seed, experiment_name)
            result_key = f"multihead_{experiment_name}"

            print(f"\nEvaluating UNetMultiHead {experiment_name}")
            print(f"Checkpoint: {ckpt}")

            if ckpt.exists():
                results["results_by_seed"][str(seed)][result_key] = evaluate_unet_multihead(
                    device=device,
                    checkpoint_path=ckpt,
                    seed=seed,
                )
            else:
                print(f"WARNING: checkpoint not found, skipping: {ckpt}")

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    out_path = OUTPUT_DIR / "full_evaluation_results.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL EVALUATION COMPLETE =====")
    print(f"Saved results to: {out_path}")

    # Optional: print full JSON.
    # This can be very long, so keep it commented unless needed.
    # print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()