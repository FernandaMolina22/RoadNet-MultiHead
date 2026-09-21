from __future__ import annotations

"""
evaluate_connectivity_corrected_lcc_only.py

Fast evaluation script to recompute only the connectivity/fragmentation metrics
needed for the thesis after redefining the LCC ratio.

It does NOT retrain any model. It loads the saved checkpoints and runs inference
on the fixed test set.

Metrics computed
----------------
1. Mean connected components:
   - computed on the full predicted binary mask
   - averaged over all test patches

2. Mean Corrected LCC ratio:
   - the numerator is computed only from correctly predicted road pixels
     (true-positive mask = prediction AND ground truth)
   - the denominator remains the total predicted foreground pixels
   - this penalizes false positives because they remain in the denominator
   - if a patch has no predicted foreground pixels, the Corrected LCC ratio is 0.0

Formula
-------
Corrected LCC =
    pixels in largest connected component of (prediction AND ground truth)
    ---------------------------------------------------------------------
                   total predicted foreground pixels

This avoids rewarding models that produce large connected false-positive regions,
while still measuring how much of the full prediction corresponds to one dominant
correctly predicted connected road structure.
"""

from pathlib import Path
import csv
import json
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
import torch

from dataset_awr_seg_mosaic_split import get_dataloaders
from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead

from evaluation_utils import logits_to_binary, tensor_to_numpy_binary


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"

SEEDS = [42, 123, 2025]

OUTPUT_DIR = Path("evaluation_outputs_corrected_lcc_only")

BATCH_SIZE = 8
VAL_FRACTION = 0.10
NUM_WORKERS = 4
THRESHOLD = 0.5
CONNECTIVITY = 2  # 1 = 4-connectivity, 2 = 8-connectivity
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
# CONNECTIVITY HELPERS
# ============================================================

def ensure_2d_binary(mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) or (1,H,W) mask to a clean 2D binary uint8 array."""
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    return (mask > 0).astype(np.uint8)


def connected_components(mask: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    """Return labeled components and component count for a binary mask."""
    mask_2d = ensure_2d_binary(mask)

    if connectivity == 1:
        structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    elif connectivity == 2:
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    else:
        raise ValueError("connectivity must be 1 or 2")

    labeled, num_components = ndimage.label(mask_2d, structure=structure)
    return labeled, int(num_components)


def lcc_ratio(mask: np.ndarray, connectivity: int = 2) -> float:
    """
    Standard LCC ratio for a binary mask:
    largest component size / total foreground pixels.
    """
    mask_2d = ensure_2d_binary(mask)
    total_foreground = int(mask_2d.sum())

    if total_foreground == 0:
        return 0.0

    labeled, num_components = connected_components(mask_2d, connectivity=connectivity)

    if num_components == 0:
        return 0.0

    component_sizes = np.bincount(labeled.ravel())[1:]  # ignore background
    largest_size = int(component_sizes.max())
    return float(largest_size / total_foreground)


def largest_component_size(mask: np.ndarray, connectivity: int = 2) -> int:
    """
    Return the number of pixels in the largest connected component of a binary mask.
    If the mask has no foreground pixels, return 0.
    """
    mask_2d = ensure_2d_binary(mask)

    if int(mask_2d.sum()) == 0:
        return 0

    labeled, num_components = connected_components(mask_2d, connectivity=connectivity)

    if num_components == 0:
        return 0

    component_sizes = np.bincount(labeled.ravel())[1:]  # ignore background
    return int(component_sizes.max())


def corrected_lcc_ratio(pred: np.ndarray, gt: np.ndarray, connectivity: int = 2) -> float:
    """
    Corrected LCC ratio.

    The numerator is computed after removing false positives:
        TP mask = pred AND gt

    The denominator remains the full predicted foreground:
        total predicted foreground pixels = TP + FP

    Formula:
        Corrected LCC =
            largest connected component size in TP mask / total predicted foreground pixels

    Interpretation:
        How much of the model's full predicted road mask belongs to one dominant
        connected component that is actually correct?

    Edge cases:
        - If the model predicts no foreground pixels, return 0.0.
        - If the model predicts foreground but none overlaps with GT, return 0.0.
    """
    pred_2d = ensure_2d_binary(pred).astype(bool)
    gt_2d = ensure_2d_binary(gt).astype(bool)

    total_predicted_foreground = int(pred_2d.sum())
    if total_predicted_foreground == 0:
        return 0.0

    tp_mask = np.logical_and(pred_2d, gt_2d).astype(np.uint8)
    largest_tp_component = largest_component_size(tp_mask, connectivity=connectivity)

    return float(largest_tp_component / total_predicted_foreground)


class RunningConnectivityCorrectedLCC:
    """Accumulate connectivity metrics patch by patch."""

    def __init__(self, connectivity: int = 2) -> None:
        self.connectivity = connectivity

        self.num_components_values: List[float] = []
        self.old_lcc_values: List[float] = []
        self.corrected_lcc_values: List[float] = []

        self.n_patches = 0
        self.n_patches_no_predicted_foreground = 0
        self.n_patches_predicted_foreground_zero_tp = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred_2d = ensure_2d_binary(pred)
        gt_2d = ensure_2d_binary(gt)

        _, num_components = connected_components(pred_2d, connectivity=self.connectivity)
        self.num_components_values.append(float(num_components))

        # Old metric is kept only for comparison/debugging.
        self.old_lcc_values.append(lcc_ratio(pred_2d, connectivity=self.connectivity))

        corrected_value = corrected_lcc_ratio(pred_2d, gt_2d, connectivity=self.connectivity)
        self.corrected_lcc_values.append(float(corrected_value))

        self.n_patches += 1

        pred_bool = pred_2d.astype(bool)
        gt_bool = gt_2d.astype(bool)

        if int(pred_bool.sum()) == 0:
            self.n_patches_no_predicted_foreground += 1
        elif int(np.logical_and(pred_bool, gt_bool).sum()) == 0:
            self.n_patches_predicted_foreground_zero_tp += 1

    def compute(self) -> Dict[str, float]:
        return {
            "num_components_mean": float(np.mean(self.num_components_values)) if self.num_components_values else 0.0,
            "old_lcc_ratio_mean": float(np.mean(self.old_lcc_values)) if self.old_lcc_values else 0.0,
            "corrected_lcc_ratio_mean": float(np.mean(self.corrected_lcc_values)) if self.corrected_lcc_values else 0.0,
            "n_patches": int(self.n_patches),
            "n_patches_no_predicted_foreground": int(self.n_patches_no_predicted_foreground),
            "n_patches_predicted_foreground_zero_tp": int(self.n_patches_predicted_foreground_zero_tp),
        }


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
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_checkpoint(
    *,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    model_type: str,
) -> Dict[str, float]:
    """
    Evaluate one checkpoint.

    model_type:
      - "single"
      - "multihead"
    """
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=seed,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        return_topology=True,
    )

    if model_type == "single":
        model = load_unet_single(device, checkpoint_path)
    elif model_type == "multihead":
        model = load_unet_multihead(device, checkpoint_path)
    else:
        raise ValueError("model_type must be 'single' or 'multihead'")

    pred_meter = RunningConnectivityCorrectedLCC(connectivity=CONNECTIVITY)
    gt_meter = RunningConnectivityCorrectedLCC(connectivity=CONNECTIVITY)

    for images, masks, centers, junctions in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if model_type == "single":
            logits = model(images)
        else:
            outputs = model(images)
            logits = outputs["mask"]

        pred_mask = logits_to_binary(logits, threshold=THRESHOLD)

        pred_np = tensor_to_numpy_binary(pred_mask)
        gt_np = tensor_to_numpy_binary(masks)

        for i in range(pred_np.shape[0]):
            pred_meter.update(pred_np[i], gt_np[i])

            # Ground-truth reference: use GT as both prediction and GT.
            # In this case, corrected LCC equals the standard GT LCC.
            gt_meter.update(gt_np[i], gt_np[i])

    return {
        "prediction": pred_meter.compute(),
        "ground_truth_reference": gt_meter.compute(),
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def aggregate_results(results_by_seed: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    model_names = sorted({
        model_name
        for seed_results in results_by_seed.values()
        for model_name in seed_results.keys()
        if not model_name.startswith("_")
    })

    summary: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        num_components = []
        old_lcc = []
        corrected_lcc = []

        for seed_str, seed_results in results_by_seed.items():
            if model_name not in seed_results:
                continue

            pred_metrics = seed_results[model_name]["prediction"]
            num_components.append(pred_metrics["num_components_mean"])
            old_lcc.append(pred_metrics["old_lcc_ratio_mean"])
            corrected_lcc.append(pred_metrics["corrected_lcc_ratio_mean"])

        nc_mean, nc_std = mean_std(num_components)
        old_mean, old_std = mean_std(old_lcc)
        corrected_mean, corrected_std = mean_std(corrected_lcc)

        summary[model_name] = {
            "num_components_mean": nc_mean,
            "num_components_std": nc_std,
            "old_lcc_ratio_mean": old_mean,
            "old_lcc_ratio_std": old_std,
            "corrected_lcc_ratio_mean": corrected_mean,
            "corrected_lcc_ratio_std": corrected_std,
        }

    return summary


def save_summary_csv(summary: Dict[str, Dict[str, float]], path: Path) -> None:
    rows = []
    for model_name, metrics in summary.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    fieldnames = [
        "model",
        "num_components_mean",
        "num_components_std",
        "old_lcc_ratio_mean",
        "old_lcc_ratio_std",
        "corrected_lcc_ratio_mean",
        "corrected_lcc_ratio_std",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    print("\n===== SUMMARY ACROSS SEEDS =====")
    print("Use Corrected LCC for the thesis. old_lcc_ratio is kept only for comparison/debugging.\n")
    print(f"{'Model':28s} | {'Components':>18s} | {'Corrected LCC':>18s} | {'Old LCC':>18s}")
    print("-" * 92)

    for model_name, m in summary.items():
        comp = f"{m['num_components_mean']:.3f} ± {m['num_components_std']:.3f}"
        corrected_lcc = f"{m['corrected_lcc_ratio_mean']:.3f} ± {m['corrected_lcc_ratio_std']:.3f}"
        old_lcc = f"{m['old_lcc_ratio_mean']:.3f} ± {m['old_lcc_ratio_std']:.3f}"
        print(f"{model_name:28s} | {comp:>18s} | {corrected_lcc:>18s} | {old_lcc:>18s}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Connectivity: {CONNECTIVITY} (2 means 8-connectivity)")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {
        "metadata": {
            "seeds": SEEDS,
            "threshold": THRESHOLD,
            "connectivity": CONNECTIVITY,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
            "corrected_lcc_definition": (
                "For each patch, compute TP mask = prediction AND ground truth. "
                "Corrected LCC is the largest connected component in the TP mask "
                "divided by total predicted foreground pixels. If the model predicts "
                "no foreground pixels, Corrected LCC is 0.0. This keeps false positives "
                "in the denominator, so they are penalized, but excludes false positives "
                "from the numerator, so they cannot increase the largest connected component."
            ),
        },
        "results_by_seed": {},
    }

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"Evaluating seed {seed}")
        print("=" * 60)

        seed_key = str(seed)
        results["results_by_seed"][seed_key] = {}

        # UNetSingle
        single_ckpt = SINGLE_CKPTS[seed]
        print("\nEvaluating UNetSingle")
        print(f"Checkpoint: {single_ckpt}")

        if single_ckpt.exists():
            results["results_by_seed"][seed_key]["unet_single"] = evaluate_checkpoint(
                device=device,
                checkpoint_path=single_ckpt,
                seed=seed,
                model_type="single",
            )
        else:
            print(f"WARNING: checkpoint not found, skipping: {single_ckpt}")

        # MultiHead equal
        equal_ckpt = MULTIHEAD_EQUAL_CKPTS[seed]
        print("\nEvaluating UNetMultiHead equal")
        print(f"Checkpoint: {equal_ckpt}")

        if equal_ckpt.exists():
            results["results_by_seed"][seed_key]["multihead_equal"] = evaluate_checkpoint(
                device=device,
                checkpoint_path=equal_ckpt,
                seed=seed,
                model_type="multihead",
            )
        else:
            print(f"WARNING: checkpoint not found, skipping: {equal_ckpt}")

        # Weighted MultiHead experiments
        for experiment_name in WEIGHTED_EXPERIMENTS:
            ckpt = weighted_multihead_ckpt(seed, experiment_name)
            result_key = f"multihead_{experiment_name}"

            print(f"\nEvaluating UNetMultiHead {experiment_name}")
            print(f"Checkpoint: {ckpt}")

            if ckpt.exists():
                results["results_by_seed"][seed_key][result_key] = evaluate_checkpoint(
                    device=device,
                    checkpoint_path=ckpt,
                    seed=seed,
                    model_type="multihead",
                )
            else:
                print(f"WARNING: checkpoint not found, skipping: {ckpt}")

    summary = aggregate_results(results["results_by_seed"])
    results["summary_across_seeds"] = summary

    # Optional: add a single ground-truth reference from the first available model/seed.
    # It should be the same across models for the same test set.
    for seed_results in results["results_by_seed"].values():
        for model_result in seed_results.values():
            results["ground_truth_reference"] = model_result["ground_truth_reference"]
            break
        if "ground_truth_reference" in results:
            break

    json_path = OUTPUT_DIR / "corrected_lcc_connectivity_results.json"
    csv_path = OUTPUT_DIR / "corrected_lcc_connectivity_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_summary_csv(summary, csv_path)
    print_summary_table(summary)

    print("\n===== CORRECTED LCC CONNECTIVITY EVALUATION COMPLETE =====")
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
