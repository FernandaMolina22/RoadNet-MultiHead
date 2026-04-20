from __future__ import annotations

"""
skeleton_utils.py

This module provides utilities for evaluating the STRUCTURAL quality of road
segmentation predictions.

While standard metrics (Dice, IoU) measure pixel-wise overlap, they do not
capture the topology of the road network (e.g., connectivity, continuity,
and correctness of intersections).

To address this limitation, I introduce a skeleton-based evaluation:

1. The predicted binary road mask is skeletonized into a 1-pixel-wide structure
   representing the road centerlines.

2. This skeleton is compared against a ground-truth centerline map, which is
   precomputed from the original road masks using morphological skeletonization.

This allows us to evaluate how well the predicted road network preserves the
underlying structure, independent of road thickness.

This module is used in:
- evaluate_models.py → to compute structural metrics for all methods

Key idea:
    skeleton(predicted mask) vs ground-truth centerline
"""

from typing import Dict
import numpy as np
from skimage.morphology import skeletonize

from evaluation_utils import compute_mask_metrics_from_arrays, RunningMaskMetrics


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_2d_binary(mask: np.ndarray) -> np.ndarray:
    """
    Ensure the input is a 2D binary array with values {0,1}.

    This function standardizes input masks coming from different sources,
    such as:
        - (H, W)
        - (1, H, W)

    It also guarantees binary values (0 or 1), which is required for
    skeletonization and metric computation.

    Parameters
    ----------
    mask : np.ndarray
        Input mask in any supported shape

    Returns
    -------
    np.ndarray
        2D binary mask of shape (H, W)
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    return (mask > 0).astype(np.uint8)


# ============================================================
# SKELETONIZATION
# ============================================================

def skeletonize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary mask into a one-pixel-wide centerline.

    This operation reduces thick road segments into a minimal representation
    that preserves their topology (connectivity and structure).

    Why this is important:
        - Removes dependency on road thickness
        - Focuses evaluation on connectivity and geometry
        - Enables fair comparison between different methods

    Parameters
    ----------
    mask : np.ndarray
        Binary road mask (H, W) or (1, H, W)

    Returns
    -------
    np.ndarray
        Skeletonized binary mask (H, W), values {0,1}
    """
    mask_2d = ensure_2d_binary(mask)

    # skimage expects boolean input
    skeleton = skeletonize(mask_2d.astype(bool))

    return skeleton.astype(np.uint8)


# ============================================================
# SINGLE-PASS METRICS
# ============================================================

def compute_skeleton_metrics_against_centerline(
    pred_mask: np.ndarray,
    gt_centerline: np.ndarray,
) -> Dict[str, float]:
    """
    Compute structural metrics for a single prediction.

    Pipeline:
        predicted mask → skeletonize → predicted centerline
        compare with → ground-truth centerline

    Important:
    The ground-truth centerline is NOT recomputed here.
    It is pre-generated (centerline_phase1) from the original masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary road mask
    gt_centerline : np.ndarray
        Ground-truth centerline (precomputed)

    Returns
    -------
    Dict[str, float]
        Standard metrics (precision, recall, Dice, IoU)
        computed on skeletons
    """
    pred_skel = skeletonize_binary_mask(pred_mask)
    gt_center = ensure_2d_binary(gt_centerline)

    return compute_mask_metrics_from_arrays(pred_skel, gt_center)


# ============================================================
# RUNNING (GLOBAL) METRICS
# ============================================================

class RunningSkeletonMetrics:
    """
    Accumulate structural metrics over an entire dataset.

    Instead of computing metrics per patch and averaging, this class:
        - accumulates TP / FP / FN / TN globally
        - computes final metrics at the end

    This is consistent with the mask-level evaluation and ensures
    a fair and robust comparison.

    Evaluation performed:
        skeleton(predicted final mask) vs ground-truth centerline

    Usage:
        meter = RunningSkeletonMetrics()

        for each batch:
            meter.update(pred_mask, gt_centerline)

        final_metrics = meter.compute()
    """

    def __init__(self) -> None:
        self.meter = RunningMaskMetrics()

    def update(self, pred_mask: np.ndarray, gt_centerline: np.ndarray) -> None:
        """
        Update global structural metrics for one sample.

        Steps:
        1. Skeletonize predicted mask
        2. Compare with GT centerline
        3. Accumulate confusion counts

        Parameters
        ----------
        pred_mask : np.ndarray
            Predicted binary mask
        gt_centerline : np.ndarray
            Ground-truth centerline
        """
        pred_skel = skeletonize_binary_mask(pred_mask)
        gt_center = ensure_2d_binary(gt_centerline)

        self.meter.update(pred_skel, gt_center)

    def compute(self) -> Dict[str, float]:
        """
        Compute final structural metrics over the dataset.
        """
        return self.meter.compute()