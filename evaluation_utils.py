from __future__ import annotations

from typing import Dict
import numpy as np
import torch


# ============================================================
# BASIC CONVERSIONS
# ============================================================
# The models do NOT output binary masks directly.
# They output raw logits, i.e. unconstrained real values in (-inf, +inf).
#
# Example:
#   negative logit -> more likely background
#   positive logit -> more likely road
#
# In order to compute segmentation metrics such as Precision, Recall,
# Dice, and IoU, we first need to transform logits into binary masks.
#
# The full conversion pipeline is:
#   logits -> sigmoid -> probabilities -> threshold -> binary mask
#
# This is important both practically and scientifically, because it defines
# the exact evaluation protocol used in the thesis.


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits to probabilities using the sigmoid function.

    Why this is needed
    ------------------
    BCEWithLogitsLoss expects raw logits during training. Therefore, the model
    outputs are not probabilities and are not binary values either.
    To interpret the output as a confidence score per pixel, we apply sigmoid.

    Mathematical meaning
    --------------------
    Sigmoid maps values from (-inf, +inf) to (0, 1), so each pixel can be
    interpreted as the predicted probability of belonging to the road class.

    Example
    -------
    logit = -3.0  -> probability ~ 0.05
    logit =  0.0  -> probability = 0.50
    logit =  2.0  -> probability ~ 0.88

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape (B, 1, H, W), containing raw model outputs.

    Returns
    -------
    torch.Tensor
        Tensor of probabilities in [0, 1], same shape as input.
    """
    return torch.sigmoid(logits)


def probs_to_binary(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert probabilities into binary predictions.

    Why this is needed
    ------------------
    Metrics such as connected components, skeletonization, Dice on binary masks,
    and IoU comparison all require a hard decision:
        road / not road
    rather than a soft confidence value.

    Evaluation choice
    -----------------
    A fixed threshold of 0.5 is used to convert probabilities into the final
    binary segmentation mask. This keeps the evaluation consistent across
    methods.

    Rule
    ----
    If probability >= threshold -> predicted as road (1)
    If probability <  threshold -> predicted as background (0)

    Parameters
    ----------
    probs : torch.Tensor
        Tensor of probabilities in [0, 1].
    threshold : float
        Threshold used for binarization.

    Returns
    -------
    torch.Tensor
        Binary tensor with values {0, 1}, stored as float for convenience.
    """
    return (probs >= threshold).float()


def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert raw logits directly into a binary mask.

    This is just a convenience wrapper around the two-step process:
        logits -> probabilities -> binary mask

    Why this helper is useful
    -------------------------
    In practice, this same conversion will be needed many times:
    - mask metric computation
    - structural evaluation
    - connectivity analysis
    - visualization
    - fair comparison between models

    Keeping it in one function avoids repeating the same code and ensures
    that exactly the same evaluation logic is used everywhere.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits produced by the model.
    threshold : float
        Threshold applied after sigmoid.

    Returns
    -------
    torch.Tensor
        Binary tensor with values {0, 1}.
    """
    probs = logits_to_probs(logits)
    return probs_to_binary(probs, threshold=threshold)


def tensor_to_numpy_binary(x: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor into a NumPy binary array with dtype uint8.

    Why this is needed
    ------------------
    Model inference is performed in PyTorch, often on GPU.
    However, many evaluation operations are more naturally handled in NumPy,
    and later also in libraries such as scikit-image or scipy.

    This helper:
    1. detaches the tensor from the computation graph,
    2. moves it to CPU,
    3. converts it to a NumPy array,
    4. ensures the result is strictly binary {0,1}.

    Supported shapes
    ----------------
    This works for typical segmentation outputs such as:
    - (B, 1, H, W)
    - (1, H, W)
    - (H, W)

    Note
    ----
    Even if the tensor already contains values 0.0 and 1.0, we apply
    (x > 0) to guarantee a clean binary representation before computing
    confusion counts or structural metrics.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor expected to represent a binary mask.

    Returns
    -------
    np.ndarray
        NumPy array of type uint8 with values in {0,1}.
    """
    x_np = x.detach().cpu().numpy()
    x_np = (x_np > 0).astype(np.uint8)
    return x_np


# ============================================================
# CONFUSION COUNTS
# ============================================================
# Once we have binary masks, we compare prediction and ground truth pixel
# by pixel. This gives the four standard confusion quantities:
#
#   TP = True Positive  -> predicted road and actually road
#   FP = False Positive -> predicted road but actually background
#   FN = False Negative -> predicted background but actually road
#   TN = True Negative  -> predicted background and actually background
#
# These counts are the basis for computing Precision, Recall, Dice/F1, and IoU.


def compute_confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """
    Compute TP, FP, FN, TN for binary segmentation.

    Concept
    -------
    This function performs a pixel-wise comparison between a predicted binary
    mask and the ground-truth binary mask.

    Interpretation
    --------------
    TP: pixels correctly predicted as road
    FP: pixels incorrectly predicted as road
    FN: road pixels that the model missed
    TN: pixels correctly predicted as background

    Why this matters
    ----------------
    Almost all standard segmentation metrics derive from these counts.
    Therefore, this function is the base of the evaluation pipeline.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction array with values {0,1}.
    gt : np.ndarray
        Binary ground-truth array with values {0,1}.

    Returns
    -------
    Dict[str, int]
        Dictionary containing tp, fp, fn, tn.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()
    tn = np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# METRIC COMPUTATION
# ============================================================
# These functions take the confusion counts and turn them into the standard
# segmentation metrics used in the thesis.
#
# Precision = among predicted road pixels, how many are correct?
# Recall    = among real road pixels, how many are recovered?
# F1/Dice   = harmonic balance between precision and recall
# IoU       = overlap between prediction and ground truth
#
# In binary segmentation, Dice and F1 are numerically equivalent.


def precision_recall_f1_iou_from_counts(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute standard segmentation metrics from confusion counts.

    Formulas
    --------
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)
    IoU       = TP / (TP + FP + FN)

    Notes
    -----
    - TN is not used in these formulas because segmentation quality is mainly
      driven by the foreground class (road).
    - A small epsilon is added to avoid division by zero in edge cases.
    - In binary segmentation, Dice and F1 are equivalent, so both names are
      provided for convenience.

    Parameters
    ----------
    tp, fp, fn : int
        Confusion counts.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, f1, dice, and iou.
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(f1),   # In binary segmentation, Dice = F1
        "iou": float(iou),
    }


def compute_mask_metrics_from_arrays(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute standard mask metrics directly from two binary arrays.

    Purpose
    -------
    This is a convenience function for the simple case where a predicted
    binary mask and a ground-truth binary mask are already available.

    Workflow
    --------
    1. Compute TP, FP, FN, TN
    2. Compute Precision, Recall, F1/Dice, and IoU
    3. Return both the counts and the final metrics

    This can be useful for:
    - debugging one specific patch
    - evaluating one image pair
    - unit testing the metric functions

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask.
    gt : np.ndarray
        Binary ground-truth mask.

    Returns
    -------
    Dict[str, float]
        Dictionary containing both confusion counts and standard metrics.
    """
    counts = compute_confusion_counts(pred, gt)
    metrics = precision_recall_f1_iou_from_counts(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
    )
    metrics.update(counts)
    return metrics


# ============================================================
# RUNNING GLOBAL METRICS
# ============================================================
# During final evaluation, the test set contains many patches.
# Instead of computing a metric independently for each batch and then averaging,
# we accumulate TP/FP/FN/TN over the entire dataset first.
#
# This is preferable for final reporting because it gives one global estimate
# over all test pixels and avoids distortions caused by batch-level averaging.
#
# In other words:
#   average of batch Dice  !=  Dice computed from global TP/FP/FN
#
# For the final thesis results, the global version is methodologically cleaner.


class RunningMaskMetrics:
    """
    Accumulate TP/FP/FN/TN across many patches, then compute final global metrics.

    Why this is better for final evaluation
    ---------------------------------------
    Training-time monitoring often computes Dice or IoU batch by batch and then
    averages those values. That is acceptable for monitoring learning progress.

    However, for final model comparison in the thesis, it is better to aggregate
    all confusion counts over the whole test set and compute the metrics once.
    This yields one global result for the full dataset.

    Stored attributes
    -----------------
    tp, fp, fn, tn : int
        Running totals accumulated over all evaluated samples.
    """

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """
        Update the running totals using one batch or one patch.

        Parameters
        ----------
        pred : np.ndarray
            Binary prediction array.
        gt : np.ndarray
            Binary ground-truth array.

        Effect
        ------
        Adds the TP, FP, FN, TN from this input pair to the global totals.
        """
        counts = compute_confusion_counts(pred, gt)
        self.tp += counts["tp"]
        self.fp += counts["fp"]
        self.fn += counts["fn"]
        self.tn += counts["tn"]

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from the accumulated global counts.

        Returns
        -------
        Dict[str, float]
            Dictionary containing final Precision, Recall, F1/Dice, IoU,
            and the corresponding total TP, FP, FN, TN.
        """
        metrics = precision_recall_f1_iou_from_counts(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
        )
        metrics.update({
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        })
        return metrics


# ============================================================
# OPTIONAL TORCH HELPER
# ============================================================
# This helper connects the PyTorch inference stage with the NumPy-based
# metric stage.
#
# It is useful during evaluation because the model produces logits in torch,
# while the metric accumulator expects binary NumPy arrays.


def update_running_metrics_from_logits(
    meter: RunningMaskMetrics,
    logits: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """
    Update a RunningMaskMetrics object starting directly from model logits.

    Full workflow
    -------------
    1. Convert logits to binary predictions using sigmoid + threshold
    2. Convert prediction and ground truth from torch tensors to NumPy arrays
    3. Update the global confusion counts

    Why this helper is useful
    -------------------------
    It keeps the evaluation loop cleaner and makes explicit the full evaluation
    protocol applied to each batch.

    Parameters
    ----------
    meter : RunningMaskMetrics
        Accumulator object storing the global confusion counts.
    logits : torch.Tensor
        Raw model outputs.
    gt : torch.Tensor
        Ground-truth mask tensor.
    threshold : float
        Threshold used after sigmoid to obtain the binary prediction.
    """
    pred = logits_to_binary(logits, threshold=threshold)
    pred_np = tensor_to_numpy_binary(pred)
    gt_np = tensor_to_numpy_binary(gt)
    meter.update(pred_np, gt_np)