from __future__ import annotations

"""
connectivity_utils.py

Utilities for evaluating the connectivity / fragmentation of road predictions.

These metrics are intended to complement standard segmentation metrics
and skeleton-based metrics by quantifying how fragmented the predicted
road masks are.

Implemented metrics
-------------------
1. Number of connected components
2. Largest connected component ratio (LCC ratio)

Definitions
-----------
- Connected component:
    A group of foreground pixels (road pixels) that are connected to each other.

- Largest connected component ratio:
    Number of pixels in the largest component divided by the total number
    of foreground pixels.

Interpretation
--------------
- Fewer connected components usually indicates less fragmentation.
- A higher LCC ratio usually indicates a more coherent network.
"""

from typing import Dict, List
import numpy as np
from scipy import ndimage


def ensure_2d_binary(mask: np.ndarray) -> np.ndarray:
    """
    Ensure the input is a 2D binary mask with values {0,1}.

    Supports:
    - (H, W)
    - (1, H, W)
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    return (mask > 0).astype(np.uint8)


def compute_connected_components(mask: np.ndarray, connectivity: int = 2) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with values {0,1}
    connectivity : int
        1 -> 4-connectivity
        2 -> 8-connectivity

    Returns
    -------
    labeled : np.ndarray
        Labeled array where each connected component has a unique integer label
    num_components : int
        Number of connected foreground components
    """
    mask_2d = ensure_2d_binary(mask)

    if connectivity == 1:
        structure = ndimage.generate_binary_structure(2, 1)  # 4-connected
    elif connectivity == 2:
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connected
    else:
        raise ValueError("connectivity must be 1 or 2")

    labeled, num_components = ndimage.label(mask_2d, structure=structure)
    return labeled, int(num_components)


def largest_connected_component_ratio(mask: np.ndarray, connectivity: int = 2) -> float:
    """
    Compute the ratio of pixels belonging to the largest connected component.

    Formula
    -------
    LCC ratio = size of largest connected component / total foreground pixels

    Edge case
    ---------
    If the mask has no foreground pixels, returns 0.0.
    """
    mask_2d = ensure_2d_binary(mask)
    total_foreground = int(mask_2d.sum())

    if total_foreground == 0:
        return 0.0

    labeled, num_components = compute_connected_components(mask_2d, connectivity=connectivity)

    if num_components == 0:
        return 0.0

    component_sizes = np.bincount(labeled.ravel())[1:]  # ignore background label 0
    largest_size = int(component_sizes.max())

    return float(largest_size / total_foreground)


def compute_connectivity_metrics(mask: np.ndarray, connectivity: int = 2) -> Dict[str, float]:
    """
    Compute connectivity metrics for a single binary mask.

    Returns
    -------
    Dict[str, float]
        num_components, lcc_ratio
    """
    _, num_components = compute_connected_components(mask, connectivity=connectivity)
    lcc_ratio = largest_connected_component_ratio(mask, connectivity=connectivity)

    return {
        "num_components": float(num_components),
        "lcc_ratio": float(lcc_ratio),
    }


class RunningConnectivityMetrics:
    """
    Accumulate connectivity metrics over multiple patches and report means.

    Since connected components are not additive like TP/FP/FN/TN, we compute
    them patch by patch and then average the results over the dataset.
    """

    def __init__(self, connectivity: int = 2) -> None:
        self.connectivity = connectivity
        self.num_components_values: List[float] = []
        self.lcc_ratio_values: List[float] = []

    def update(self, mask: np.ndarray) -> None:
        """
        Compute connectivity metrics for one mask and store them.
        """
        metrics = compute_connectivity_metrics(mask, connectivity=self.connectivity)
        self.num_components_values.append(metrics["num_components"])
        self.lcc_ratio_values.append(metrics["lcc_ratio"])

    def compute(self) -> Dict[str, float]:
        """
        Return mean connectivity statistics over all processed masks.
        """
        if not self.num_components_values:
            return {
                "num_components_mean": 0.0,
                "lcc_ratio_mean": 0.0,
            }

        return {
            "num_components_mean": float(np.mean(self.num_components_values)),
            "lcc_ratio_mean": float(np.mean(self.lcc_ratio_values)),
        }