# ============================================
# make_topology_labels_phase1.py
#
# Phase 1 topology labels:
#   - Generate centerlines via skeletonization from binary masks
#   - Detect junctions from the skeleton
#
# Output:
#   centerline_phase1/<name>.png
#   junction_phase1/<name>.png
#
# This works at FULL IMAGE level (not patch level).
# Patching is still handled by dataset_awr_seg.py.
# ============================================

from pathlib import Path

import numpy as np
from skimage import io, morphology
from scipy.ndimage import convolve, binary_dilation


def build_phase1_labels(
    data_dir: str,
    mask_dirname: str = "mask",
    centerline_dirname: str = "centerline_phase1",
    junction_dirname: str = "junction_phase1",
    junction_min_neighbors: int = 3,
):
    """
    Generate centerline and junction GT from binary road masks.

    Parameters
    ----------
    data_dir : str
        Root directory of AWR dataset (same as DATA_DIR in training).
    mask_dirname : str
        Name of the subfolder containing mask PNGs.
    centerline_dirname : str
        Name of the output subfolder for centerline masks.
    junction_dirname : str
        Name of the output subfolder for junction masks.
    junction_min_neighbors : int
        Minimum number of skeleton neighbors (in 8-connected sense)
        for a pixel to be considered a junction.
    """
    data_path = Path(data_dir)
    mask_dir = data_path / mask_dirname
    out_center_dir = data_path / centerline_dirname
    out_junction_dir = data_path / junction_dirname

    out_center_dir.mkdir(parents=True, exist_ok=True)
    out_junction_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_dir.glob("*.png"))
    print(f"Found {len(mask_paths)} mask images in {mask_dir}")

    # 3x3 kernel to count 8-neighbors (center=0 so we don't count the pixel itself)
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    for mask_path in mask_paths:
        name = mask_path.name
        print(f"Processing {name}...")

        # -----------------------
        # 1) Load and binarize mask
        # -----------------------
        mask = io.imread(mask_path)

        # Handle RGB or single-channel
        if mask.ndim == 3:
            # Convert to grayscale by taking one channel (they are usually identical)
            mask = mask[..., 0]

        # Normalize to {0,1}
        # If mask is 0/255 uint8, this yields 0 or 1.
        mask_bin = (mask > 0).astype(np.uint8)

        # Skip completely empty masks (just in case)
        if mask_bin.sum() == 0:
            print("  -> Empty mask, writing empty centerline & junction.")
            centerline = np.zeros_like(mask_bin, dtype=np.uint8)
            junctions = np.zeros_like(mask_bin, dtype=np.uint8)
        else:
            # -----------------------
            # 2) Skeletonization → centerline
            # -----------------------
            # skimage expects boolean mask
            skeleton = morphology.skeletonize(mask_bin.astype(bool))
            centerline = skeleton.astype(np.uint8)

            # -----------------------
            # 3) Junction detection
            # -----------------------
            # Count neighbors in 8-connected sense
            neighbor_count = convolve(centerline, kernel, mode="constant", cval=0)

            # A junction is a skeleton pixel with at least `junction_min_neighbors` neighbors
            junctions_bool = (centerline == 1) & (neighbor_count >= junction_min_neighbors)

            # Optional: dilate junctions a bit to make them more visible / learnable
            junctions_bool = binary_dilation(
                junctions_bool,
                structure=np.ones((3, 3), dtype=bool),
                iterations=1,
            )

            junctions = junctions_bool.astype(np.uint8)

        # -----------------------
        # 4) Save results as PNG (0 and 255)
        # -----------------------
        centerline_path = out_center_dir / name
        junction_path = out_junction_dir / name

        io.imsave(centerline_path.as_posix(), (centerline * 255).astype(np.uint8))
        io.imsave(junction_path.as_posix(), (junctions * 255).astype(np.uint8))

    print("\nDone. Phase 1 labels written to:")
    print(f"  Centerlines: {out_center_dir}")
    print(f"  Junctions:   {out_junction_dir}")


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"
    build_phase1_labels(DATA_DIR)
