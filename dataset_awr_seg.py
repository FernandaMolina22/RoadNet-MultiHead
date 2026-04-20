# dataset_awr_seg_patch split.py
"""
Dataset and patch extraction utilities for the Amazon Wild Roads (AWR) dataset.

Key characteristics:
- it first builds all patches from all training mosaics
- then it randomly splits those patches into train and validation
- Uses the same mosaics (satellite images) and masks as Faria et al. (2023).
- Extracts fixed-size patches of 128x128 pixels with a stride of 64.
- Respects the official train/test split defined at the mosaic level.

Output format (default):
    image_tensor: (3, H, W), normalized using ImageNet statistics
    mask_tensor:  (1, H, W), binary values in {0,1}

Optional:
    If topology labels are available and enabled, the dataset can also return:
    centerline_tensor and junction_tensor.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


# ---------------------------------------------------------------------
# 1. Official split from Faria's paper
# ---------------------------------------------------------------------
# These filenames define the dataset split at the MOSAIC level.
# This is critical to avoid spatial leakage between train and test sets.

TRAIN_FILENAMES = [
    "PA1.png","PA8.png","PA5.png","PA12.png",
    "MT1.png","RR2.png","MT3.png","AM4.png",
    "PA11.png","AM2.png","PA10.png","PA7.png",
    "AM1.png","PA3.png","RR1.png","AM5.png",
    "TO1.png","MT2.png","PA4.png","PA6.png"
]

TEST_FILENAMES = [
    "AM3.png","AM7.png","AC1.png","AC2.png",
    "AM6.png","RO1.png","PA2.png","PA9.png"
]

# Patch extraction configuration
PATCH_SIZE = 128   # spatial size of each patch
STRIDE     = 64    # overlap between patches


# ---------------------------------------------------------------------
# 2. Helper dataclass to store patch metadata
# ---------------------------------------------------------------------
# PatchMeta encapsulates all relevant information for a single patch
# extracted from a larger mosaic.

@dataclass
class PatchMeta:
    image_name: str                 # name of the source mosaic
    x: int                          # top-left x-coordinate of the patch
    y: int                          # top-left y-coordinate of the patch
    image_patch: np.ndarray         # (H, W, 3) RGB patch
    mask_patch: np.ndarray          # (H, W) mask patch (0..255)
    centerline_patch: Optional[np.ndarray] = None   # optional topology label
    junction_patch: Optional[np.ndarray] = None     # optional topology label


# ---------------------------------------------------------------------
# 3. Low-level helpers
# ---------------------------------------------------------------------

def read_rgb(path: str) -> np.ndarray:
    """
    Read an image from disk using OpenCV and convert it to RGB format.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8.
    """
    im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def build_patches_for_images(
    data_dir: str,
    filenames: List[str],
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> List[PatchMeta]:
    """
    Generate PatchMeta objects for a list of mosaic filenames.

    Expected folder structure inside data_dir:
        image_png/<filename>
        mask/<filename>

    Optional topology folders:
        centerline_phase1/<filename>
        junction_phase1/<filename>

    If topology folders are present, their patches are extracted and stored.
    Otherwise, topology fields remain None.

    Returns:
        List[PatchMeta]: all extracted patches across mosaics.
    """
    # Define dataset subdirectories
    img_dir      = os.path.join(data_dir, "image_png")
    mask_dir     = os.path.join(data_dir, "mask")
    center_dir   = os.path.join(data_dir, "centerline_phase1")
    junction_dir = os.path.join(data_dir, "junction_phase1")

    # Check availability of optional topology labels
    has_center   = os.path.isdir(center_dir)
    has_junction = os.path.isdir(junction_dir)

    if has_center:
        print(f"[build_patches_for_images] Found centerline dir: {center_dir}")
    else:
        print("[build_patches_for_images] No centerline_phase1/ dir found (centerlines will be None).")

    if has_junction:
        print(f"[build_patches_for_images] Found junction dir: {junction_dir}")
    else:
        print("[build_patches_for_images] No junction_phase1/ dir found (junctions will be None).")

    patches: List[PatchMeta] = []

    # Iterate over all mosaics
    for fname in filenames:
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)

        # Safety checks
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load image and mask
        image    = read_rgb(img_path)                 # (H, W, 3)
        mask_pil = Image.open(mask_path).convert("L") # grayscale mask

        # Load optional topology rasters (if available)
        center_pil = None
        if has_center:
            c_path = os.path.join(center_dir, fname)
            if os.path.exists(c_path):
                center_pil = Image.open(c_path).convert("L")

        junction_pil = None
        if has_junction:
            j_path = os.path.join(junction_dir, fname)
            if os.path.exists(j_path):
                junction_pil = Image.open(j_path).convert("L")

        # PIL uses (width, height) ordering
        width, height = mask_pil.size
        print(f"Building patches for {fname} ({width}x{height})")

        # Sliding window over the mosaic
        x = 0
        while x + patch_size <= width:
            y = 0
            while y + patch_size <= height:

                # Define crop region: (left, upper, right, lower)
                box = (x, y, x + patch_size, y + patch_size)

                # Extract mask patch (PIL -> NumPy)
                cropped_mask = mask_pil.crop(box)
                mask_np = np.array(cropped_mask)

                # Extract image patch (NumPy slicing: [rows, cols])
                img_patch = image[y:y + patch_size, x:x + patch_size, :]

                # Extract optional topology patches
                center_np = None
                if center_pil is not None:
                    center_np = np.array(center_pil.crop(box))

                junction_np = None
                if junction_pil is not None:
                    junction_np = np.array(junction_pil.crop(box))

                # Store patch metadata
                patches.append(
                    PatchMeta(
                        image_name=fname,
                        x=x,
                        y=y,
                        image_patch=img_patch,
                        mask_patch=mask_np,
                        centerline_patch=center_np,
                        junction_patch=junction_np,
                    )
                )

                y += stride
            x += stride

    print(f"Total patches: {len(patches)}")
    return patches


def build_train_val_patches(
    data_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PatchMeta], List[PatchMeta]]:
    """
    Build patches from TRAIN mosaics and split them into train/validation sets.

    IMPORTANT:
    - The split is performed at the PATCH level (not mosaic level).
    - Mosaic-level separation is already enforced by TRAIN_FILENAMES.

    Returns:
        patches_train, patches_val
    """
    all_patches = build_patches_for_images(data_dir, TRAIN_FILENAMES)

    # Shuffle indices for reproducibility
    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_patches))
    rng.shuffle(idx)

    # Compute split
    n_val = int(len(idx) * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Build subsets
    patches_train = [all_patches[i] for i in train_idx]
    patches_val   = [all_patches[i] for i in val_idx]

    print(f"Train patches: {len(patches_train)} | Val patches: {len(patches_val)}")
    return patches_train, patches_val


def build_test_patches(data_dir: str) -> List[PatchMeta]:
    """
    Build patches from TEST mosaics.

    NOTE:
    - No splitting is applied here.
    - Test data is completely independent at mosaic level.
    """
    return build_patches_for_images(data_dir, TEST_FILENAMES)


# ---------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------

class RoadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for road segmentation on AWR patches.

    Default output:
        image: (3, H, W) normalized tensor
        mask:  (1, H, W) binary tensor

    Optional output (if return_topology=True and available):
        centerline: (1, H, W)
        junction:   (1, H, W)
    """

    def __init__(
        self,
        patches: List[PatchMeta],
        train: bool = True,
        return_topology: bool = False,
    ):
        self.patches = patches
        self.train = train
        self.return_topology = return_topology

        # Standard ImageNet normalization (common practice for pretrained encoders)
        self.img_transform = T.Compose([
            T.ToTensor(),  # scales to [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # NOTE: data augmentation can be added here if needed

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        meta = self.patches[idx]

        img  = meta.image_patch
        mask = meta.mask_patch

        # Convert to PIL for torchvision transforms
        img_pil  = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)

        # Transform image
        img_t = self.img_transform(img_pil)

        # Convert mask to binary {0,1}
        mask_np = np.array(mask_pil, dtype=np.uint8)
        mask_bin = (mask_np > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)

        # If topology is disabled or not available, return basic pair
        if (not self.return_topology) or (meta.centerline_patch is None) or (meta.junction_patch is None):
            return img_t, mask_t

        # Build topology tensors
        center_np = np.array(meta.centerline_patch, dtype=np.uint8)
        center_bin = (center_np > 0).astype(np.float32)
        center_t = torch.from_numpy(center_bin).unsqueeze(0)

        junction_np = np.array(meta.junction_patch, dtype=np.uint8)
        junction_bin = (junction_np > 0).astype(np.float32)
        junction_t = torch.from_numpy(junction_bin).unsqueeze(0)

        return img_t, mask_t, center_t, junction_t


# ---------------------------------------------------------------------
# 5. Convenience function: DataLoaders
# ---------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_topology: bool = False,
):
    """
    Build DataLoaders for training, validation, and testing.

    This function:
    1. Extracts patches from mosaics
    2. Splits train/validation
    3. Wraps everything into PyTorch DataLoaders

    Returns:
        train_loader, val_loader, test_loader
    """

    # Build patch datasets
    train_patches, val_patches = build_train_val_patches(
        data_dir=data_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    test_patches = build_test_patches(data_dir)

    # Wrap into Dataset objects
    train_ds = RoadSegmentationDataset(
        train_patches,
        train=True,
        return_topology=return_topology,
    )
    val_ds   = RoadSegmentationDataset(
        val_patches,
        train=False,
        return_topology=return_topology,
    )
    test_ds  = RoadSegmentationDataset(
        test_patches,
        train=False,
        return_topology=return_topology,
    )

    # Common DataLoader configuration
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Create DataLoaders
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader