# dataset_awr_seg_mosaic split.py
"""
Dataset and patch extraction utilities for the Amazon Wild Roads (AWR) dataset.

Key characteristics:
- it first splits the training mosaics themselves
- then it extracts patches separately for train mosaics and validation mosaics
- Uses the same mosaics (satellite images) and masks as Faria et al. (2023).
- Extracts fixed-size patches of 128x128 pixels with a stride of 64.
- Respects the official train/test split defined at the mosaic level.

Output format (default):
    image_tensor: (3, H, W), normalized using ImageNet statistics
    mask_tensor:  (1, H, W), binary values in {0,1}

Optional:
    If topology labels are available and enabled, the dataset can also return
    centerline and junction tensors.
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
# These filenames define the official split at the mosaic level.
# Keeping train and test separated by full mosaics helps prevent
# spatial leakage between subsets.

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
PATCH_SIZE = 128
STRIDE     = 64


# ---------------------------------------------------------------------
# 2. Helper dataclass to store patch metadata
# ---------------------------------------------------------------------
# PatchMeta stores the content and spatial origin of a single patch
# extracted from a larger mosaic.

@dataclass
class PatchMeta:
    image_name: str
    x: int
    y: int
    image_patch: np.ndarray          # (H, W, 3) RGB patch
    mask_patch: np.ndarray           # (H, W) grayscale mask patch
    centerline_patch: Optional[np.ndarray] = None   # optional centerline label patch
    junction_patch: Optional[np.ndarray] = None     # optional junction label patch


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
    Build a list of PatchMeta objects from a list of mosaic filenames.

    Expected folder structure inside data_dir:
        data_dir/image_png/<filename>
        data_dir/mask/<filename>

    Optional topology folders:
        data_dir/centerline_phase1/<filename>
        data_dir/junction_phase1/<filename>

    If topology rasters are present, they are cropped alongside the
    main image and mask and stored inside each PatchMeta object.
    """
    img_dir      = os.path.join(data_dir, "image_png")
    mask_dir     = os.path.join(data_dir, "mask")
    center_dir   = os.path.join(data_dir, "centerline_phase1")
    junction_dir = os.path.join(data_dir, "junction_phase1")

    # Check whether optional topology folders exist
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

    # Process each selected mosaic independently
    for fname in filenames:
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)

        # Ensure required files exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load RGB image and grayscale road mask
        image    = read_rgb(img_path)                    # (H, W, 3)
        mask_pil = Image.open(mask_path).convert("L")    # grayscale mask

        # Optionally load full-size topology rasters
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

        # PIL reports size as (width, height)
        width, height = mask_pil.size
        print(f"Building patches for {fname} ({width}x{height})")

        # Sliding-window patch extraction
        x = 0
        while x + patch_size <= width:
            y = 0
            while y + patch_size <= height:
                # Crop region in PIL format: (left, upper, right, lower)
                box = (x, y, x + patch_size, y + patch_size)

                # Crop mask patch
                cropped_mask = mask_pil.crop(box)
                mask_np = np.array(cropped_mask)

                # Crop RGB patch using NumPy indexing: [rows, cols]
                img_patch = image[y:y + patch_size, x:x + patch_size, :]

                # Crop optional topology patches
                center_np = None
                if center_pil is not None:
                    center_np = np.array(center_pil.crop(box))

                junction_np = None
                if junction_pil is not None:
                    junction_np = np.array(junction_pil.crop(box))

                # Store extracted patch and its metadata
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
    Build training and validation patch lists using a mosaic-level split.

    In this version, TRAIN_FILENAMES is first split into training mosaics
    and validation mosaics, and only then are patches extracted.

    This is stricter than patch-level splitting because it prevents
    neighboring or overlapping patches from the same mosaic from appearing
    in both training and validation sets.
    """
    rng = np.random.default_rng(seed)

    filenames = TRAIN_FILENAMES.copy()
    rng.shuffle(filenames)

    # Select a subset of mosaics for validation
    n_val = max(1, int(len(filenames) * val_fraction))
    val_filenames = filenames[:n_val]
    train_filenames = filenames[n_val:]

    print(f"Train mosaics ({len(train_filenames)}): {train_filenames}")
    print(f"Val mosaics   ({len(val_filenames)}): {val_filenames}")

    # Extract patches independently for each subset
    patches_train = build_patches_for_images(data_dir, train_filenames)
    patches_val   = build_patches_for_images(data_dir, val_filenames)

    print(f"Train patches: {len(patches_train)} | Val patches: {len(patches_val)}")
    return patches_train, patches_val


def build_test_patches(data_dir: str) -> List[PatchMeta]:
    """
    Build the patch list for the independent test mosaics.
    """
    return build_patches_for_images(data_dir, TEST_FILENAMES)


# ---------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------

class RoadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for road segmentation on AWR patches.

    Default output:
        image: (3, H, W) float32 tensor, normalized
        mask:  (1, H, W) float32 tensor in {0,1}

    If return_topology=True and topology labels are available for the patch,
    the dataset instead returns:
        image:      (3, H, W)
        mask:       (1, H, W)
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

        # Standard image preprocessing with ImageNet normalization
        self.img_transform = T.Compose([
            T.ToTensor(),  # scales values to [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # NOTE: data augmentation can be added here later if needed

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        meta = self.patches[idx]

        img  = meta.image_patch
        mask = meta.mask_patch

        # Convert arrays to PIL format for torchvision transforms
        img_pil  = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)

        # Normalize image
        img_t = self.img_transform(img_pil)

        # Convert mask from grayscale to binary {0,1}
        mask_np = np.array(mask_pil, dtype=np.uint8)
        mask_bin = (mask_np > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)

        # If topology output is disabled or unavailable, return image and mask only
        if (not self.return_topology) or (meta.centerline_patch is None) or (meta.junction_patch is None):
            return img_t, mask_t

        # Build binary topology tensors
        center_np = np.array(meta.centerline_patch, dtype=np.uint8)
        center_bin = (center_np > 0).astype(np.float32)
        center_t = torch.from_numpy(center_bin).unsqueeze(0)

        junction_np = np.array(meta.junction_patch, dtype=np.uint8)
        junction_bin = (junction_np > 0).astype(np.float32)
        junction_t = torch.from_numpy(junction_bin).unsqueeze(0)

        return img_t, mask_t, center_t, junction_t

    

# ---------------------------------------------------------------------
# 5. Convenience function: build DataLoaders for train/val/test
# ---------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_topology: bool = False,   # <<< NEW
):
    """
    Build train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Root directory of the AWR dataset.
    batch_size : int
        Batch size used in all DataLoaders.
    val_fraction : float
        Fraction of training mosaics reserved for validation.
    seed : int
        Random seed controlling the train/validation mosaic split.
    num_workers : int
        Number of worker processes used by each DataLoader.
    pin_memory : bool
        Whether to pin memory, which may improve GPU transfer speed.
    return_topology : bool
        If True, the dataset attempts to return topology labels
        (centerline and junction) in addition to image and mask.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    # Build patch lists for each subset
    train_patches, val_patches = build_train_val_patches(
        data_dir=data_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    test_patches = build_test_patches(data_dir)

    # Wrap patch lists into PyTorch Dataset objects
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

    # Build final DataLoaders
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader