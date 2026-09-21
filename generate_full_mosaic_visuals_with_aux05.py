from __future__ import annotations

"""
generate_full_mosaic_visuals.py

Purpose
-------
Generate qualitative full-mosaic and large-crop visualizations for the thesis.

This script does NOT train models.
This script does NOT replace the quantitative evaluation protocol.

Instead, it performs qualitative sliding-window inference over a complete
held-out test mosaic. The model is still applied using the same 128x128 patch
size used during training and evaluation. Overlapping patch probabilities are
averaged to reconstruct a full-size prediction map.

This is useful for visualizing:
- larger spatial context,
- apparent road continuity,
- false-positive regions,
- differences between UNetSingle and UNetMultiHead variants.

Important
---------
These full-mosaic outputs are intended for qualitative interpretation only.
The official quantitative results remain the patch-based evaluation reported
in evaluate_models.py.
"""

from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"

# Choose one held-out test mosaic.
# Official test mosaics:
# "AM3.png", "AM7.png", "AC1.png", "AC2.png",
# "AM6.png", "RO1.png", "PA2.png", "PA9.png"
MOSAIC_NAME = "PA9.png"

# Seed controls which checkpoints are loaded.
SEED = 42
# SEED = 123
# SEED = 2025

OUTPUT_DIR = Path(f"full_mosaic_visuals_seed{SEED}_{Path(MOSAIC_NAME).stem}_with_aux05")

PATCH_SIZE = 128
STRIDE = 64
THRESHOLD = 0.5

# Device batch size for sliding-window inference.
# Reduce if you get CUDA memory errors.
INFERENCE_BATCH_SIZE = 32

# Optional crop to show a more readable region of the full mosaic.
# Format: x, y, width, height.
# Set to None to save only full-mosaic views.
#
# Example:
# CROP_BOX = (500, 500, 800, 800)
CROP_BOX = None

# If CROP_BOX is None, this automatic crop is used.
# It finds a region with many ground-truth road pixels.
AUTO_CROP_SIZE = 800

# ImageNet normalization used during training.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def single_ckpt(seed: int) -> Path:
    if seed == 42:
        return Path("runs_unet_single/best_model_unet_single.pth")
    return Path(f"runs_unet_single_seed{seed}/best_model_unet_single.pth")


def multihead_equal_ckpt(seed: int) -> Path:
    if seed == 42:
        return Path("runs_unet_multihead/best_model_unet_multihead.pth")
    return Path(f"runs_unet_multihead_seed{seed}/best_model_unet_multihead.pth")


def multihead_weighted_ckpt(seed: int, experiment_name: str) -> Path:
    return Path(
        f"runs_unet_multihead_seed{seed}_{experiment_name}/best_model_unet_multihead.pth"
    )


MODEL_CONFIGS = {
    "UNetSingle": {
        "type": "single",
        "path": single_ckpt(SEED),
    },
    "MultiHead equal": {
        "type": "multihead",
        "path": multihead_equal_ckpt(SEED),
    },
    "MultiHead aux05": {
        "type": "multihead",
        "path": multihead_weighted_ckpt(SEED, "aux05"),
    },
    "MultiHead junction-focus": {
        "type": "multihead",
        "path": multihead_weighted_ckpt(SEED, "junction_focus"),
    },
    "MultiHead center-focus": {
        "type": "multihead",
        "path": multihead_weighted_ckpt(SEED, "center_focus"),
    },
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_type: str, checkpoint_path: Path, device: torch.device):
    if model_type == "single":
        model = build_unet_single(device)
    elif model_type == "multihead":
        model = build_unet_multihead(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_all_models(device: torch.device) -> dict:
    models = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"Loading {model_name}: {cfg['path']}")
        models[model_name] = {
            "type": cfg["type"],
            "model": load_model(cfg["type"], cfg["path"], device),
        }

    return models


# ============================================================
# IMAGE LOADING
# ============================================================

def read_rgb_mosaic(mosaic_name: str) -> np.ndarray:
    path = Path(DATA_DIR) / "image_png" / mosaic_name
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_mask_mosaic(mosaic_name: str) -> np.ndarray:
    path = Path(DATA_DIR) / "mask" / mosaic_name
    mask = Image.open(path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


# ============================================================
# SLIDING-WINDOW HELPERS
# ============================================================

def compute_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """
    Compute sliding-window start coordinates.

    Includes the last possible coordinate so that the full image is covered,
    even if the image size is not exactly divisible by the stride.
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))

    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def build_patch_tensor(patches: list[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of RGB patches [H,W,3] uint8 to a normalized tensor batch.
    Output shape: [B,3,H,W]
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensors = []
    for patch in patches:
        patch_pil = Image.fromarray(patch)
        tensors.append(transform(patch_pil))

    return torch.stack(tensors, dim=0)


@torch.no_grad()
def predict_patch_batch(
    model: torch.nn.Module,
    model_type: str,
    patch_batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict probability maps for a batch of patches.

    Returns:
        probs: NumPy array of shape [B,H,W]
    """
    patch_batch = patch_batch.to(device)

    if model_type == "single":
        logits = model(patch_batch)
    else:
        outputs = model(patch_batch)
        logits = outputs["mask"]

    probs = torch.sigmoid(logits)

    # Shape [B,1,H,W] -> [B,H,W]
    probs_np = probs.detach().cpu().numpy()[:, 0, :, :]

    return probs_np


def sliding_window_predict_mosaic(
    model: torch.nn.Module,
    model_type: str,
    rgb: np.ndarray,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """
    Run sliding-window inference over a full RGB mosaic.

    The model predicts on 128x128 patches.
    Overlapping probabilities are averaged.

    Returns:
        probability_map: float array [H,W] with values in [0,1]
    """
    height, width, _ = rgb.shape

    y_starts = compute_starts(height, patch_size, stride)
    x_starts = compute_starts(width, patch_size, stride)

    prob_sum = np.zeros((height, width), dtype=np.float32)
    count_sum = np.zeros((height, width), dtype=np.float32)

    patch_list = []
    coord_list = []

    total_patches = len(y_starts) * len(x_starts)
    processed = 0

    for y in y_starts:
        for x in x_starts:
            patch = rgb[y:y + patch_size, x:x + patch_size, :]

            patch_list.append(patch)
            coord_list.append((x, y))

            if len(patch_list) == batch_size:
                patch_batch = build_patch_tensor(patch_list)
                probs = predict_patch_batch(model, model_type, patch_batch, device)

                for prob, (px, py) in zip(probs, coord_list):
                    prob_sum[py:py + patch_size, px:px + patch_size] += prob
                    count_sum[py:py + patch_size, px:px + patch_size] += 1.0

                processed += len(patch_list)
                print(f"Processed {processed}/{total_patches} patches", end="\r")

                patch_list = []
                coord_list = []

    # Process remaining patches
    if patch_list:
        patch_batch = build_patch_tensor(patch_list)
        probs = predict_patch_batch(model, model_type, patch_batch, device)

        for prob, (px, py) in zip(probs, coord_list):
            prob_sum[py:py + patch_size, px:px + patch_size] += prob
            count_sum[py:py + patch_size, px:px + patch_size] += 1.0

        processed += len(patch_list)
        print(f"Processed {processed}/{total_patches} patches", end="\r")

    print()

    # Avoid division by zero, although count_sum should be >0 everywhere.
    count_sum[count_sum == 0] = 1.0
    probability_map = prob_sum / count_sum

    return probability_map


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def overlay_mask_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay a binary mask on RGB image.

    rgb should be uint8 [H,W,3].
    mask should be binary [H,W].
    color is in RGB float format [0,1].
    """
    rgb_float = rgb.astype(np.float32) / 255.0
    out = rgb_float.copy()

    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)

    out[mask_bool] = (
        (1.0 - alpha) * out[mask_bool]
        + alpha * color_arr
    )

    return np.clip(out, 0.0, 1.0)


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Error map:
        Green = true positives
        Red   = false positives
        Blue  = false negatives
        Black = true negatives/background
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    error = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)

    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)

    error[tp] = np.array([0.0, 1.0, 0.0])
    error[fp] = np.array([1.0, 0.0, 0.0])
    error[fn] = np.array([0.0, 0.0, 1.0])

    return error


def find_auto_crop(mask: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    """
    Automatically find a crop with many ground-truth road pixels.

    Returns:
        x, y, width, height
    """
    h, w = mask.shape

    if h <= crop_size or w <= crop_size:
        return 0, 0, min(w, crop_size), min(h, crop_size)

    best_score = -1
    best_xy = (0, 0)

    # Use a coarse search to avoid checking every pixel.
    step = crop_size // 4

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            crop = mask[y:y + crop_size, x:x + crop_size]
            score = int(crop.sum())

            if score > best_score:
                best_score = score
                best_xy = (x, y)

    x, y = best_xy
    return x, y, crop_size, crop_size


def crop_array(arr: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop array using crop_box = (x, y, width, height).
    """
    x, y, width, height = crop_box
    return arr[y:y + height, x:x + width]


# ============================================================
# FIGURE SAVING
# ============================================================

def save_full_mosaic_overview(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: dict[str, np.ndarray],
    output_dir: Path,
):
    """
    Save full-mosaic overview figures.

    These may be large and visually dense, but they are useful for inspecting
    the whole prediction over the complete test mosaic.
    """
    full_dir = output_dir / "full_mosaic"
    full_dir.mkdir(parents=True, exist_ok=True)

    gt_overlay = overlay_mask_on_rgb(
        rgb,
        gt_mask,
        color=(1.0, 1.0, 0.0),
        alpha=0.60,
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"RGB mosaic: {MOSAIC_NAME}")
    plt.savefig(full_dir / "rgb_full.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(gt_overlay)
    plt.axis("off")
    plt.title("RGB + ground truth overlay")
    plt.savefig(full_dir / "rgb_gt_overlay_full.png", dpi=300, bbox_inches="tight")
    plt.close()

    for model_name, pred in pred_masks.items():
        overlay = overlay_mask_on_rgb(
            rgb,
            pred,
            color=(1.0, 0.0, 0.0),
            alpha=0.55,
        )

        safe_name = model_name.replace(" ", "_").replace("-", "_")

        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"{model_name} prediction overlay")
        plt.savefig(full_dir / f"{safe_name}_prediction_overlay_full.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_crop_comparison(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: dict[str, np.ndarray],
    crop_box: tuple[int, int, int, int],
    output_dir: Path,
):
    """
    Save a large-crop comparison figure.

    Layout:
        RGB crop
        RGB + GT overlay
        prediction overlay for each model
        error map for each model
    """
    crop_dir = output_dir / "crop_comparison"
    crop_dir.mkdir(parents=True, exist_ok=True)

    x, y, width, height = crop_box

    rgb_crop = crop_array(rgb, crop_box)
    gt_crop = crop_array(gt_mask, crop_box)

    pred_crops = {
        name: crop_array(pred, crop_box)
        for name, pred in pred_masks.items()
    }

    n_models = len(pred_crops)
    n_cols = 2 + n_models
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))

    # Row 1: RGB, GT overlay, prediction overlays
    axes[0, 0].imshow(rgb_crop)
    axes[0, 0].set_title("RGB crop")
    axes[0, 0].axis("off")

    gt_overlay = overlay_mask_on_rgb(
        rgb_crop,
        gt_crop,
        color=(1.0, 1.0, 0.0),
        alpha=0.60,
    )
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("GT overlay")
    axes[0, 1].axis("off")

    for j, (model_name, pred_crop) in enumerate(pred_crops.items()):
        overlay = overlay_mask_on_rgb(
            rgb_crop,
            pred_crop,
            color=(1.0, 0.0, 0.0),
            alpha=0.55,
        )
        axes[0, j + 2].imshow(overlay)
        axes[0, j + 2].set_title(f"{model_name}\npred overlay")
        axes[0, j + 2].axis("off")

    # Row 2: blank, GT mask, error maps
    axes[1, 0].imshow(gt_crop, cmap="gray")
    axes[1, 0].set_title("GT mask")
    axes[1, 0].axis("off")

    axes[1, 1].axis("off")

    for j, (model_name, pred_crop) in enumerate(pred_crops.items()):
        error = create_error_map(pred_crop, gt_crop)
        axes[1, j + 2].imshow(error)
        axes[1, j + 2].set_title(f"{model_name}\nerror map")
        axes[1, j + 2].axis("off")

    fig.suptitle(
        f"Mosaic: {MOSAIC_NAME} | crop x={x}, y={y}, w={width}, h={height}\n"
        f"Green=TP, Red=FP, Blue=FN",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(crop_dir / "crop_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also save the crop coordinates for reproducibility.
    with (crop_dir / "crop_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"mosaic={MOSAIC_NAME}\n")
        f.write(f"x={x}\n")
        f.write(f"y={y}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"patch_size={PATCH_SIZE}\n")
        f.write(f"stride={STRIDE}\n")
        f.write(f"threshold={THRESHOLD}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Mosaic: {MOSAIC_NAME}")
    print(f"Seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")

    rgb = read_rgb_mosaic(MOSAIC_NAME)
    gt_mask = read_mask_mosaic(MOSAIC_NAME)

    print(f"RGB shape: {rgb.shape}")
    print(f"GT mask shape: {gt_mask.shape}")
    print(f"GT road pixels: {int(gt_mask.sum())}")

    models = load_all_models(device)

    probability_maps = {}
    pred_masks = {}

    for model_name, item in models.items():
        print("\n" + "=" * 60)
        print(f"Running sliding-window inference for: {model_name}")
        print("=" * 60)

        prob_map = sliding_window_predict_mosaic(
            model=item["model"],
            model_type=item["type"],
            rgb=rgb,
            device=device,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            batch_size=INFERENCE_BATCH_SIZE,
        )

        pred_mask = (prob_map > THRESHOLD).astype(np.uint8)

        probability_maps[model_name] = prob_map
        pred_masks[model_name] = pred_mask

    # Save probability maps and binary masks as NumPy arrays.
    arrays_dir = OUTPUT_DIR / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    for model_name, prob_map in probability_maps.items():
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        np.save(arrays_dir / f"{safe_name}_probability_map.npy", prob_map)
        np.save(arrays_dir / f"{safe_name}_binary_mask.npy", pred_masks[model_name])

    # Save full-mosaic overview images.
    save_full_mosaic_overview(
        rgb=rgb,
        gt_mask=gt_mask,
        pred_masks=pred_masks,
        output_dir=OUTPUT_DIR,
    )

    # Determine crop region.
    if CROP_BOX is None:
        crop_box = find_auto_crop(gt_mask, AUTO_CROP_SIZE)
        print(f"Auto-selected crop box: {crop_box}")
    else:
        crop_box = CROP_BOX
        print(f"Using manual crop box: {crop_box}")

    # Save large-crop comparison figure.
    save_crop_comparison(
        rgb=rgb,
        gt_mask=gt_mask,
        pred_masks=pred_masks,
        crop_box=crop_box,
        output_dir=OUTPUT_DIR,
    )

    print("\nDone.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()