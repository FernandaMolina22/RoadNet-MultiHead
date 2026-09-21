from __future__ import annotations

"""
generate_qualitative_candidates.py

Purpose
-------
Search the held-out TEST set for strong qualitative examples for the thesis.

This script does NOT train models and does NOT change the quantitative
evaluation protocol.

It does two things:

1. Patch-level qualitative comparison:
   - Uses the same 128x128 test patches used during evaluation.
   - Runs inference with trained models.
   - Saves error-map figures comparing predictions with the ground truth.

2. Larger-context visualization:
   - Uses the original full test mosaic.
   - Extracts a larger RGB crop around the selected 128x128 patch.
   - Overlays the ground-truth mask.
   - Draws a rectangle showing exactly where the evaluated patch is located.

This helps with qualitative interpretation when the RGB image appears to contain
road-like structures that are not annotated in the ground-truth mask.

Error map colors
----------------
Green = true positives
Red   = false positives
Blue  = false negatives
Black = true negatives / background
"""

from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from dataset_awr_seg_mosaic_split import get_dataloaders
from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead
from evaluation_utils import logits_to_binary, tensor_to_numpy_binary


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"

# Start with seed 42. You can later change to 123 or 2025 if you want
# to inspect qualitative examples from other trained runs.
SEED = 42
# SEED = 123
# SEED = 2025

OUTPUT_DIR = Path(f"visual_results_candidates_seed{SEED}_with_aux05")

BATCH_SIZE = 8
VAL_FRACTION = 0.10
NUM_WORKERS = 4
PIN_MEMORY = True
THRESHOLD = 0.5

# Number of candidate figures to save per category.
TOP_K = 10

# Minimum number of ground-truth road pixels required in the 128x128 patch.
# Increase this if the selected patches contain too little annotated road.
MIN_GT_PIXELS = 500

# Size of the larger contextual crop around the 128x128 evaluated patch.
# This does NOT change model inference. It is only for visualization.
CONTEXT_SIZE = 512

# Patch size used in the dataset.
PATCH_SIZE = 128

# ImageNet normalization parameters used by the dataset.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def single_ckpt(seed: int) -> Path:
    """
    Return checkpoint path for UNetSingle.

    Seed 42 uses the original folder name.
    Other seeds use seed-specific folders.
    """
    if seed == 42:
        return Path("runs_unet_single/best_model_unet_single.pth")
    return Path(f"runs_unet_single_seed{seed}/best_model_unet_single.pth")


def multihead_equal_ckpt(seed: int) -> Path:
    """
    Return checkpoint path for the original equal-weight UNetMultiHead.
    """
    if seed == 42:
        return Path("runs_unet_multihead/best_model_unet_multihead.pth")
    return Path(f"runs_unet_multihead_seed{seed}/best_model_unet_multihead.pth")


def multihead_weighted_ckpt(seed: int, experiment_name: str) -> Path:
    """
    Return checkpoint path for a weighted multi-head experiment.
    """
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
    """
    Build the model architecture and load trained weights.
    """
    if model_type == "single":
        model = build_unet_single(device)
    elif model_type == "multihead":
        model = build_unet_multihead(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_all_models(device: torch.device) -> dict:
    """
    Load all models listed in MODEL_CONFIGS.
    """
    models = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        ckpt = cfg["path"]

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt}")

        print(f"Loading {model_name}: {ckpt}")

        models[model_name] = {
            "type": cfg["type"],
            "model": load_model(cfg["type"], ckpt, device),
        }

    return models


# ============================================================
# DATA LOADING
# ============================================================

def get_test_loader(device: torch.device):
    """
    Load test dataloader.

    The dataset internally stores PatchMeta objects, so later we can recover:
        - original mosaic filename,
        - x coordinate,
        - y coordinate,
        - patch position in the full mosaic.
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
    return test_loader


# ============================================================
# FULL MOSAIC / CONTEXT HELPERS
# ============================================================

def read_full_rgb(image_name: str) -> np.ndarray:
    """
    Read the complete original RGB mosaic from DATA_DIR/image_png.
    """
    path = Path(DATA_DIR) / "image_png" / image_name
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_full_mask(image_name: str) -> np.ndarray:
    """
    Read the complete original ground-truth mask from DATA_DIR/mask.
    """
    path = Path(DATA_DIR) / "mask" / image_name
    mask = Image.open(path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def get_patch_metadata(test_loader, batch_index: int, sample_index: int) -> dict:
    """
    Recover metadata for a saved candidate.

    Candidate filename example:
        01_helpful_recovery_b263_s4_score0.956.png

    Here:
        batch_index = 263
        sample_index = 4

    Since the test dataloader uses shuffle=False, the global dataset index is:
        global_index = batch_index * BATCH_SIZE + sample_index
    """
    global_index = batch_index * BATCH_SIZE + sample_index
    meta = test_loader.dataset.patches[global_index]

    return {
        "global_index": global_index,
        "image_name": meta.image_name,
        "x": int(meta.x),
        "y": int(meta.y),
        "patch_size": int(meta.image_patch.shape[0]),
        "full_image_path": str(Path(DATA_DIR) / "image_png" / meta.image_name),
        "full_mask_path": str(Path(DATA_DIR) / "mask" / meta.image_name),
    }


def extract_context_crop(
    full_image: np.ndarray,
    full_mask: np.ndarray,
    patch_x: int,
    patch_y: int,
    patch_size: int = PATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
) -> dict:
    """
    Extract a larger crop around the selected 128x128 patch.

    The returned rectangle indicates where the evaluated patch lies inside
    the context crop.
    """
    h, w = full_mask.shape

    patch_center_x = patch_x + patch_size // 2
    patch_center_y = patch_y + patch_size // 2
    half = context_size // 2

    x0 = max(0, patch_center_x - half)
    y0 = max(0, patch_center_y - half)
    x1 = min(w, x0 + context_size)
    y1 = min(h, y0 + context_size)

    # If the crop hits the right/bottom border, shift it back when possible.
    x0 = max(0, x1 - context_size)
    y0 = max(0, y1 - context_size)

    context_rgb = full_image[y0:y1, x0:x1, :]
    context_mask = full_mask[y0:y1, x0:x1]

    # Patch rectangle coordinates relative to the context crop.
    rect_x = patch_x - x0
    rect_y = patch_y - y0

    return {
        "context_rgb": context_rgb,
        "context_mask": context_mask,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "rect_x": rect_x,
        "rect_y": rect_y,
        "rect_w": patch_size,
        "rect_h": patch_size,
    }


def overlay_mask_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image.

    Default color is yellow, which is useful for ground-truth visualization.
    """
    rgb_float = rgb.astype(np.float32) / 255.0
    overlay = rgb_float.copy()

    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)

    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool]
        + alpha * color_arr
    )

    return np.clip(overlay, 0.0, 1.0)


# ============================================================
# TENSOR / MASK CONVERSION HELPERS
# ============================================================

def tensor_to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized RGB tensor [3,H,W] back to displayable RGB [H,W,3].
    """
    image = image_tensor.detach().cpu().numpy()

    if image.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got {image.shape}")

    image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)

    return image


def squeeze_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert mask from [1,H,W] or [H,W] to [H,W].
    """
    if mask_np.ndim == 3:
        return mask_np[0].astype(np.uint8)
    return mask_np.astype(np.uint8)


def mask_tensor_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor mask to binary NumPy array [H,W].
    """
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    mask_np = tensor_to_numpy_binary(mask_tensor.unsqueeze(0))
    return squeeze_mask(mask_np[0])


# ============================================================
# METRICS AND ERROR MAPS
# ============================================================

def patch_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict:
    """
    Compute patch-level metrics used only for ranking qualitative candidates.

    These are not the final thesis quantitative metrics.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
    }


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Create RGB error map.

    Green = true positives
    Red   = false positives
    Blue  = false negatives
    Black = true negatives/background
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    h, w = gt.shape
    error = np.zeros((h, w, 3), dtype=np.float32)

    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)

    error[tp] = np.array([0.0, 1.0, 0.0])
    error[fp] = np.array([1.0, 0.0, 0.0])
    error[fn] = np.array([0.0, 0.0, 1.0])

    return error


@torch.no_grad()
def predict_all_models(models: dict, images: torch.Tensor, device: torch.device) -> dict:
    """
    Run inference for all models on a batch.

    For multi-head models, only the mask head is used, because this visual
    analysis compares final road segmentation outputs.
    """
    images = images.to(device, non_blocking=True)
    predictions = {}

    for model_name, item in models.items():
        model = item["model"]
        model_type = item["type"]

        if model_type == "single":
            logits = model(images)
        else:
            outputs = model(images)
            logits = outputs["mask"]

        pred = logits_to_binary(logits, threshold=THRESHOLD)
        predictions[model_name] = pred.detach().cpu()

    return predictions


# ============================================================
# FIGURE SAVING
# ============================================================

def plot_patch_comparison_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: dict,
    metadata: dict,
    title: str,
    save_path: Path,
):
    """
    Save a 128x128 patch-level comparison.

    Layout:
        Input RGB
        RGB + GT overlay
        GT mask
        Error map for each model
    """
    model_names = list(pred_masks.keys())
    n_cols = 3 + len(model_names)

    gt_overlay = overlay_mask_on_rgb(
        (image * 255).astype(np.uint8),
        gt_mask,
        color=(1.0, 1.0, 0.0),
        alpha=0.60,
    )

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input RGB", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("RGB + GT overlay", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("GT Mask", fontsize=11)
    axes[2].axis("off")

    for j, model_name in enumerate(model_names):
        pred = pred_masks[model_name]
        error_map = create_error_map(pred, gt_mask)

        axes[j + 3].imshow(error_map)
        axes[j + 3].set_title(model_name, fontsize=10)
        axes[j + 3].axis("off")

    subtitle = (
        f"{title}\n"
        f"Mosaic: {metadata['image_name']} | "
        f"x={metadata['x']}, y={metadata['y']} | "
        f"global_index={metadata['global_index']}"
    )
    fig.suptitle(subtitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_context_figure(
    metadata: dict,
    title: str,
    save_path: Path,
):
    """
    Save a larger-context visualization around the selected patch.

    This figure does not contain model predictions. Its purpose is to show:
        - the surrounding RGB context,
        - the ground-truth annotation overlay,
        - the position of the evaluated 128x128 patch.
    """
    full_rgb = read_full_rgb(metadata["image_name"])
    full_mask = read_full_mask(metadata["image_name"])

    context = extract_context_crop(
        full_image=full_rgb,
        full_mask=full_mask,
        patch_x=metadata["x"],
        patch_y=metadata["y"],
        patch_size=metadata["patch_size"],
        context_size=CONTEXT_SIZE,
    )

    rgb_context = context["context_rgb"]
    mask_context = context["context_mask"]
    overlay_context = overlay_mask_on_rgb(
        rgb_context,
        mask_context,
        color=(1.0, 1.0, 0.0),
        alpha=0.55,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Context RGB
    axes[0].imshow(rgb_context)
    axes[0].set_title("Larger RGB context", fontsize=11)
    axes[0].axis("off")

    # Context RGB + GT overlay
    axes[1].imshow(overlay_context)
    axes[1].set_title("Context + GT overlay", fontsize=11)
    axes[1].axis("off")

    # Context with patch rectangle
    axes[2].imshow(overlay_context)
    axes[2].add_patch(
        Rectangle(
            (context["rect_x"], context["rect_y"]),
            context["rect_w"],
            context["rect_h"],
            linewidth=2.5,
            edgecolor="cyan",
            facecolor="none",
        )
    )
    axes[2].set_title("Evaluated 128x128 patch", fontsize=11)
    axes[2].axis("off")

    subtitle = (
        f"{title}\n"
        f"Mosaic: {metadata['image_name']} | "
        f"patch x={metadata['x']}, y={metadata['y']} | "
        f"context crop=({context['crop_x0']}, {context['crop_y0']})"
        f"-({context['crop_x1']}, {context['crop_y1']})"
    )
    fig.suptitle(subtitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_candidate(
    candidate: dict,
    category: str,
    rank: int,
    output_dir: Path,
):
    """
    Save both:
        1. patch-level model comparison,
        2. larger-context mosaic visualization.

    The filenames include mosaic name, x, y, batch, sample, and score.
    """
    category_dir = output_dir / category
    patch_dir = category_dir / "patch_comparisons"
    context_dir = category_dir / "context_views"

    patch_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    batch_idx = candidate["batch_index"]
    sample_idx = candidate["sample_index"]
    score = candidate["score"]
    metadata = candidate["metadata"]

    image_stem = Path(metadata["image_name"]).stem
    x = metadata["x"]
    y = metadata["y"]

    base_name = (
        f"{rank:02d}_{category}_"
        f"{image_stem}_x{x}_y{y}_"
        f"b{batch_idx}_s{sample_idx}_score{score:.3f}"
    )

    patch_save_path = patch_dir / f"{base_name}_patch.png"
    context_save_path = context_dir / f"{base_name}_context.png"

    title = f"{category} | score={score:.3f}"

    plot_patch_comparison_figure(
        image=candidate["image"],
        gt_mask=candidate["gt_mask"],
        pred_masks=candidate["pred_masks"],
        metadata=metadata,
        title=title,
        save_path=patch_save_path,
    )

    plot_context_figure(
        metadata=metadata,
        title=title,
        save_path=context_save_path,
    )


def save_scores_csv(candidates: list[dict], output_path: Path):
    """
    Save patch metadata, paths, and metrics for all saved candidates.
    """
    if not candidates:
        return

    fieldnames = [
        "category",
        "rank",
        "batch_index",
        "sample_index",
        "global_index",
        "image_name",
        "x",
        "y",
        "patch_size",
        "full_image_path",
        "full_mask_path",
        "score",
        "gt_pixels",
        "single_precision",
        "single_recall",
        "single_dice",
        "equal_precision",
        "equal_recall",
        "equal_dice",
        "aux05_precision",
        "aux05_recall",
        "aux05_dice",
        "junction_precision",
        "junction_recall",
        "junction_dice",
        "center_precision",
        "center_recall",
        "center_dice",
        "single_fp",
        "equal_fp",
        "aux05_fp",
        "junction_fp",
        "center_fp",
        "single_fn",
        "equal_fn",
        "aux05_fn",
        "junction_fn",
        "center_fn",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cand in candidates:
            row = {
                "category": cand.get("category", ""),
                "rank": cand.get("rank", ""),
                "batch_index": cand.get("batch_index", ""),
                "sample_index": cand.get("sample_index", ""),
                "global_index": cand["metadata"]["global_index"],
                "image_name": cand["metadata"]["image_name"],
                "x": cand["metadata"]["x"],
                "y": cand["metadata"]["y"],
                "patch_size": cand["metadata"]["patch_size"],
                "full_image_path": cand["metadata"]["full_image_path"],
                "full_mask_path": cand["metadata"]["full_mask_path"],
                "score": cand.get("score", ""),
                "gt_pixels": cand.get("gt_pixels", ""),
                "single_precision": cand.get("single_precision", ""),
                "single_recall": cand.get("single_recall", ""),
                "single_dice": cand.get("single_dice", ""),
                "equal_precision": cand.get("equal_precision", ""),
                "equal_recall": cand.get("equal_recall", ""),
                "equal_dice": cand.get("equal_dice", ""),
                "aux05_precision": cand.get("aux05_precision", ""),
                "aux05_recall": cand.get("aux05_recall", ""),
                "aux05_dice": cand.get("aux05_dice", ""),
                "junction_precision": cand.get("junction_precision", ""),
                "junction_recall": cand.get("junction_recall", ""),
                "junction_dice": cand.get("junction_dice", ""),
                "center_precision": cand.get("center_precision", ""),
                "center_recall": cand.get("center_recall", ""),
                "center_dice": cand.get("center_dice", ""),
                "single_fp": cand.get("single_fp", ""),
                "equal_fp": cand.get("equal_fp", ""),
                "aux05_fp": cand.get("aux05_fp", ""),
                "junction_fp": cand.get("junction_fp", ""),
                "center_fp": cand.get("center_fp", ""),
                "single_fn": cand.get("single_fn", ""),
                "equal_fn": cand.get("equal_fn", ""),
                "aux05_fn": cand.get("aux05_fn", ""),
                "junction_fn": cand.get("junction_fn", ""),
                "center_fn": cand.get("center_fn", ""),
            }
            writer.writerow(row)


# ============================================================
# MAIN SEARCH LOGIC
# ============================================================

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")

    models = load_all_models(device)
    test_loader = get_test_loader(device)

    helpful_candidates = []
    noisy_candidates = []
    balanced_candidates = []

    print("Scanning test set...")

    for batch_index, batch in enumerate(test_loader):
        images, masks, centers, junctions = batch

        predictions = predict_all_models(models, images, device)

        for sample_index in range(images.shape[0]):
            metadata = get_patch_metadata(
                test_loader=test_loader,
                batch_index=batch_index,
                sample_index=sample_index,
            )

            image_np = tensor_to_display_image(images[sample_index])
            gt_mask = mask_tensor_to_numpy(masks[sample_index])

            if gt_mask.sum() < MIN_GT_PIXELS:
                continue

            pred_masks = {}
            metrics = {}

            for model_name, pred_tensor in predictions.items():
                pred_np = mask_tensor_to_numpy(pred_tensor[sample_index])
                pred_masks[model_name] = pred_np
                metrics[model_name] = patch_metrics(pred_np, gt_mask)

            single = metrics["UNetSingle"]
            equal = metrics["MultiHead equal"]
            aux05 = metrics["MultiHead aux05"]
            junction = metrics["MultiHead junction-focus"]
            center = metrics["MultiHead center-focus"]

            # Helpful recovery now focuses on aux05, because aux05 had
            # the highest mean recall in the quantitative pixel-wise results.
            helpful_score = (
                aux05["recall"] - single["recall"]
                + 0.25 * (aux05["dice"] - single["dice"])
            )

            noisy_score = (
                (equal["fp"] - single["fp"]) / max(1, gt_mask.sum())
                + 0.5 * (equal["recall"] - single["recall"])
            )

            balanced_score = (
                (equal["fp"] - center["fp"]) / max(1, gt_mask.sum())
                + 0.5 * (center["recall"] - single["recall"])
            )

            base_candidate = {
                "batch_index": batch_index,
                "sample_index": sample_index,
                "metadata": metadata,
                "image": image_np,
                "gt_mask": gt_mask,
                "pred_masks": pred_masks,
                "gt_pixels": int(gt_mask.sum()),
                "single_precision": single["precision"],
                "single_recall": single["recall"],
                "single_dice": single["dice"],
                "equal_precision": equal["precision"],
                "equal_recall": equal["recall"],
                "equal_dice": equal["dice"],
                "aux05_precision": aux05["precision"],
                "aux05_recall": aux05["recall"],
                "aux05_dice": aux05["dice"],
                "junction_precision": junction["precision"],
                "junction_recall": junction["recall"],
                "junction_dice": junction["dice"],
                "center_precision": center["precision"],
                "center_recall": center["recall"],
                "center_dice": center["dice"],
                "single_fp": single["fp"],
                "equal_fp": equal["fp"],
                "aux05_fp": aux05["fp"],
                "junction_fp": junction["fp"],
                "center_fp": center["fp"],
                "single_fn": single["fn"],
                "equal_fn": equal["fn"],
                "aux05_fn": aux05["fn"],
                "junction_fn": junction["fn"],
                "center_fn": center["fn"],
            }

            helpful_candidates.append({
                **base_candidate,
                "category": "helpful_recovery",
                "score": helpful_score,
            })

            noisy_candidates.append({
                **base_candidate,
                "category": "noisy_tradeoff",
                "score": noisy_score,
            })

            balanced_candidates.append({
                **base_candidate,
                "category": "balanced_weighting",
                "score": balanced_score,
            })

        if batch_index % 20 == 0:
            print(f"Processed batch {batch_index}")

    helpful_candidates = sorted(helpful_candidates, key=lambda x: x["score"], reverse=True)
    noisy_candidates = sorted(noisy_candidates, key=lambda x: x["score"], reverse=True)
    balanced_candidates = sorted(balanced_candidates, key=lambda x: x["score"], reverse=True)

    print("Saving candidate figures...")

    all_saved_rows = []

    for category, candidates in [
        ("helpful_recovery", helpful_candidates),
        ("noisy_tradeoff", noisy_candidates),
        ("balanced_weighting", balanced_candidates),
    ]:
        top_candidates = candidates[:TOP_K]

        for rank, cand in enumerate(top_candidates, start=1):
            cand["rank"] = rank
            save_candidate(cand, category, rank, OUTPUT_DIR)
            all_saved_rows.append(cand)

    csv_path = OUTPUT_DIR / "candidate_scores.csv"
    save_scores_csv(all_saved_rows, csv_path)

    print(f"Done. Candidate figures saved in: {OUTPUT_DIR}")
    print(f"Candidate scores saved to: {csv_path}")


if __name__ == "__main__":
    main()