from __future__ import annotations

"""
generate_connectivity_fragmentation_candidates_with_aux05.py

Purpose
-------
Search the held-out TEST set for qualitative examples that explain the
connectivity and fragmentation results of the thesis.

This script does NOT train models and does NOT change the quantitative
evaluation protocol.

It scans the same 128x128 test patches used during evaluation and saves
figures that highlight connected components in the predicted road masks.

Visual meaning
--------------
For each model prediction:
    - Each disconnected predicted road component is shown with a different color.
    - Black/background means no predicted road.
    - The panel title reports:
        number of connected components,
        LCC ratio,
        FP,
        FN.

This is useful for explaining why:
    - UNetSingle is less fragmented.
    - MultiHead equal can predict more disconnected road-like regions.
    - center-focus can reduce fragmentation compared with equal weighting.
    - high LCC ratio does not always mean better topology.

Candidate folders
-----------------
1. fragmentation_tradeoff
   Cases where MultiHead equal has more components than UNetSingle.

2. center_focus_improves_fragmentation
   Cases where center-focus reduces fragmentation compared with equal weighting.

3. lcc_caution
   Cases where aux05 has a high LCC ratio but also many false positives.
"""

from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from scipy.ndimage import label as connected_component_label

from dataset_awr_seg_mosaic_split import get_dataloaders
from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead
from evaluation_utils import logits_to_binary, tensor_to_numpy_binary


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"

SEED = 42
# SEED = 123
# SEED = 2025

OUTPUT_DIR = Path(f"conn_vis_s{SEED}")

BATCH_SIZE = 8
VAL_FRACTION = 0.10
NUM_WORKERS = 4
PIN_MEMORY = True
THRESHOLD = 0.5

TOP_K = 12

# Keep this high enough to avoid patches with almost no road annotation.
MIN_GT_PIXELS = 500

# Optional: avoid completely empty predictions.
MIN_PRED_PIXELS = 30

CONTEXT_SIZE = 512
PATCH_SIZE = 128

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    "MultiHead center-focus": {
        "type": "multihead",
        "path": multihead_weighted_ckpt(SEED, "center_focus"),
    },
    "MultiHead junction-focus": {
        "type": "multihead",
        "path": multihead_weighted_ckpt(SEED, "junction_focus"),
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_all_models(device: torch.device) -> dict:
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
    path = Path(DATA_DIR) / "image_png" / image_name
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_full_mask(image_name: str) -> np.ndarray:
    path = Path(DATA_DIR) / "mask" / image_name
    mask = Image.open(path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def get_patch_metadata(test_loader, batch_index: int, sample_index: int) -> dict:
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
    h, w = full_mask.shape

    patch_center_x = patch_x + patch_size // 2
    patch_center_y = patch_y + patch_size // 2
    half = context_size // 2

    x0 = max(0, patch_center_x - half)
    y0 = max(0, patch_center_y - half)
    x1 = min(w, x0 + context_size)
    y1 = min(h, y0 + context_size)

    x0 = max(0, x1 - context_size)
    y0 = max(0, y1 - context_size)

    context_rgb = full_image[y0:y1, x0:x1, :]
    context_mask = full_mask[y0:y1, x0:x1]

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
    image = image_tensor.detach().cpu().numpy()

    if image.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got {image.shape}")

    image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)

    return image


def squeeze_mask(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.ndim == 3:
        return mask_np[0].astype(np.uint8)
    return mask_np.astype(np.uint8)


def mask_tensor_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    mask_np = tensor_to_numpy_binary(mask_tensor.unsqueeze(0))
    return squeeze_mask(mask_np[0])


# ============================================================
# METRICS AND CONNECTED COMPONENTS
# ============================================================

def patch_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict:
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


def connected_component_metrics(pred: np.ndarray) -> dict:
    """
    Compute patch-level connected component metrics.

    Connectivity is 8-connectivity, matching the thesis description.
    """
    pred_bool = pred.astype(bool)

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num_components = connected_component_label(pred_bool, structure=structure)

    pred_pixels = int(pred_bool.sum())

    if pred_pixels == 0 or num_components == 0:
        return {
            "num_components": 0,
            "lcc_pixels": 0,
            "lcc_ratio": 0.0,
            "labeled": labeled.astype(np.int32),
        }

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    lcc_pixels = int(component_sizes.max())
    lcc_ratio = float(lcc_pixels / max(1, pred_pixels))

    return {
        "num_components": int(num_components),
        "lcc_pixels": lcc_pixels,
        "lcc_ratio": lcc_ratio,
        "labeled": labeled.astype(np.int32),
    }


def component_label_to_rgb(labeled: np.ndarray) -> np.ndarray:
    """
    Convert connected-component label image to RGB.

    Background is black. Each component receives a different color.
    This is only for visualization.
    """
    labeled = labeled.astype(np.int32)
    h, w = labeled.shape

    rgb = np.zeros((h, w, 3), dtype=np.float32)

    max_label = int(labeled.max())
    if max_label == 0:
        return rgb

    cmap = plt.get_cmap("tab20", max_label + 1)

    for component_id in range(1, max_label + 1):
        rgb[labeled == component_id] = cmap(component_id)[:3]

    return rgb


@torch.no_grad()
def predict_all_models(models: dict, images: torch.Tensor, device: torch.device) -> dict:
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

def plot_patch_connectivity_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: dict,
    metrics: dict,
    component_metrics: dict,
    metadata: dict,
    title: str,
    save_path: Path,
):
    """
    Save a patch-level connectivity comparison.

    Layout:
        Input RGB
        RGB + GT overlay
        GT mask
        Connected component maps for each model
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
    axes[2].set_title(f"GT mask\npixels={int(gt_mask.sum())}", fontsize=11)
    axes[2].axis("off")

    for j, model_name in enumerate(model_names):
        comp = component_metrics[model_name]
        met = metrics[model_name]

        comp_rgb = component_label_to_rgb(comp["labeled"])

        axes[j + 3].imshow(comp_rgb)
        axes[j + 3].set_title(
            f"{model_name}\n"
            f"Comp={comp['num_components']} | "
            f"LCC={comp['lcc_ratio']:.2f}\n"
            f"FP={met['fp']} | FN={met['fn']}",
            fontsize=9,
        )
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

    axes[0].imshow(rgb_context)
    axes[0].set_title("Larger RGB context", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay_context)
    axes[1].set_title("Context + GT overlay", fontsize=11)
    axes[1].axis("off")

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

    patch_save_path = patch_dir / f"{base_name}_components.png"
    context_save_path = context_dir / f"{base_name}_context.png"

    title = f"{category} | score={score:.3f}"

    plot_patch_connectivity_figure(
        image=candidate["image"],
        gt_mask=candidate["gt_mask"],
        pred_masks=candidate["pred_masks"],
        metrics=candidate["metrics"],
        component_metrics=candidate["component_metrics"],
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
        "score",
        "gt_pixels",

        "single_components",
        "equal_components",
        "aux05_components",
        "center_components",
        "junction_components",

        "single_lcc",
        "equal_lcc",
        "aux05_lcc",
        "center_lcc",
        "junction_lcc",

        "single_precision",
        "single_recall",
        "single_dice",
        "equal_precision",
        "equal_recall",
        "equal_dice",
        "aux05_precision",
        "aux05_recall",
        "aux05_dice",
        "center_precision",
        "center_recall",
        "center_dice",
        "junction_precision",
        "junction_recall",
        "junction_dice",

        "single_fp",
        "equal_fp",
        "aux05_fp",
        "center_fp",
        "junction_fp",

        "single_fn",
        "equal_fn",
        "aux05_fn",
        "center_fn",
        "junction_fn",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cand in candidates:
            row = {
                "category": cand.get("category", ""),
                "rank": cand.get("rank", ""),
                "batch_index": cand["batch_index"],
                "sample_index": cand["sample_index"],
                "global_index": cand["metadata"]["global_index"],
                "image_name": cand["metadata"]["image_name"],
                "x": cand["metadata"]["x"],
                "y": cand["metadata"]["y"],
                "patch_size": cand["metadata"]["patch_size"],
                "score": cand["score"],
                "gt_pixels": cand["gt_pixels"],
            }

            name_map = {
                "single": "UNetSingle",
                "equal": "MultiHead equal",
                "aux05": "MultiHead aux05",
                "center": "MultiHead center-focus",
                "junction": "MultiHead junction-focus",
            }

            for short, full in name_map.items():
                row[f"{short}_components"] = cand["component_metrics"][full]["num_components"]
                row[f"{short}_lcc"] = cand["component_metrics"][full]["lcc_ratio"]

                row[f"{short}_precision"] = cand["metrics"][full]["precision"]
                row[f"{short}_recall"] = cand["metrics"][full]["recall"]
                row[f"{short}_dice"] = cand["metrics"][full]["dice"]
                row[f"{short}_fp"] = cand["metrics"][full]["fp"]
                row[f"{short}_fn"] = cand["metrics"][full]["fn"]

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

    fragmentation_candidates = []
    center_focus_candidates = []
    lcc_caution_candidates = []

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
            component_metrics = {}

            for model_name, pred_tensor in predictions.items():
                pred_np = mask_tensor_to_numpy(pred_tensor[sample_index])

                if pred_np.sum() < MIN_PRED_PIXELS:
                    continue

                pred_masks[model_name] = pred_np
                metrics[model_name] = patch_metrics(pred_np, gt_mask)
                component_metrics[model_name] = connected_component_metrics(pred_np)

            required_models = set(MODEL_CONFIGS.keys())
            if set(pred_masks.keys()) != required_models:
                continue

            single = metrics["UNetSingle"]
            equal = metrics["MultiHead equal"]
            aux05 = metrics["MultiHead aux05"]
            center = metrics["MultiHead center-focus"]
            junction = metrics["MultiHead junction-focus"]

            single_c = component_metrics["UNetSingle"]
            equal_c = component_metrics["MultiHead equal"]
            aux05_c = component_metrics["MultiHead aux05"]
            center_c = component_metrics["MultiHead center-focus"]
            junction_c = component_metrics["MultiHead junction-focus"]

            gt_pixels = max(1, int(gt_mask.sum()))

            # ------------------------------------------------------------
            # 1. Fragmentation trade-off
            #
            # Select cases where MultiHead equal predicts more disconnected
            # components than UNetSingle, ideally while recovering similar
            # or more road pixels.
            # ------------------------------------------------------------
            fragmentation_score = (
                (equal_c["num_components"] - single_c["num_components"])
                + 2.0 * (equal["recall"] - single["recall"])
                - 0.25 * ((equal["fp"] - single["fp"]) / gt_pixels)
            )

            # ------------------------------------------------------------
            # 2. Center-focus improves fragmentation
            #
            # Select cases where center-focus reduces the number of
            # components compared with equal weighting, while preserving
            # similar recall.
            # ------------------------------------------------------------
            center_focus_score = (
                (equal_c["num_components"] - center_c["num_components"])
                + 1.5 * (center["recall"] - single["recall"])
                - 0.20 * (center["fp"] / gt_pixels)
            )

            # ------------------------------------------------------------
            # 3. LCC caution
            #
            # Select cases where aux05 has a high LCC ratio but also many
            # false positives. This helps explain why high LCC is not
            # automatically better.
            # ------------------------------------------------------------
            lcc_caution_score = (
                2.0 * (aux05_c["lcc_ratio"] - single_c["lcc_ratio"])
                + 0.75 * ((aux05["fp"] - single["fp"]) / gt_pixels)
                + 0.5 * (aux05["recall"] - single["recall"])
            )

            base_candidate = {
                "batch_index": batch_index,
                "sample_index": sample_index,
                "metadata": metadata,
                "image": image_np,
                "gt_mask": gt_mask,
                "pred_masks": pred_masks,
                "metrics": metrics,
                "component_metrics": component_metrics,
                "gt_pixels": int(gt_mask.sum()),
            }

            fragmentation_candidates.append({
                **base_candidate,
                "category": "fragmentation_tradeoff",
                "score": float(fragmentation_score),
            })

            center_focus_candidates.append({
                **base_candidate,
                "category": "center_focus_improves_fragmentation",
                "score": float(center_focus_score),
            })

            lcc_caution_candidates.append({
                **base_candidate,
                "category": "lcc_caution",
                "score": float(lcc_caution_score),
            })

        if batch_index % 20 == 0:
            print(f"Processed batch {batch_index}")

    fragmentation_candidates = sorted(
        fragmentation_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )
    center_focus_candidates = sorted(
        center_focus_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )
    lcc_caution_candidates = sorted(
        lcc_caution_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    print("Saving candidate figures...")

    all_saved_rows = []

    for category, candidates in [
        ("fragmentation_tradeoff", fragmentation_candidates),
        ("center_focus_improves_fragmentation", center_focus_candidates),
        ("lcc_caution", lcc_caution_candidates),
    ]:
        top_candidates = candidates[:TOP_K]

        for rank, cand in enumerate(top_candidates, start=1):
            cand["rank"] = rank
            save_candidate(cand, category, rank, OUTPUT_DIR)
            all_saved_rows.append(cand)

    csv_path = OUTPUT_DIR / "connectivity_candidate_scores.csv"
    save_scores_csv(all_saved_rows, csv_path)

    print(f"Done. Candidate figures saved in: {OUTPUT_DIR}")
    print(f"Candidate scores saved to: {csv_path}")


if __name__ == "__main__":
    main()