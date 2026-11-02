import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD
from .utils import overlay_mask
from .xai import (
    extract_feature_maps,
    generate_gradcam,
    generate_scorecam,
    overlay_gradcam,
    save_feature_map_grid,
    select_gradcam_layer,
)


def _forward_model(model, images, tabular=None, return_features: bool = False):
    uses_tabular = hasattr(model, "vision_model") or getattr(model, "tabular_dim", 0) > 0
    if tabular is not None and uses_tabular:
        return model(images, tabular, return_features=return_features)
    return model(images, return_features=return_features)


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    if set(np.unique(mask).tolist()).issubset({0, 255}):
        mask = (mask > 0).astype(np.uint8)
    return np.clip(mask, 0, 4).astype(np.uint8)


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGE_STD + IMAGE_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _ensure_output_path(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_dataset_debug_grid(
    metadata_df: pd.DataFrame,
    dataset_root: Path,
    output_path: Path,
    num_samples: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    _ensure_output_path(output_path)

    rng = random.Random(seed)
    indices = rng.sample(range(len(metadata_df)), k=min(num_samples, len(metadata_df)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    sampled_tiles = []
    alignment_ok = True
    for row_idx, sample_idx in enumerate(indices):
        row = metadata_df.iloc[sample_idx]
        sampled_tiles.append(row["tile_id"])

        pre_image = _load_rgb(dataset_root / row["pre_image_path"])
        post_image = _load_rgb(dataset_root / row["post_image_path"])
        post_mask = _load_mask(dataset_root / row["post_mask_path"])

        if pre_image.shape[:2] != post_image.shape[:2] or post_image.shape[:2] != post_mask.shape[:2]:
            alignment_ok = False

        axes[row_idx, 0].imshow(pre_image)
        axes[row_idx, 0].set_title(f'{row["tile_id"]}\npre')
        axes[row_idx, 1].imshow(post_image)
        axes[row_idx, 1].set_title("post")
        axes[row_idx, 2].imshow(post_mask, cmap="magma", vmin=0, vmax=4)
        axes[row_idx, 2].set_title("post mask")

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "sampled_tiles": sampled_tiles,
        "alignment_ok": alignment_ok,
        "output_path": str(output_path),
    }


def run_model_sanity_check(model, dataset, device: torch.device, sample_index: int = 0) -> Dict[str, object]:
    model.eval()
    images, tabular, mask = dataset[sample_index]
    images = images.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _forward_model(model, images, tabular)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

    predictions = outputs.argmax(dim=1)
    logits_std = float(outputs.std().item())
    prediction_unique = torch.unique(predictions).cpu().tolist()

    return {
        "output_shape": tuple(outputs.shape),
        "mask_shape": tuple(mask.shape),
        "shape_matches": tuple(outputs.shape[-2:]) == tuple(mask.shape[-2:]),
        "has_nan": bool(torch.isnan(outputs).any().item()),
        "is_constant_output": bool(logits_std < 1e-8),
        "logits_min": float(outputs.min().item()),
        "logits_max": float(outputs.max().item()),
        "logits_mean": float(outputs.mean().item()),
        "logits_std": logits_std,
        "prediction_classes": [int(value) for value in prediction_unique],
    }


def validate_siamese_fusion(model, dataset, device: torch.device, sample_index: int = 0) -> Dict[str, object]:
    model.eval()
    images, tabular, _ = dataset[sample_index]
    images = images.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _forward_model(model, images, tabular, return_features=True)

    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise ValueError("Model does not expose intermediate features for fusion validation.")

    logits, feature_dict = outputs
    pre_encoder = feature_dict["pre_encoder"]
    post_encoder = feature_dict["post_encoder"]
    fused = feature_dict["fused"]
    decoder = feature_dict["decoder"]

    fusion_checks = {}
    for level in pre_encoder.keys():
        pre_shape = tuple(pre_encoder[level].shape)
        post_shape = tuple(post_encoder[level].shape)
        fused_shape = tuple(fused[level].shape)
        fusion_checks[level] = {
            "pre_shape": pre_shape,
            "post_shape": post_shape,
            "fused_shape": fused_shape,
            "encoder_shapes_match": pre_shape == post_shape,
            "fused_channels_expected": fused_shape[1] == pre_shape[1] * 2,
        }

    return {
        "logits_shape": tuple(logits.shape),
        "fusion_checks": fusion_checks,
        "decoder_shapes": {name: tuple(tensor.shape) for name, tensor in decoder.items()},
        "bottleneck_shape": tuple(feature_dict["bottleneck"].shape),
    }


def save_prediction_panel(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    output_path: Path,
    title: str = "",
):
    output_path = Path(output_path)
    _ensure_output_path(output_path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(pre_image)
    axes[0].set_title("Pre")
    axes[1].imshow(post_image)
    axes[1].set_title("Post")
    axes[2].imshow(overlay_mask(post_image, ground_truth_mask, alpha=0.55))
    axes[2].set_title("Ground Truth")
    axes[3].imshow(overlay_mask(post_image, predicted_mask, alpha=0.55))
    axes[3].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_epoch_visualization(
    model,
    dataset,
    device: torch.device,
    output_path: Path,
    sample_indices: Optional[Sequence[int]] = None,
    max_samples: int = 4,
) -> Path:
    output_path = Path(output_path)
    _ensure_output_path(output_path)

    if sample_indices is None:
        sample_indices = list(range(min(max_samples, len(dataset))))

    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()
    for row_idx, sample_index in enumerate(sample_indices):
        image_tensor, tabular_tensor, mask_tensor = dataset[sample_index]
        images = image_tensor.unsqueeze(0).to(device)
        tabular = tabular_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = _forward_model(model, images, tabular)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predicted_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pre_image = _denormalize_image(image_tensor[:3])
        post_image = _denormalize_image(image_tensor[3:])
        ground_truth_mask = mask_tensor.cpu().numpy().astype(np.uint8)

        axes[row_idx, 0].imshow(pre_image)
        axes[row_idx, 0].set_title(f"Sample {sample_index}\nPre")
        axes[row_idx, 1].imshow(post_image)
        axes[row_idx, 1].set_title("Post")
        axes[row_idx, 2].imshow(overlay_mask(post_image, ground_truth_mask, alpha=0.55))
        axes[row_idx, 2].set_title("Ground Truth")
        axes[row_idx, 3].imshow(overlay_mask(post_image, predicted_mask, alpha=0.55))
        axes[row_idx, 3].set_title("Prediction")

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_explainability_visuals(
    model,
    dataset,
    device: torch.device,
    output_dir: Path,
    sample_index: int = 0,
    use_scorecam: bool = True,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_tensor, tabular_tensor, mask_tensor = dataset[sample_index]
    images = image_tensor.unsqueeze(0).to(device)
    tabular = tabular_tensor.unsqueeze(0).to(device)
    post_image = _denormalize_image(image_tensor[3:])
    ground_truth_mask = mask_tensor.cpu().numpy().astype(np.uint8)

    with torch.no_grad():
        outputs = _forward_model(model, images, tabular)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        predicted_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    prediction_panel_path = output_dir / f"sample_{sample_index:03d}_prediction.png"
    save_prediction_panel(
        _denormalize_image(image_tensor[:3]),
        post_image,
        ground_truth_mask,
        predicted_mask,
        prediction_panel_path,
        title=f"Sample {sample_index}",
    )

    target_layer = select_gradcam_layer(model)
    gradcam_map = generate_gradcam(model, target_layer, images, tabular_tensor=tabular)
    gradcam_path = output_dir / f"sample_{sample_index:03d}_gradcam.png"
    cv2.imwrite(
        str(gradcam_path),
        cv2.cvtColor(overlay_gradcam(post_image, gradcam_map), cv2.COLOR_RGB2BGR),
    )

    scorecam_path = ""
    if use_scorecam:
        scorecam_map = generate_scorecam(model, target_layer, images, tabular_tensor=tabular)
        scorecam_path = str(output_dir / f"sample_{sample_index:03d}_scorecam.png")
        cv2.imwrite(
            scorecam_path,
            cv2.cvtColor(overlay_gradcam(post_image, scorecam_map), cv2.COLOR_RGB2BGR),
        )

    feature_paths = {}
    feature_maps = extract_feature_maps(model, images, tabular)
    for group_name, group_value in feature_maps.items():
        if isinstance(group_value, dict):
            for tensor_name, tensor_value in group_value.items():
                feature_path = output_dir / f"sample_{sample_index:03d}_{group_name}_{tensor_name}.png"
                save_feature_map_grid(tensor_value, feature_path, title=f"{group_name}.{tensor_name}")
                feature_paths[f"{group_name}.{tensor_name}"] = str(feature_path)
        elif isinstance(group_value, torch.Tensor):
            feature_path = output_dir / f"sample_{sample_index:03d}_{group_name}.png"
            save_feature_map_grid(group_value, feature_path, title=group_name)
            feature_paths[group_name] = str(feature_path)

    return {
        "prediction_panel": str(prediction_panel_path),
        "gradcam": str(gradcam_path),
        "scorecam": scorecam_path,
        **feature_paths,
    }
