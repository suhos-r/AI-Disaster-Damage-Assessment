from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_target_category(logits: torch.Tensor) -> int:
    probabilities = torch.softmax(logits.detach(), dim=1)
    class_scores = probabilities.mean(dim=(0, 2, 3))

    if class_scores.numel() > 1:
        foreground_scores = class_scores[1:]
        if torch.any(foreground_scores > 0):
            return int(torch.argmax(foreground_scores).item() + 1)
    return int(torch.argmax(class_scores).item())


class SemanticSegmentationTarget:
    def __init__(self, category: Optional[int] = None):
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        category = self.category
        if category is None:
            category = _infer_target_category(model_output)

        category_logits = model_output[:, category, :, :]
        predicted_mask = model_output.argmax(dim=1) == category

        if predicted_mask.any():
            return category_logits[predicted_mask].mean()
        return category_logits.mean()


def select_gradcam_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_gradcam_target_layer"):
        return model.get_gradcam_target_layer()

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("Unable to find a convolutional layer for Grad-CAM.")


def generate_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    tabular_tensor: Optional[torch.Tensor] = None,
    target_category: Optional[int] = None,
) -> np.ndarray:
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0])

    was_training = model.training
    model.eval()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        image_tensor = input_tensor.clone().detach().requires_grad_(True)

        if tabular_tensor is not None:
            logits = model(image_tensor, tabular_tensor)
        else:
            logits = model(image_tensor)

        scalar_target = SemanticSegmentationTarget(target_category)(logits)
        scalar_target.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activation_map = activations[-1]
        gradient_map = gradients[-1]

        weights = gradient_map.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()

        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam.astype(np.float32)
    finally:
        forward_handle.remove()
        backward_handle.remove()
        if was_training:
            model.train()


def generate_scorecam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    tabular_tensor: Optional[torch.Tensor] = None,
    target_category: Optional[int] = None,
    max_channels: int = 8,
) -> np.ndarray:
    activations = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    was_training = model.training
    model.eval()
    handle = target_layer.register_forward_hook(forward_hook)

    try:
        with torch.no_grad():
            if tabular_tensor is not None:
                baseline_logits = model(input_tensor, tabular_tensor)
            else:
                baseline_logits = model(input_tensor)

        if not activations:
            raise RuntimeError("Score-CAM could not capture the target-layer activations.")

        activation_map = activations[-1]
        channel_scores = activation_map[0].mean(dim=(1, 2))
        num_channels = min(max_channels, activation_map.shape[1])
        selected_channels = torch.topk(channel_scores, k=num_channels).indices.tolist()

        score_cam = torch.zeros(
            input_tensor.shape[-2:],
            device=input_tensor.device,
            dtype=torch.float32,
        )

        for channel_index in selected_channels:
            channel_map = activation_map[:, channel_index : channel_index + 1, :, :]
            channel_map = F.interpolate(
                channel_map,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            channel_map = channel_map - channel_map.amin(dim=(2, 3), keepdim=True)
            channel_map = channel_map / channel_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)

            masked_input = input_tensor * channel_map
            with torch.no_grad():
                if tabular_tensor is not None:
                    masked_logits = model(masked_input, tabular_tensor)
                else:
                    masked_logits = model(masked_input)

            if target_category is None:
                current_category = _infer_target_category(baseline_logits)
            else:
                current_category = target_category

            score = torch.softmax(masked_logits, dim=1)[:, current_category, :, :].mean()
            score_cam += score * channel_map[0, 0]

        score_cam = F.relu(score_cam)
        score_cam = score_cam.detach().cpu().numpy()
        score_cam -= score_cam.min()
        score_cam /= max(score_cam.max(), 1e-8)
        return score_cam.astype(np.float32)
    finally:
        handle.remove()
        if was_training:
            model.train()


def overlay_gradcam(
    image: np.ndarray,
    grayscale_cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    heatmap = cv2.resize(
        grayscale_cam.astype(np.float32),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap -= heatmap.min()
    heatmap /= max(heatmap.max(), 1e-8)

    colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = np.clip((1.0 - alpha) * image + alpha * colored_heatmap, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def explain_tabular(model: nn.Module, feature_names=None):
    if hasattr(model, "vision_model"):
        model = model.vision_model

    if hasattr(model, "tabular_film") and isinstance(model.tabular_film, nn.Sequential):
        first_linear = next(
            (layer for layer in model.tabular_film if isinstance(layer, nn.Linear)),
            None,
        )
        if first_linear is None:
            return {}

        weights = first_linear.weight.detach().abs().mean(dim=0).cpu().numpy()
        importance = weights / max(weights.sum(), 1e-8)

        if feature_names is None:
            feature_names = [f"Feature {index}" for index in range(len(importance))]

        return dict(zip(feature_names, importance.tolist()))

    return {}


def extract_feature_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    tabular_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if hasattr(model, "forward"):
        try:
            if tabular_tensor is not None:
                _, features = model(input_tensor, tabular_tensor, return_features=True)
            else:
                _, features = model(input_tensor, return_features=True)
        except TypeError as exc:
            raise ValueError("Model does not expose return_features support.") from exc
        return features

    raise ValueError("Unable to extract feature maps from the provided model.")


def save_feature_map_grid(
    feature_tensor: torch.Tensor,
    output_path: Path,
    title: str,
    max_channels: int = 8,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_tensor = feature_tensor.detach().cpu()
    feature_tensor = feature_tensor[0]
    channel_count = min(max_channels, feature_tensor.shape[0])

    fig, axes = plt.subplots(1, channel_count, figsize=(3 * channel_count, 3))
    if channel_count == 1:
        axes = [axes]

    for axis, channel_index in zip(axes, range(channel_count)):
        channel_map = feature_tensor[channel_index].numpy()
        channel_map -= channel_map.min()
        channel_map /= max(channel_map.max(), 1e-8)
        axis.imshow(channel_map, cmap="viridis")
        axis.set_title(f"{title}\nch={channel_index}")
        axis.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
