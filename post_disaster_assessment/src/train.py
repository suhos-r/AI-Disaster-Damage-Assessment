import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import (
    DisasterAssessmentDataset,
    IMAGE_MEAN,
    IMAGE_STD,
    build_ida_metadata,
    get_train_transforms,
    get_val_transforms,
    split_metadata,
)
from .model import MultimodalDamageAssessment, SiameseUNet
from .utils import calculate_dice, calculate_iou, calculate_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        include_background: bool = True,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        if not self.include_background and dice_score.numel() > 1:
            dice_score = dice_score[1:]

        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = True,
        smooth: float = 1.0,
        ce_mode: str = "cross_entropy",
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes,
            include_background=include_background,
            smooth=smooth,
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        if ce_mode == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        elif ce_mode == "focal":
            self.base_loss = FocalLoss(gamma=focal_gamma, weight=self.class_weights)
        else:
            raise ValueError(f"Unsupported ce_mode '{ce_mode}'.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return self.ce_weight * base_loss + self.dice_weight * dice_loss


class DiceCrossEntropyLoss(CombinedSegmentationLoss):
    def __init__(
        self,
        num_classes: int = 5,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = True,
        smooth: float = 1.0,
    ):
        super().__init__(
            num_classes=num_classes,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            class_weights=class_weights,
            include_background=include_background,
            smooth=smooth,
            ce_mode="cross_entropy",
        )


def build_loss(
    num_classes: int = 5,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    include_background: bool = True,
    smooth: float = 1.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
):
    return CombinedSegmentationLoss(
        num_classes=num_classes,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
        class_weights=class_weights,
        include_background=include_background,
        smooth=smooth,
        ce_mode="focal" if use_focal else "cross_entropy",
        focal_gamma=focal_gamma,
    )


def run_segmentation_model(
    model: nn.Module,
    images: torch.Tensor,
    tabular: Optional[torch.Tensor] = None,
):
    uses_tabular = hasattr(model, "vision_model") or getattr(model, "tabular_dim", 0) > 0
    if tabular is not None and uses_tabular:
        return model(images, tabular)
    return model(images)


def _prepare_model_inputs(images: torch.Tensor, use_post_only_input: bool = False) -> torch.Tensor:
    if not use_post_only_input:
        return images
    post = images[:, 3:, :, :]
    return torch.cat([post, post], dim=1)


def estimate_class_weights(
    metadata_df,
    dataset_root,
    num_classes: int = 5,
) -> torch.Tensor:
    dataset_root = Path(dataset_root)
    histogram = torch.ones(num_classes, dtype=torch.float32)

    for _, row in metadata_df.iterrows():
        mask = cv2.imread(str(dataset_root / row["post_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if set(np.unique(mask).tolist()).issubset({0, 255}):
            mask = (mask > 0).astype(np.uint8)
        mask = np.clip(mask, 0, num_classes - 1)

        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            histogram[int(value)] += float(count)

    weights = histogram.sum() / histogram
    weights = weights / weights.mean()
    return weights


def _to_float_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _epoch_pass(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_post_only_input: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, tabular, masks in dataloader:
        images = _prepare_model_inputs(images.to(device), use_post_only_input=use_post_only_input)
        tabular = tabular.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = run_segmentation_model(model, images, tabular)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, masks)

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(masks.detach().cpu())

    epoch_loss = running_loss / max(len(dataloader.dataset), 1)
    predictions = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(predictions, targets)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "iou": calculate_iou(predictions, targets),
        "dice": calculate_dice(predictions, targets),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_post_only_input: bool = False):
    return _epoch_pass(
        model,
        dataloader,
        criterion,
        device,
        optimizer=optimizer,
        use_post_only_input=use_post_only_input,
    )


def validate(model, dataloader, criterion, device, use_post_only_input: bool = False):
    return _epoch_pass(
        model,
        dataloader,
        criterion,
        device,
        optimizer=None,
        use_post_only_input=use_post_only_input,
    )


@dataclass
class TrainingConfig:
    dataset_root: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_size: float = 0.2
    seed: int = 42
    use_multimodal: bool = True
    overfit_subset_size: int = 0
    use_weighted_ce: bool = False
    use_focal: bool = False
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    include_background: bool = True
    save_every: int = 1
    checkpoint_metric: str = "dice"
    num_classes: int = 5
    disable_augmentation: bool = False
    use_post_only_input: bool = False
    demo_outputs_dir: str = "artifacts/demo_outputs"
    demo_checkpoint_path: str = "checkpoints/demo_model.pth"


def build_model(config: TrainingConfig) -> nn.Module:
    if config.use_multimodal:
        return MultimodalDamageAssessment(tabular_features=4, out_classes=config.num_classes)
    return SiameseUNet(in_channels=3, out_classes=config.num_classes)


def prepare_dataloaders(config: TrainingConfig, metadata_df: Optional[pd.DataFrame] = None):
    dataset_root = Path(config.dataset_root)
    metadata_df = metadata_df if metadata_df is not None else build_ida_metadata(dataset_root)

    if config.overfit_subset_size > 0:
        subset_df = metadata_df.sample(
            n=min(config.overfit_subset_size, len(metadata_df)),
            random_state=config.seed,
        ).reset_index(drop=True)
        train_df = subset_df
        val_df = subset_df.copy()
    else:
        train_df, val_df = split_metadata(
            metadata_df,
            val_size=config.val_size,
            random_state=config.seed,
        )

    train_transform = get_train_transforms()
    if config.overfit_subset_size > 0 or config.disable_augmentation:
        train_transform = get_val_transforms()

    train_dataset = DisasterAssessmentDataset(
        train_df,
        dataset_root,
        transform=train_transform,
    )
    val_dataset = DisasterAssessmentDataset(
        val_df,
        dataset_root,
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_df, val_df, train_dataset, val_dataset, train_loader, val_loader


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def _denormalize_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGE_STD + IMAGE_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _save_demo_epoch_output(
    model: nn.Module,
    dataset: DisasterAssessmentDataset,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    use_post_only_input: bool = False,
):
    if len(dataset) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = (epoch - 1) % len(dataset)
    image_tensor, tabular_tensor, mask_tensor = dataset[sample_index]

    image_batch = _prepare_model_inputs(
        image_tensor.unsqueeze(0).to(device),
        use_post_only_input=use_post_only_input,
    )
    tabular_batch = tabular_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = run_segmentation_model(model, image_batch, tabular_batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    post_image = _denormalize_rgb(image_tensor[3:])
    ground_truth_mask = mask_tensor.cpu().numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(post_image)
    axes[0].set_title("Post Image")
    axes[1].imshow(ground_truth_mask, cmap="magma", vmin=0, vmax=4)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(predicted_mask, cmap="magma", vmin=0, vmax=4)
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:03d}.png", bbox_inches="tight")
    plt.close(fig)


def train_model(config: TrainingConfig, metadata_df: Optional[pd.DataFrame] = None):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(config.dataset_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_outputs_dir = Path(config.demo_outputs_dir)
    demo_checkpoint_path = Path(config.demo_checkpoint_path)

    train_df, val_df, train_dataset, val_dataset, train_loader, val_loader = prepare_dataloaders(
        config,
        metadata_df=metadata_df,
    )

    class_weights = None
    if config.use_weighted_ce or config.use_focal:
        class_weights = estimate_class_weights(train_df, dataset_root, num_classes=config.num_classes).to(device)

    criterion = build_loss(
        num_classes=config.num_classes,
        dice_weight=config.dice_weight,
        ce_weight=config.ce_weight,
        class_weights=class_weights,
        include_background=config.include_background,
        use_focal=config.use_focal,
        focal_gamma=config.focal_gamma,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2))

    history = []
    best_metric = float("-inf")
    best_checkpoint_path = output_dir / "checkpoints" / "best_model.pth"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _to_float_metrics(
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_post_only_input=config.use_post_only_input,
            )
        )
        val_metrics = _to_float_metrics(
            validate(
                model,
                val_loader,
                criterion,
                device,
                use_post_only_input=config.use_post_only_input,
            )
        )

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        _save_demo_epoch_output(
            model,
            train_dataset,
            device,
            epoch,
            demo_outputs_dir,
            use_post_only_input=config.use_post_only_input,
        )

        tracked_metric = val_metrics[config.checkpoint_metric]
        if config.overfit_subset_size > 0:
            tracked_metric = train_metrics[config.checkpoint_metric]

        if tracked_metric > best_metric:
            best_metric = tracked_metric
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                best_checkpoint_path,
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                demo_checkpoint_path,
            )

    history_df = pd.DataFrame(history)
    history_path = output_dir / "history.csv"
    history_df.to_csv(history_path, index=False)

    final_checkpoint_path = output_dir / "checkpoints" / "last_model.pth"
    save_checkpoint(
        model,
        optimizer,
        config.epochs,
        history[-1],
        final_checkpoint_path,
    )

    return {
        "device": str(device),
        "history": history_df,
        "history_path": str(history_path),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(final_checkpoint_path),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "best_metric": float(best_metric),
    }


def run_overfit_test(
    dataset_root: str,
    output_dir: str,
    subset_size: int = 8,
    epochs: int = 30,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    use_weighted_ce: bool = True,
    use_multimodal: bool = True,
    seed: int = 42,
):
    config = TrainingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_weighted_ce=use_weighted_ce,
        use_multimodal=use_multimodal,
        overfit_subset_size=subset_size,
        seed=seed,
    )
    result = train_model(config)
    history = result["history"]
    final_train_dice = float(history["train_dice"].iloc[-1])
    final_train_iou = float(history["train_iou"].iloc[-1])
    result["overfit_success"] = final_train_dice >= 0.90 or final_train_iou >= 0.85
    result["final_train_dice"] = final_train_dice
    result["final_train_iou"] = final_train_iou
    return result


def run_demo_training(
    dataset_root: str,
    output_dir: str = "artifacts/demo_run",
    subset_size: int = 10,
    epochs: int = 120,
    seed: int = 42,
):
    config = TrainingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=max(100, epochs),
        batch_size=1,
        learning_rate=1e-3,
        seed=seed,
        use_multimodal=False,
        overfit_subset_size=max(5, min(20, subset_size)),
        use_weighted_ce=True,
        use_focal=False,
        disable_augmentation=True,
        use_post_only_input=False,
        save_every=1,
        demo_outputs_dir="artifacts/demo_outputs",
        demo_checkpoint_path="checkpoints/demo_model.pth",
    )
    result = train_model(config)
    history = result["history"]
    final_train_dice = float(history["train_dice"].iloc[-1])
    final_train_iou = float(history["train_iou"].iloc[-1])
    result["final_train_dice"] = final_train_dice
    result["final_train_iou"] = final_train_iou

    # Fallback requested by user: post-only input if Siamese-diff mode cannot overfit.
    if final_train_dice < 0.90 and final_train_iou < 0.85:
        print("Demo fallback: enabling post-only input to force stronger memorization.")
        config.use_post_only_input = True
        config.output_dir = str(Path(output_dir) / "post_only_fallback")
        result = train_model(config)
        history = result["history"]
        result["final_train_dice"] = float(history["train_dice"].iloc[-1])
        result["final_train_iou"] = float(history["train_iou"].iloc[-1])
        result["used_post_only_fallback"] = True
    else:
        result["used_post_only_fallback"] = False

    return result


def run_full_training(
    dataset_root: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    use_weighted_ce: bool = True,
    use_multimodal: bool = True,
    use_focal: bool = False,
    seed: int = 42,
):
    config = TrainingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_weighted_ce=use_weighted_ce,
        use_multimodal=use_multimodal,
        use_focal=use_focal,
        seed=seed,
    )
    return train_model(config)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the disaster assessment segmentation model.")
    parser.add_argument("--dataset-root", required=True, help="Path to the PRJ-3563 dataset/data folder.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints, logs, and visual outputs.")
    parser.add_argument("--mode", choices=["overfit", "train", "demo"], default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-only", action="store_true", help="Use vision-only Siamese U-Net instead of the multimodal wrapper.")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss instead of cross entropy as the classification term.")
    parser.add_argument("--no-weighted-ce", action="store_true", help="Disable class weighting.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.mode == "demo":
        result = run_demo_training(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
            epochs=args.epochs,
            seed=args.seed,
        )
    elif args.mode == "overfit":
        result = run_overfit_test(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_weighted_ce=not args.no_weighted_ce,
            use_multimodal=not args.vision_only,
            seed=args.seed,
        )
    else:
        result = run_full_training(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_weighted_ce=not args.no_weighted_ce,
            use_multimodal=not args.vision_only,
            use_focal=args.use_focal,
            seed=args.seed,
        )

    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
