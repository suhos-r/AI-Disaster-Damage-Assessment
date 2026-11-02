import json
import sys
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DisasterAssessmentDataset,
    build_ida_metadata,
    get_val_transforms,
    split_metadata,
)
from src.model import MultimodalDamageAssessment, SiameseUNet, SimpleViTSegmentation, UNet
from src.train import run_segmentation_model
from src.utils import calculate_dice, calculate_iou, calculate_metrics


WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "PRJ-3563" / "dataset" / "data"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "demo_model.pth"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eval_results"
EVAL_OUTPUT_JSON = EVAL_OUTPUT_DIR / "per_class_metrics.json"

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}


def _prepare_checkpoint_state(state):
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

        cleaned_state = {}
        for key, value in state.items():
            cleaned_key = key.replace("module.", "").replace("model.", "")
            cleaned_state[cleaned_key] = value
        return cleaned_state
    return state


def _load_demo_checkpoint(device: torch.device, checkpoint_path: Union[str, Path] = CHECKPOINT_PATH) -> SiameseUNet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SiameseUNet(in_channels=3, out_classes=5).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_prepare_checkpoint_state(state), strict=False)
    model.eval()
    return model


def _collect_predictions(model, dataloader, device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, masks in dataloader:
            images = images.to(device)
            tabular = tabular.to(device)
            masks = masks.to(device)

            outputs = run_segmentation_model(model, images, tabular)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            all_predictions.append(outputs.cpu())
            all_targets.append(masks.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return predictions, targets


def _per_class_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    predicted_labels = predictions.argmax(dim=1)
    per_class = {}

    for class_id in range(5):
        pred_mask = predicted_labels == class_id
        target_mask = targets == class_id
        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        total = pred_mask.sum().item() + target_mask.sum().item()

        per_class[CLASS_NAMES[class_id]] = {
            "class_id": class_id,
            "iou": float(intersection / union) if union > 0 else 0.0,
            "dice": float((2.0 * intersection) / total) if total > 0 else 0.0,
            "pred_pixels": int(pred_mask.sum().item()),
            "target_pixels": int(target_mask.sum().item()),
        }

    return per_class


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, targets = _collect_predictions(model, dataloader, device)
    metrics = calculate_metrics(predictions, targets)

    return {
        "IoU": calculate_iou(predictions, targets),
        "Dice": calculate_dice(predictions, targets),
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
    }


def run_demo_evaluation(
    dataset_root: Union[str, Path] = DATASET_ROOT,
    checkpoint_path: Union[str, Path] = CHECKPOINT_PATH,
    output_json: Union[str, Path] = EVAL_OUTPUT_JSON,
) -> dict:
    dataset_root = Path(dataset_root)
    checkpoint_path = Path(checkpoint_path)
    output_json = Path(output_json)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")

    metadata_df = build_ida_metadata(dataset_root)
    _, val_df = split_metadata(metadata_df)
    val_dataset = DisasterAssessmentDataset(val_df, dataset_root, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_demo_checkpoint(device, checkpoint_path=checkpoint_path)

    predictions, targets = _collect_predictions(model, val_loader, device)
    overall = calculate_metrics(predictions, targets)
    overall_metrics = {
        "IoU": float(calculate_iou(predictions, targets)),
        "Dice": float(calculate_dice(predictions, targets)),
        "Accuracy": float(overall["accuracy"]),
        "AUC": float(overall["auc"]),
    }
    per_class = _per_class_metrics(predictions, targets)

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "validation_samples": int(len(val_df)),
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Evaluation complete")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Validation samples: {len(val_df)}")
    print(
        "Overall: "
        f"IoU={overall_metrics['IoU']:.4f} "
        f"Dice={overall_metrics['Dice']:.4f} "
        f"Accuracy={overall_metrics['Accuracy']:.4f} "
        f"AUC={overall_metrics['AUC']:.4f}"
    )
    print("Per-class metrics:")
    for class_name, scores in per_class.items():
        print(
            f"  {class_name}: "
            f"IoU={scores['iou']:.4f} "
            f"Dice={scores['dice']:.4f}"
        )
    print(f"Saved per-class metrics to {output_json}")

    return payload


def run_comparative_analysis(
    dataset_root: Union[str, Path],
    metadata: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if metadata is None:
        metadata_df = build_ida_metadata(dataset_root)
    elif isinstance(metadata, pd.DataFrame):
        metadata_df = metadata
    else:
        metadata_df = pd.read_csv(metadata)

    _, val_df = split_metadata(metadata_df)
    val_dataset = DisasterAssessmentDataset(val_df, dataset_root, transform=get_val_transforms())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    models = {
        "U-Net Baseline": UNet(in_channels=6, out_classes=5).to(device),
        "Siamese U-Net": SiameseUNet(in_channels=3, out_classes=5).to(device),
        "Multimodal Siamese U-Net": MultimodalDamageAssessment(
            tabular_features=4,
            out_classes=5,
        ).to(device),
        "Vision Transformer": SimpleViTSegmentation(in_channels=6, out_classes=5).to(device),
    }

    rows = []
    for model_name, model in models.items():
        model.eval()
        metrics = evaluate_model(model, val_loader, device)
        metrics["Model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_demo_evaluation()
