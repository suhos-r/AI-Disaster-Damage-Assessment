from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DisasterAssessmentDataset, build_ida_metadata, get_val_transforms, split_metadata
from src.evaluate import _collect_predictions, _load_demo_checkpoint


WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "PRJ-3563" / "dataset" / "data"
EVAL_DIR = PROJECT_ROOT / "artifacts" / "eval_results"
METRICS_JSON = EVAL_DIR / "per_class_metrics.json"
OUTPUT_FIGURE = EVAL_DIR / "figure8_quantitative.png"

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed",
]


def load_metrics() -> dict:
    with METRICS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_validation_confusion() -> np.ndarray:
    metadata_df = build_ida_metadata(DATASET_ROOT)
    _, val_df = split_metadata(metadata_df)
    val_dataset = DisasterAssessmentDataset(val_df, DATASET_ROOT, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_demo_checkpoint(device, PROJECT_ROOT / "checkpoints" / "demo_model.pth")

    predictions, targets = _collect_predictions(model, val_loader, device)
    predicted_labels = predictions.argmax(dim=1)

    confusion = np.zeros((5, 5), dtype=np.float64)
    for true_class in range(5):
        true_mask = targets == true_class
        if true_mask.any():
            confusion[true_class] = np.bincount(
                predicted_labels[true_mask].numpy().reshape(-1),
                minlength=5,
            )

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)
    return normalized


def plot_figure(confusion_matrix: np.ndarray, metrics_payload: dict) -> Path:
    per_class = metrics_payload["per_class_metrics"]
    dice_scores = [per_class[name]["dice"] for name in CLASS_NAMES]
    iou_scores = [per_class[name]["iou"] for name in CLASS_NAMES]

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    cm_image = ax_cm.imshow(confusion_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax_cm.set_title("Normalized Confusion Matrix", fontsize=13)
    ax_cm.set_xlabel("Predicted Label", fontsize=11)
    ax_cm.set_ylabel("True Label", fontsize=11)
    ax_cm.set_xticks(range(len(CLASS_NAMES)))
    ax_cm.set_yticks(range(len(CLASS_NAMES)))
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=11)

    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row, col]
            ax_cm.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=11, color="black")

    ax_cm.figure.colorbar(cm_image, ax=ax_cm, fraction=0.046, pad=0.04)

    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    dice_bars = ax_bar.bar(x - width / 2, dice_scores, width, label="Dice", color="#1f77b4")
    iou_bars = ax_bar.bar(x + width / 2, iou_scores, width, label="IoU", color="#006400")

    ax_bar.set_title("Per-Class Segmentation Quality", fontsize=13)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax_bar.legend(fontsize=11)

    for bars in (dice_bars, iou_bars):
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    plt.tight_layout()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUTPUT_FIGURE


def main() -> None:
    metrics_payload = load_metrics()
    confusion_matrix = build_validation_confusion()
    output_path = plot_figure(confusion_matrix, metrics_payload)
    print(f"Saved figure to {output_path}")
    print(f"File exists: {output_path.exists()}")


if __name__ == "__main__":
    main()
