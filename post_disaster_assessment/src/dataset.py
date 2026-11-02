import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMAGE_SIZE = 256
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TABULAR_FEATURES = ("rainfall", "wind_speed", "elevation", "slope")
FILENAME_PATTERN = re.compile(
    r"^(?P<tile_id>.+)_(?P<phase>pre|post)_disaster(?P<extension>\.[^.]+)$"
)


def parse_disaster_filename(filename: str) -> Dict[str, str]:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match the expected *_pre/post_disaster.* format."
        )

    data = match.groupdict()
    data["basename"] = filename
    return data


def _resolve_path(path_like: Union[str, Path]) -> Path:
    return Path(path_like).expanduser().resolve()


def _load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_grayscale_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    return mask


def _normalize_post_mask(mask: np.ndarray) -> np.ndarray:
    normalized = mask.astype(np.uint8)
    unique_values = set(np.unique(normalized).tolist())

    if unique_values.issubset({0, 255}):
        # Treat binary masks as background/building, mapping buildings to "no damage".
        normalized = (normalized > 0).astype(np.uint8)
    elif np.max(normalized) > 4:
        normalized = np.clip(normalized, 0, 4).astype(np.uint8)

    return normalized


def _normalize_pre_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGE_MEAN) / IMAGE_STD


def _metadata_to_dataframe(
    metadata: Union[pd.DataFrame, str, Path, None],
    dataset_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    if isinstance(metadata, pd.DataFrame):
        return metadata.reset_index(drop=True).copy()

    if metadata is None:
        if dataset_root is None:
            raise ValueError("Either metadata or dataset_root must be provided.")
        return build_ida_metadata(dataset_root)

    metadata_path = _resolve_path(metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return pd.read_csv(metadata_path)


def build_ida_metadata(
    dataset_root: Union[str, Path],
    weather_csv: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    dataset_root = _resolve_path(dataset_root)
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images' and 'masks' folders inside {dataset_root}."
        )

    paired_samples: Dict[str, Dict[str, str]] = {}

    for file_path in sorted(image_dir.glob("*.png")):
        parsed = parse_disaster_filename(file_path.name)
        tile_entry = paired_samples.setdefault(parsed["tile_id"], {})
        tile_entry[f'{parsed["phase"]}_image_path'] = str(
            Path("images") / file_path.name
        )

    for file_path in sorted(mask_dir.glob("*.png")):
        parsed = parse_disaster_filename(file_path.name)
        tile_entry = paired_samples.setdefault(parsed["tile_id"], {})
        tile_entry[f'{parsed["phase"]}_mask_path'] = str(Path("masks") / file_path.name)

    rows: List[Dict[str, Union[str, int, float, bool]]] = []
    for tile_id, entry in sorted(paired_samples.items()):
        required = ("pre_image_path", "post_image_path", "post_mask_path")
        missing = [item for item in required if item not in entry]
        if missing:
            raise ValueError(f"Tile '{tile_id}' is missing required files: {missing}")

        post_mask = _normalize_post_mask(
            _load_grayscale_mask(dataset_root / entry["post_mask_path"])
        )
        pre_mask_path = entry.get("pre_mask_path")
        pre_mask = (
            _normalize_pre_mask(_load_grayscale_mask(dataset_root / pre_mask_path))
            if pre_mask_path
            else None
        )

        class_values, class_counts = np.unique(post_mask, return_counts=True)
        class_histogram = {int(k): int(v) for k, v in zip(class_values, class_counts)}

        row = {
            "tile_id": tile_id,
            "pre_image_path": entry["pre_image_path"],
            "post_image_path": entry["post_image_path"],
            "pre_mask_path": pre_mask_path,
            "post_mask_path": entry["post_mask_path"],
            "mask_path": entry["post_mask_path"],
            "is_no_damage_sample": bool(np.max(post_mask) == 0),
            "max_damage_class": int(np.max(post_mask)),
            "foreground_pixels_pre": int(pre_mask.sum()) if pre_mask is not None else 0,
            "foreground_pixels_post": int(np.count_nonzero(post_mask > 0)),
            "class_histogram": str(class_histogram),
        }

        for feature in TABULAR_FEATURES:
            row[feature] = 0.0

        rows.append(row)

    metadata_df = pd.DataFrame(rows)

    if weather_csv is not None:
        weather_df = pd.read_csv(_resolve_path(weather_csv))
        if "tile_id" not in weather_df.columns:
            raise ValueError("Weather metadata must contain a 'tile_id' column.")
        metadata_df = metadata_df.drop(columns=[c for c in TABULAR_FEATURES if c in weather_df.columns], errors="ignore")
        metadata_df = metadata_df.merge(weather_df, on="tile_id", how="left")

    for feature in TABULAR_FEATURES:
        if feature not in metadata_df.columns:
            metadata_df[feature] = 0.0
        metadata_df[feature] = metadata_df[feature].fillna(0.0).astype(np.float32)

    if output_csv is not None:
        output_path = _resolve_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(output_path, index=False)

    return metadata_df.reset_index(drop=True)


def split_metadata(
    metadata_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metadata_df.empty:
        raise ValueError("Cannot split an empty metadata dataframe.")

    stratify = metadata_df["max_damage_class"]
    if stratify.nunique() < 2 or stratify.value_counts().min() < 2:
        stratify = None

    train_df, val_df = train_test_split(
        metadata_df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def summarize_metadata(
    metadata_df: pd.DataFrame,
    dataset_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[int, Dict[str, int], float]]:
    summary = {
        "num_samples": int(len(metadata_df)),
        "num_no_damage_samples": int(metadata_df["is_no_damage_sample"].sum())
        if "is_no_damage_sample" in metadata_df.columns
        else 0,
        "max_damage_distribution": metadata_df["max_damage_class"].value_counts().sort_index().to_dict()
        if "max_damage_class" in metadata_df.columns
        else {},
    }

    if dataset_root is not None:
        dataset_root = _resolve_path(dataset_root)
        pixel_histogram = {index: 0 for index in range(5)}
        for _, row in metadata_df.iterrows():
            mask = _normalize_post_mask(
                _load_grayscale_mask(dataset_root / row["post_mask_path"])
            )
            values, counts = np.unique(mask, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                pixel_histogram[int(value)] = pixel_histogram.get(int(value), 0) + int(count)
        summary["pixel_class_distribution"] = pixel_histogram

    return summary


def validate_dataset(
    metadata_df: pd.DataFrame,
    dataset_root: Union[str, Path],
    num_samples: int = 4,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[str, int, List[str]]]:
    dataset_root = _resolve_path(dataset_root)
    rng = random.Random(seed)
    indices = list(range(len(metadata_df)))
    if not indices:
        raise ValueError("Cannot validate an empty dataset.")

    sample_indices = rng.sample(indices, k=min(num_samples, len(indices)))
    alignment_issues: List[str] = []

    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample_idx in enumerate(sample_indices):
        row = metadata_df.iloc[sample_idx]
        pre_image = _load_rgb_image(dataset_root / row["pre_image_path"])
        post_image = _load_rgb_image(dataset_root / row["post_image_path"])
        post_mask = _normalize_post_mask(
            _load_grayscale_mask(dataset_root / row["post_mask_path"])
        )

        if pre_image.shape[:2] != post_image.shape[:2] or pre_image.shape[:2] != post_mask.shape[:2]:
            alignment_issues.append(row["tile_id"])

        axes[row_idx, 0].imshow(pre_image)
        axes[row_idx, 0].set_title(f'{row["tile_id"]} | Pre')
        axes[row_idx, 1].imshow(post_image)
        axes[row_idx, 1].set_title("Post")
        axes[row_idx, 2].imshow(post_mask, cmap="magma", vmin=0, vmax=4)
        axes[row_idx, 2].set_title("Post Mask")

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()

    saved_path = None
    if output_path is not None:
        output_path = _resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        saved_path = str(output_path)
    plt.close(fig)

    return {
        "sample_count": len(sample_indices),
        "alignment_issues": alignment_issues,
        "saved_path": saved_path or "",
    }


class DisasterAssessmentDataset(Dataset):
    def __init__(
        self,
        metadata: Union[pd.DataFrame, str, Path, None],
        image_dir: Optional[Union[str, Path]] = None,
        transform: Optional[A.Compose] = None,
    ):
        self.image_dir = _resolve_path(image_dir) if image_dir is not None else None
        self.metadata_df = _metadata_to_dataframe(metadata, dataset_root=self.image_dir)
        if self.image_dir is None:
            raise ValueError("image_dir/dataset_root must be provided for file loading.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int):
        row = self.metadata_df.iloc[idx]

        pre_image = _load_rgb_image(self.image_dir / row["pre_image_path"])
        post_image = _load_rgb_image(self.image_dir / row["post_image_path"])
        post_mask = _normalize_post_mask(
            _load_grayscale_mask(self.image_dir / row["post_mask_path"])
        )
        pre_mask = (
            _normalize_pre_mask(_load_grayscale_mask(self.image_dir / row["pre_mask_path"]))
            if pd.notna(row.get("pre_mask_path"))
            else np.zeros_like(post_mask, dtype=np.uint8)
        )

        if self.transform is not None:
            augmented = self.transform(
                image=pre_image,
                post_image=post_image,
                mask=post_mask,
                pre_mask=pre_mask,
            )
            pre_image = augmented["image"]
            post_image = augmented["post_image"]
            post_mask = augmented["mask"]

        pre_image = _normalize_image(pre_image)
        post_image = _normalize_image(post_image)

        pre_tensor = torch.from_numpy(pre_image.transpose(2, 0, 1)).float()
        post_tensor = torch.from_numpy(post_image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(post_mask.astype(np.int64))

        input_tensor = torch.cat([pre_tensor, post_tensor], dim=0)
        tabular_tensor = torch.tensor(
            [float(row.get(feature, 0.0)) for feature in TABULAR_FEATURES],
            dtype=torch.float32,
        )

        return input_tensor, tabular_tensor, mask_tensor


def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.5,
            ),
        ],
        additional_targets={"post_image": "image", "pre_mask": "mask"},
    )


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [A.Resize(image_size, image_size)],
        additional_targets={"post_image": "image", "pre_mask": "mask"},
    )


def prepare_inference_batch(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    rainfall: float = 0.0,
    wind_speed: float = 0.0,
    elevation: float = 0.0,
    slope: float = 0.0,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    resized = get_val_transforms(image_size=image_size)(
        image=pre_image,
        post_image=post_image,
        mask=np.zeros(pre_image.shape[:2], dtype=np.uint8),
        pre_mask=np.zeros(pre_image.shape[:2], dtype=np.uint8),
    )

    pre_tensor = torch.from_numpy(
        _normalize_image(resized["image"]).transpose(2, 0, 1)
    ).float()
    post_tensor = torch.from_numpy(
        _normalize_image(resized["post_image"]).transpose(2, 0, 1)
    ).float()
    image_tensor = torch.cat([pre_tensor, post_tensor], dim=0).unsqueeze(0)

    tabular_tensor = torch.tensor(
        [[rainfall, wind_speed, elevation, slope]], dtype=torch.float32
    )

    if device is not None:
        image_tensor = image_tensor.to(device)
        tabular_tensor = tabular_tensor.to(device)

    return image_tensor, tabular_tensor


def generate_mock_data(base_dir: Union[str, Path], num_samples: int = 10) -> pd.DataFrame:
    """
    Keeps a tiny synthetic dataset helper for local smoke tests, while the app and
    training pipeline use the real PRJ-3563/xBD-style dataset.
    """
    base_dir = _resolve_path(base_dir)
    (base_dir / "pre").mkdir(parents=True, exist_ok=True)
    (base_dir / "post").mkdir(parents=True, exist_ok=True)
    (base_dir / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(num_samples):
        pre_path = Path("pre") / f"pre_{index}.png"
        post_path = Path("post") / f"post_{index}.png"
        mask_path = Path("masks") / f"mask_{index}.png"

        pre_image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        post_image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        mask = np.random.randint(0, 5, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        cv2.imwrite(str(base_dir / pre_path), cv2.cvtColor(pre_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base_dir / post_path), cv2.cvtColor(post_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base_dir / mask_path), mask)

        rows.append(
            {
                "tile_id": f"mock_{index}",
                "pre_image_path": str(pre_path),
                "post_image_path": str(post_path),
                "pre_mask_path": None,
                "post_mask_path": str(mask_path),
                "mask_path": str(mask_path),
                "is_no_damage_sample": bool(np.max(mask) == 0),
                "max_damage_class": int(np.max(mask)),
                "rainfall": float(np.random.uniform(0, 100)),
                "wind_speed": float(np.random.uniform(0, 150)),
                "elevation": float(np.random.uniform(0, 1000)),
                "slope": float(np.random.uniform(0, 45)),
            }
        )

    return pd.DataFrame(rows)
