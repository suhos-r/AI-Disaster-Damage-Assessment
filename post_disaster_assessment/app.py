import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit.runtime import exists as streamlit_runtime_exists

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.dataset import IMAGE_SIZE, build_ida_metadata, prepare_inference_batch
from src.model import MultimodalDamageAssessment, SiameseUNet, SimpleViTSegmentation, UNet
from src.train import run_segmentation_model
from src.utils import overlay_mask
from src.xai import generate_gradcam, overlay_gradcam, select_gradcam_layer


DATASET_ROOT = WORKSPACE_DIR / "PRJ-3563" / "dataset" / "data"
CHECKPOINT_DIR = CURRENT_DIR / "checkpoints"
MODEL_PATTERNS = {
    "Siamese U-Net (Change Detection)": [
        "demo_model.pth",
        "siamese_unet*.pth",
        "siamese*.pth",
    ],
    "Multimodal Siamese U-Net": ["multimodal*.pth", "siamese_multimodal*.pth"],
    "U-Net Baseline": ["unet*.pth"],
    "Vision Transformer (Reference)": ["vit*.pth", "transformer*.pth"],
}
DEMO_MODEL_NAME = "Siamese U-Net (Change Detection)"

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}
CLASS_COLORS = {
    0: "#8B949E",
    1: "#639922",
    2: "#185FA5",
    3: "#F2A23A",
    4: "#D85A30",
}
if __name__ == "__main__" and not streamlit_runtime_exists():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        check=False,
    )
    raise SystemExit(0)


st.set_page_config(page_title="Post-Disaster Damage Intelligence", page_icon="🛰️", layout="wide")


def _inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #0D1117;
            color: #FFFFFF;
        }
        .stApp { background: #0D1117; }
        [data-testid="stSidebar"] {
            background: #161B22;
            border-right: 1px solid rgba(15,110,86,0.35);
        }
        [data-testid="stSidebar"] * { color: #FFFFFF; }

        #MainMenu, header, footer { visibility: hidden; }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #161B22; }
        ::-webkit-scrollbar-thumb { background: #0F6E56; border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: #119E7A; }

        .block-container {
            max-width: 1360px;
            padding-top: 0.65rem;
            padding-bottom: 1.1rem;
        }

        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
            background: linear-gradient(120deg, #1C2333 0%, #161B22 100%);
            border: 1px solid rgba(15,110,86,0.30);
            border-radius: 12px;
        }
        .header-title {
            color: #FFFFFF;
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .header-meta {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            color: #8B949E;
            font-size: 0.86rem;
        }
        .live-pill {
            background: rgba(99,153,34,0.22);
            color: #B7E081;
            border: 1px solid rgba(99,153,34,0.55);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-weight: 600;
            font-size: 0.8rem;
        }

        .sidebar-logo {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.8rem;
        }
        .sidebar-logo h3 {
            margin: 0;
            color: #0F6E56;
            font-size: 1rem;
            font-weight: 700;
        }
        .sidebar-logo p {
            margin: 0.15rem 0 0;
            color: #8B949E;
            font-size: 0.78rem;
        }
        .sidebar-card {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-left: 4px solid #0F6E56;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.75rem;
        }
        .sidebar-card .title {
            color: #8B949E;
            font-size: 0.76rem;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .sidebar-card .value {
            color: #FFFFFF;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .metric-card {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-top: 3px solid #0F6E56;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: #FFFFFF;
            font-size: 1.4rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-label {
            color: #8B949E;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }
        .metric-delta {
            color: #0F6E56;
            font-size: 0.72rem;
            margin-top: 0.1rem;
        }

        .section-title {
            border-left: 3px solid #0F6E56;
            padding-left: 0.55rem;
            color: #FFFFFF;
            font-size: 1.05rem;
            font-weight: 650;
            margin: 0.45rem 0 0.5rem;
        }
        .teal-rule {
            border: none;
            border-top: 1px solid rgba(15,110,86,0.55);
            margin: 0.35rem 0 0.75rem;
        }
        .card-shell {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-top: 2px solid #0F6E56;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 0.45rem;
        }

        div[data-testid="stImage"] {
            border: 1px solid rgba(15, 110, 86, 0.3);
            border-radius: 8px;
            padding: 8px;
            background: #1C2333;
            margin-bottom: 0.3rem;
        }
        div[data-testid="stImage"] img {
            border-radius: 6px;
        }

        [data-testid="stRadio"] > div {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stRadio"] label {
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
        }

        .stRadio label, .stSelectbox label, div[data-testid="stRadio"] p,
        div[data-testid="stSelectbox"] label, .stRadio div[role="radiogroup"] label span {
            color: #E6EDF3 !important;
            font-size: 14px !important;
        }
        div[data-baseweb="select"] div {
            color: #E6EDF3 !important;
        }

        [data-testid="stSelectbox"] > div > div {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.45);
            border-radius: 8px;
        }
        [data-testid="stSelectbox"] input {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            color: #FFFFFF !important;
        }

        [data-testid="stCheckbox"] {
            background: #1C2333;
            border: 1px solid rgba(15,110,86,0.30);
            border-radius: 8px;
            padding: 0.35rem 0.55rem;
        }

        .stButton > button {
            width: 100%;
            border: none;
            color: #FFFFFF;
            font-weight: 700;
            border-radius: 10px;
            padding: 0.62rem 0.8rem;
            background: linear-gradient(90deg, #0F6E56 0%, #185FA5 100%);
            box-shadow: 0 0 0 rgba(24,95,165,0.0);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 0 14px rgba(24,95,165,0.38);
            transform: translateY(-1px);
        }

        [data-testid="stMetricDelta"] { color: #0F6E56 !important; }
        [data-testid="stMetricDelta"] svg { fill: #0F6E56 !important; }
        [data-testid="stMetricDelta"][data-direction="down"] { color: #D85A30 !important; }

        .result-badge {
            display: inline-block;
            padding: 0.48rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            margin-bottom: 0.45rem;
            border: 1px solid rgba(255,255,255,0.14);
        }

        .confidence-wrap {
            width: 100%;
            background: #0F1724;
            border: 1px solid rgba(15,110,86,0.35);
            border-radius: 999px;
            height: 12px;
            position: relative;
            overflow: hidden;
            margin-top: 0.25rem;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #0F6E56, #185FA5);
            border-radius: 999px;
        }
        .info-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.16);
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_image(path: Path) -> np.ndarray:
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


def _resize_image(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


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


def _find_checkpoint(model_name: str):
    for pattern in MODEL_PATTERNS.get(model_name, []):
        matches = sorted(CHECKPOINT_DIR.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _short_path(path: Path, keep_parts: int = 4) -> str:
    path_parts = path.parts
    if len(path_parts) <= keep_parts:
        return str(path)
    return ".../" + "/".join(path_parts[-keep_parts:])


def _overall_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    iou_scores = []
    dice_scores = []
    for class_id in range(5):
        pred_mask = prediction == class_id
        target_mask = target == class_id
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        if union > 0:
            iou_scores.append(intersection / union)
        if total > 0:
            dice_scores.append((2.0 * intersection) / total)

    return {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "pixel_accuracy": float((prediction == target).mean()),
    }


def _dominant_damage_class(mask: np.ndarray) -> int:
    counts = np.bincount(mask.reshape(-1), minlength=5)
    if counts[1:].sum() == 0:
        return 1
    return int(np.argmax(counts[1:]) + 1)


def _confidence_for_class(probabilities: np.ndarray, class_id: int) -> float:
    return float(probabilities[class_id])


def _get_image_metadata(source_name: str, image: np.ndarray, approx_size_bytes: int) -> str:
    height, width = image.shape[:2]
    size_kb = approx_size_bytes / 1024.0
    return f"{source_name} | {width}x{height}px | {size_kb:.1f} KB"


def _render_image_card(
    title: str,
    image: np.ndarray,
    badge_text: str,
    badge_color: str,
    metadata: str,
):
    st.markdown(
        f"""
        <div style="font-size:13px; color:#0F6E56; font-weight:650; margin-bottom:0.2rem;">{title}</div>
        <span style="
            display:inline-block;
            background:{badge_color}33;
            color:#E6EDF3;
            border:1px solid {badge_color}88;
            border-radius:999px;
            padding:0.12rem 0.48rem;
            font-size:11px;
            font-weight:700;
            margin-bottom:0.35rem;">{badge_text}</span>
        """,
        unsafe_allow_html=True,
    )
    st.image(image, use_container_width=True)
    if metadata:
        st.caption(metadata)


def _render_section_header(title: str):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def _render_top_metric_card(label: str, value: str, delta_text: str = ""):
    delta_html = f"<div class='metric-delta'>{delta_text}</div>" if delta_text else "<div class='metric-delta'>&nbsp;</div>"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset_metadata():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_ROOT}. Expected the PRJ-3563 folder beside the app."
        )

    metadata_df = build_ida_metadata(DATASET_ROOT)
    class_counts = metadata_df["max_damage_class"].value_counts().sort_index()
    class_counts = class_counts.reindex([0, 1, 2, 3, 4], fill_value=0)

    complete_rows = 0
    for _, row in metadata_df.iterrows():
        if (
            (DATASET_ROOT / row["pre_image_path"]).exists()
            and (DATASET_ROOT / row["post_image_path"]).exists()
            and (DATASET_ROOT / row["post_mask_path"]).exists()
        ):
            complete_rows += 1
    completeness = (complete_rows / max(len(metadata_df), 1)) * 100.0

    positive_counts = class_counts[class_counts > 0]
    imbalance_ratio = (
        float(positive_counts.max() / positive_counts.min()) if len(positive_counts) > 1 else 1.0
    )
    dominant_class_id = int(class_counts.idxmax())
    dominant_class_name = CLASS_NAMES.get(dominant_class_id, f"Class {dominant_class_id}")

    class_dist_df = pd.DataFrame(
        {
            "Bucket": ["Class Distribution"] * len(class_counts),
            "ClassLabel": [CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}") for class_id in class_counts.index],
            "Count": class_counts.values.astype(int),
            "ClassColor": [CLASS_COLORS.get(int(class_id), "#8B949E") for class_id in class_counts.index],
        }
    )

    return {
        "metadata_df": metadata_df,
        "class_counts": class_counts,
        "class_dist_df": class_dist_df,
        "completeness_pct": completeness,
        "imbalance_ratio": imbalance_ratio,
        "dominant_class_id": dominant_class_id,
        "dominant_class_name": dominant_class_name,
    }


@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "Siamese U-Net (Change Detection)": SiameseUNet(in_channels=3, out_classes=5).to(device),
        "Multimodal Siamese U-Net": MultimodalDamageAssessment(
            tabular_features=4,
            out_classes=5,
        ).to(device),
        "U-Net Baseline": UNet(in_channels=6, out_classes=5).to(device),
        "Vision Transformer (Reference)": SimpleViTSegmentation(
            in_channels=6,
            out_classes=5,
        ).to(device),
    }

    checkpoint_status = {}
    for model_name, model in models.items():
        checkpoint_path = _find_checkpoint(model_name)
        if checkpoint_path is not None and checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(_prepare_checkpoint_state(state), strict=False)
            checkpoint_status[model_name] = str(checkpoint_path)
        else:
            checkpoint_status[model_name] = ""
        model.eval()

    return models, checkpoint_status, device


def _load_dataset_sample(metadata_df: pd.DataFrame, tile_id: str):
    row = metadata_df.loc[metadata_df["tile_id"] == tile_id].iloc[0]
    pre_path = DATASET_ROOT / row["pre_image_path"]
    post_path = DATASET_ROOT / row["post_image_path"]
    mask_path = DATASET_ROOT / row["post_mask_path"]
    pre_image = _load_image(pre_path)
    post_image = _load_image(post_path)
    post_mask = _load_mask(mask_path)
    return row, pre_path, post_path, mask_path, pre_image, post_image, post_mask


def _render_distribution_chart(class_dist_df: pd.DataFrame):
    color_domain = class_dist_df["ClassLabel"].tolist()
    color_range = class_dist_df["ClassColor"].tolist()
    chart = (
        alt.Chart(class_dist_df)
        .mark_bar()
        .encode(
            y=alt.Y("Bucket:N", title=None, axis=None),
            x=alt.X("Count:Q", title=None, stack=True),
            color=alt.Color(
                "ClassLabel:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title=None, orient="bottom", labelColor="#8B949E"),
            ),
            tooltip=["ClassLabel:N", "Count:Q"],
        )
        .properties(height=55)
        .configure_view(strokeWidth=0)
        .configure(background="#161B22")
        .configure_axis(labelColor="#8B949E", titleColor="#8B949E", domain=False, tickColor="#8B949E")
    )
    st.altair_chart(chart, use_container_width=True)


_inject_global_styles()

models, checkpoint_status, device = load_models()
dataset_bundle = load_dataset_metadata()
metadata_df = dataset_bundle["metadata_df"]
class_counts = dataset_bundle["class_counts"]
class_dist_df = dataset_bundle["class_dist_df"]

timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f"""
    <div class="header-bar">
        <div class="header-title">🛰️ Post-Disaster Damage Assessment Dashboard</div>
        <div class="header-meta">
            <span>Loaded: {timestamp_text}</span>
            <span class="live-pill">System Status: Live</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <h3>🛰️ Damage Intelligence</h3>
            <p>Operational dashboard for rapid post-disaster screening</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="title">Dataset Root</div>
            <div class="value">{_short_path(DATASET_ROOT)}</div>
            <div class="title" style="margin-top:0.45rem;">Total Samples</div>
            <div class="value">{len(metadata_df)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="title">Dataset Health</div>
            <div class="value">Completeness: {dataset_bundle['completeness_pct']:.1f}%</div>
            <div class="value" style="margin-top:0.2rem;">Imbalance Ratio: {dataset_bundle['imbalance_ratio']:.2f}</div>
            <div class="value" style="margin-top:0.2rem;">Dominant: {dataset_bundle['dominant_class_name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_checkpoint = checkpoint_status.get(DEMO_MODEL_NAME, "")
    st.markdown("**Active Model**")
    st.caption(f"Architecture: {DEMO_MODEL_NAME}")
    if active_checkpoint:
        st.success(f"Loaded checkpoint: {Path(active_checkpoint).name}")
    else:
        st.warning("Loaded checkpoint: random init")

    st.markdown("<div style='color:#8B949E; font-size:0.8rem; margin:0.25rem 0 0.4rem;'>Class Distribution</div>", unsafe_allow_html=True)
    _render_distribution_chart(class_dist_df)

total_samples = len(metadata_df)
damage_classes = int(class_counts[class_counts > 0].shape[0])
dominant_class_name = dataset_bundle["dominant_class_name"]
imbalance_ratio = dataset_bundle["imbalance_ratio"]

metric_cols = st.columns(4, gap="medium")
with metric_cols[0]:
    _render_top_metric_card("Total Samples", f"{total_samples}", delta_text="Loaded")
with metric_cols[1]:
    _render_top_metric_card("Damage Classes", f"{damage_classes}", delta_text="Active")
with metric_cols[2]:
    _render_top_metric_card("Dominant Class", dominant_class_name, delta_text="From labels")
with metric_cols[3]:
    _render_top_metric_card("Class Imbalance Ratio", f"{imbalance_ratio:.2f}", delta_text="max / min")

controls_left, controls_right = st.columns([2.4, 1.2], gap="large")
with controls_left:
    _render_section_header("Input Configuration")
    source_mode = st.radio("Input Source", ["Dataset sample", "Upload images"], horizontal=True)

with controls_right:
    _render_section_header("Options")

selected_row = None
pre_image = None
post_image = None
ground_truth_mask = None
pre_meta = ""
post_meta = ""

if source_mode == "Dataset sample":
    selected_tile = st.selectbox(
        "Dataset Tile ℹ️",
        metadata_df["tile_id"].tolist(),
        help="AOI tiles are pre/post image pairs extracted from specific geographic regions.",
    )
    (
        selected_row,
        pre_path,
        post_path,
        _,
        pre_image,
        post_image,
        ground_truth_mask,
    ) = _load_dataset_sample(metadata_df, selected_tile)
    pre_meta = _get_image_metadata(pre_path.name, pre_image, pre_path.stat().st_size)
    post_meta = _get_image_metadata(post_path.name, post_image, post_path.stat().st_size)
else:
    upload_col1, upload_col2 = st.columns(2, gap="large")
    with upload_col1:
        pre_file = st.file_uploader("Pre-disaster image", type=["png", "jpg", "jpeg"])
    with upload_col2:
        post_file = st.file_uploader("Post-disaster image", type=["png", "jpg", "jpeg"])

    if pre_file is not None and post_file is not None:
        pre_bytes = pre_file.getvalue()
        post_bytes = post_file.getvalue()
        pre_image = np.array(Image.open(pre_file).convert("RGB"))
        post_image = np.array(Image.open(post_file).convert("RGB"))
        pre_meta = _get_image_metadata(pre_file.name, pre_image, len(pre_bytes))
        post_meta = _get_image_metadata(post_file.name, post_image, len(post_bytes))

if st.button("Run Damage Assessment", type="primary"):
    if pre_image is None or post_image is None:
        st.warning("Provide both pre- and post-disaster images to run inference.")
    else:
        model = models[DEMO_MODEL_NAME]
        pre_display = _resize_image(pre_image)
        post_display = _resize_image(post_image)

        image_tensor, tabular_tensor = prepare_inference_batch(
            pre_image,
            post_image,
            device=device,
        )

        with st.spinner("Running damage assessment model..."):
            with torch.no_grad():
                logits = run_segmentation_model(model, image_tensor, tabular_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                prediction = probabilities.argmax(axis=0).astype(np.uint8)

            prediction_overlay = overlay_mask(post_display, prediction, alpha=0.55)
            if ground_truth_mask is not None:
                target_resized = _resize_mask(ground_truth_mask)
                ground_truth_overlay = overlay_mask(post_display, target_resized, alpha=0.55)
            else:
                target_resized = None
                ground_truth_overlay = post_display.copy()

            pred_class = _dominant_damage_class(prediction)

            gt_class = _dominant_damage_class(target_resized) if target_resized is not None else None
            match_status = (gt_class == pred_class) if gt_class is not None else None

            gradcam_overlay = None
            try:
                target_layer = select_gradcam_layer(model)
                gradcam_map = generate_gradcam(
                    model,
                    target_layer,
                    image_tensor,
                    tabular_tensor=tabular_tensor,
                )
                gradcam_overlay = overlay_gradcam(post_display, gradcam_map)
            except Exception:
                gradcam_overlay = None

        st.session_state["assessment"] = {
            "pre_display": pre_display,
            "post_display": post_display,
            "prediction_overlay": prediction_overlay,
            "ground_truth_overlay": ground_truth_overlay,
            "prediction_mask": prediction,
            "target_mask": target_resized,
            "pred_class": pred_class,
            "gt_class": gt_class,
            "match_status": match_status,
            "gradcam_overlay": gradcam_overlay,
        }
        st.toast(f"Assessment complete — Class {pred_class} detected")

assessment = st.session_state.get("assessment")
if assessment is not None:
    _render_section_header("Assessment View")

    image_cols = st.columns(5, gap="small")
    with image_cols[0]:
        _render_image_card(
            "Pre-Disaster",
            assessment["pre_display"],
            "PRE",
            "#185FA5",
            pre_meta,
        )
    with image_cols[1]:
        _render_image_card(
            "Post-Disaster",
            assessment["post_display"],
            "POST",
            "#D85A30",
            post_meta,
        )
    with image_cols[2]:
        _render_image_card(
            "Predicted Damage Mask",
            assessment["prediction_overlay"],
            "PRED",
            "#0F6E56",
            "Model segmentation overlay",
        )
    with image_cols[3]:
        _render_image_card(
            "Ground Truth Overlay",
            assessment["ground_truth_overlay"],
            "GT",
            "#185FA5",
            "Reference annotation overlay",
        )
    with image_cols[4]:
        st.markdown(
            """
            <div style="font-size:13px; color:#0F6E56; font-weight:650; margin-bottom:0.2rem;">Grad-CAM Overlay</div>
            <span style="
                display:inline-block;
                background:#D85A30;
                color:#FFFFFF;
                border-radius:12px;
                padding:3px 10px;
                font-size:11px;
                font-weight:700;
                margin-bottom:0.35rem;">CAM</span>
            """,
            unsafe_allow_html=True,
        )
        if assessment["gradcam_overlay"] is not None:
            st.image(assessment["gradcam_overlay"], use_container_width=True)
            st.caption("Class activation overlay")
        else:
            st.markdown(
                """
                <div class="card-shell" style="min-height:260px; display:flex; align-items:center; justify-content:center; text-align:center; color:#8B949E;">
                    Grad-CAM unavailable for this tile
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption("Left: Input pair — Right: Model output vs ground truth")
