# Project Context

## Current Problems

### 1. Segmentation quality was weak
- The old pipeline used generic six-channel concatenation without dataset-specific pairing logic.
- The app ran randomly initialized models when no checkpoints were present, which made masks look poor even when the forward pass succeeded.
- Preprocessing was inconsistent: mock data and real data were mixed, normalization was minimal, and the target mask assumptions did not match the dataset.

### 2. Grad-CAM was misaligned
- The previous implementation guessed the target layer from model attributes instead of selecting a stable decoder convolution.
- Heatmaps were generated without a segmentation-specific target scalar, so the saliency signal could drift away from the predicted class region.
- Overlay alignment depended on ad hoc resizing in the app.

### 3. The project behaved like a scaffold
- The app benchmark and simulation tabs relied on generated mock data.
- No real checkpoint loading path was enforced, and trained weights were not present in the repository.
- The training utilities only used a single loss and did not reflect a real segmentation setup.

### 4. Multimodal inputs were collected but not used
- Rainfall and wind speed sliders existed in the UI, but inference mostly bypassed them because the main path used vision-only models.
- The prior multimodal wrapper only scaled logits after prediction, which is a weak fusion strategy.

### 5. Dataset integration was incomplete
- Real images and masks in `PRJ-3563` use xBD-style filenames with pre/post files stored in the same folder, but the loader expected separate `pre/`, `post/`, and `masks/` directories.
- Pairing between `pre_image`, `post_image`, and `post_mask` was not implemented from filenames.
- Pre-disaster masks are footprint-style binary masks (`0/255`) while post-disaster masks are damage labels (`0-4`), but the old loader treated masks generically.

## Dataset Structure

Dataset root used by the app and loader:

```text
PRJ-3563/
  dataset/
    data/
      images/
        AOI1-tile_1-3_pre_disaster.png
        AOI1-tile_1-3_post_disaster.png
        ...
      masks/
        AOI1-tile_1-3_pre_disaster.png
        AOI1-tile_1-3_post_disaster.png
        ...
```

### Pairing Logic

The loader parses filenames with the pattern:

```text
<tile_id>_<pre|post>_disaster.<ext>
```

Example:

- `AOI1-tile_1-3_pre_disaster.png`
- `AOI1-tile_1-3_post_disaster.png`

Pairing rule used for training and inference:

```text
(pre_image, post_image) -> post_mask
```

Supporting files:

- `pre_mask` is loaded only for validation/debugging.
- `post_mask` is always the training target.

## Empty Masks and Class Imbalance

### Empty masks
- Completely black post masks should be treated as valid `no damage` samples.
- They must not be dropped during parsing, splitting, or training.
- The current PRJ-3563 copy in this workspace contains fully pairable samples; the loader still preserves empty masks for future dataset extensions and other xBD-style sources.

### Class imbalance
- The dataset is heavily imbalanced toward background and lower-damage classes.
- A quick scan of the current PRJ-3563 post masks shows:
  - Class `0`: dominant background
  - Class `1`: much smaller than background
  - Class `2`: smaller again
  - Class `3`: rare
  - Class `4`: extremely rare
- This imbalance is one reason the training stack now uses Dice + CrossEntropy instead of CrossEntropy alone.

## Planned and Implemented Fixes

### Dataset / Data Pipeline
- Parse real xBD-style filenames instead of assuming separate folders.
- Build a metadata table with:
  - `tile_id`
  - `pre_image_path`
  - `post_image_path`
  - `pre_mask_path`
  - `post_mask_path`
  - `mask_path` (alias to `post_mask_path`)
  - tabular features such as rainfall and wind speed
- Keep black masks as valid samples.
- Add dataset validation utilities that visualize random `(pre, post, mask)` triplets and flag size misalignment.
- Split train/validation data from the real dataset instead of using mock evaluation data.

### Model
- Use a Siamese U-Net with a shared encoder.
- Fuse temporal information as `post_features + |post - pre|` at each scale.
- Support multimodal conditioning with tabular weather/terrain inputs at the bottleneck.

### Segmentation Training
- Use Dice Loss + CrossEntropy Loss.
- Standardize resize, augmentation, normalization, and mask handling.
- Preserve output shape as `(B, C, H, W)` with masks shaped `(B, H, W)`.

### Grad-CAM / Explainability
- Select the final decoder convolution as the default Grad-CAM target layer.
- Use a segmentation-specific target scalar based on the predicted class region.
- Normalize and upsample the CAM to match the model input shape before overlaying it on the post-disaster image.

### App
- Replace mock-data benchmark/simulation flows with the real `PRJ-3563` dataset.
- Allow direct dataset-sample inference as well as manual uploads.
- Use rainfall, wind speed, elevation, and slope in the multimodal inference path.
- Surface checkpoint availability clearly so random-init behavior is not mistaken for a trained model.

## Remaining Reality Check

- The code now supports a functional real-data pipeline.
- A baseline trained Siamese checkpoint now exists at `checkpoints/siamese_unet_best.pth`, copied from the first full training run.
- The overfit test improved after debugging, but it still did not reach a near-perfect fit on 5-10 samples, so the training stack is usable but not yet fully "stable by overfit" under the strictest criterion.

## Debugging Process

### Dataset verification
- Added reusable debug utilities in `src/debug.py`.
- Randomly sampled 5 dataset entries and saved a visualization grid at:
  - `artifacts/debug/dataset_samples.png`
- Verified on the sampled entries:
  - pre/post images were correctly paired by `tile_id`
  - post masks matched post-image spatial size
  - no alignment mismatches were detected in the sampled set

### Model sanity check
- Ran a single-sample forward pass on the real dataset.
- Verified:
  - output shape matched the mask shape
  - no NaNs were present
  - outputs were not constant
- Baseline sanity stats before training showed a narrow logit range, as expected for an untrained model.
- Sanity stats after training showed broader, non-degenerate logits and predictions spanning all five classes.

### Siamese behavior validation
- Added intermediate feature return support in `src/model.py`.
- Validated that the Siamese model:
  - splits input into pre and post branches
  - reuses the same encoder
  - fuses features as concatenated `post` and `|post - pre|`
  - decodes to a segmentation mask with the expected spatial dimensions

## Overfit Test Results

### Setup
- Used 5-6 real samples only.
- Disabled augmentation during overfit mode so the model repeatedly sees the same tiny set.
- Tested both:
  - multimodal Siamese with weighted CE
  - vision-only Siamese without weighted CE

### Observations
- Initial overfit attempt failed clearly because augmentation was still active.
- After fixing the overfit loader to use validation transforms, the tiny-set fit improved substantially.
- Best tiny-set result observed:
  - train Dice about `0.666`
  - train IoU about `0.530`
- This is better than random behavior and confirms the pipeline is learning, but it is still below the desired "nearly perfect fit" threshold.

### Interpretation
- The overfit gate is only partially satisfied.
- Current evidence suggests:
  - dataset pairing and mask wiring are correct
  - loss and gradients are working
  - the model is trainable
  - optimization on very small data is still not strong enough to claim a full overfit pass

## Stability Criteria Status

### Passed
- Loss decreases during training.
- Predictions are not random noise.
- Outputs are not completely black.
- No NaNs or crashes were observed in dataset loading, forward passes, Grad-CAM, Score-CAM, or training.
- Checkpoints, logs, and visual outputs are saved successfully.

### Partially passed / still open
- Overfit test does not yet reach near-perfect performance on 5-10 samples.
- Because of that, the strictest interpretation of "stable before full training" is not fully met.

## Training Setup and Observations

### Training utilities added
- Configurable training in `src/train.py`:
  - epochs
  - batch size
  - learning rate
  - weighted CE toggle
  - focal-loss toggle
  - multimodal toggle
  - checkpoint metric
- Checkpoint saving:
  - best checkpoint
  - last checkpoint
- CSV logging:
  - `history.csv`
- Per-epoch visual prediction panels:
  - saved under `artifacts/.../visuals/`

### First full training run
- Run type:
  - vision-only Siamese U-Net
  - weighted CrossEntropy + Dice
  - 5 epochs
  - batch size 2
  - learning rate `1e-3`
- Output directory:
  - `artifacts/full_training_initial/`
- Dataset split used:
  - train: `69`
  - val: `18`
- Best validation Dice observed:
  - about `0.310`

### Notes
- This is a baseline run intended to confirm the full pipeline can train end to end.
- It is not yet a strong final model.
- Additional training time and tuning are still needed.

## Explainability Improvements

### Primary visual method
- Prediction-vs-ground-truth mask overlays are now treated as the main interpretability view.
- Per-sample visual panels can show:
  - pre image
  - post image
  - ground truth mask overlay
  - predicted mask overlay

### Grad-CAM
- Grad-CAM was kept for compatibility.
- It now uses the decoder target layer and normalized, aligned overlays.

### Score-CAM
- Added optional Score-CAM support as a gradient-free alternative.

### Feature maps
- Added intermediate feature extraction and visualization for:
  - pre encoder
  - post encoder
  - fused features
  - decoder features

### Saved explainability artifacts
- Trained-model explainability outputs were generated under:
  - `artifacts/trained_explainability/`
- This folder contains:
  - prediction panel
  - Grad-CAM image
  - Score-CAM image
  - feature-map grids
