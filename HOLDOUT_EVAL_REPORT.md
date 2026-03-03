# Leakage-Safe Hold-Out Evaluation Report

Date: 2026-03-03

## Goal

Evaluate generalization without data leakage by training reconstruction on a subset of frames and registering unseen hold-out frames against that fixed model.

## Dataset And Split

- Source frames: `scene_dense/images/frame_%06d.png` (43 frames total, 3840x2160).
- Split rule: deterministic `80/20` style split where every 5th frame is hold-out.
- Train frames: `35`
- Hold-out frames: `8`

Generated directories:

- Train images: `_holdout_eval/train_images`
- Hold-out images: `_holdout_eval/test_images`

## Procedure

### 1) Train-only reconstruction

```bash
./run_gs_pipeline.sh \
  --input ./_holdout_eval/train_images \
  --workspace ./_holdout_eval/train_workspace \
  --profile quality \
  --matcher exhaustive \
  --cpu \
  --no-sparse-densify
```

Outcome:

- Registered train images: `35`
- Train sparse points: `26,379`
- Output model: `_holdout_eval/train_workspace/sparse/0`

### 2) Register hold-out images against fixed train model

```bash
# Add hold-out frames into the same image directory (do not rerun mapper).
cp -n _holdout_eval/test_images/* _holdout_eval/train_workspace/images/

# Ensure features/matches exist for all images, then register only.
colmap feature_extractor --database_path _holdout_eval/train_workspace/database.db \
  --image_path _holdout_eval/train_workspace/images \
  --ImageReader.camera_model OPENCV \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher --database_path _holdout_eval/train_workspace/database.db \
  --SiftMatching.use_gpu 0

colmap image_registrator \
  --database_path _holdout_eval/train_workspace/database.db \
  --input_path _holdout_eval/train_workspace/sparse/0 \
  --output_path _holdout_eval/registered_model
```

Outcome:

- Hold-out registration: `8/8` images (`100%`)

### 3) Pose-based hold-out metrics

Method:

- Convert both models to text (`model_converter --output_type TXT`).
- Align registered model to reference model (`scene_dense/sparse/triangulated`) with Sim(3) using train anchors.
- Evaluate hold-out camera-center translation and rotation differences after alignment.

Raw metric file:

- `_holdout_eval/holdout_pose_metrics.txt`

## Results

| Metric | Value |
|---|---:|
| Hold-out frames | `8` |
| Hold-out registered | `8` |
| Registration rate | `1.000000` |
| Hold-out translation error (mean) | `0.360508682` |
| Hold-out translation error (median) | `0.385152351` |
| Hold-out translation error (max) | `0.571164662` |
| Hold-out rotation error (mean, deg) | `26.391099018` |
| Hold-out rotation error (median, deg) | `26.299028056` |
| Hold-out rotation error (max, deg) | `27.593647093` |

Additional context:

- Alignment scale (Sim(3)): `1.022577567`
- Train-anchor translation error mean: `0.332775692`
- Train sparse points: `26,379`
- Full reference sparse points: `36,881`

## Interpretation

- Registration success is strong (`100%` on unseen hold-out views), so the train model supports consistent pose recovery for omitted frames.
- Translation error is moderate and stable across hold-out views.
- Rotation error is reported for cross-reconstruction comparison and should be interpreted as a relative consistency signal (not an absolute rendering-quality metric).

## Important Limitation

This report evaluates reconstruction/registration generalization, not rendered-image fidelity on hold-out views.  
For true hold-out PSNR/SSIM/LPIPS, render the hold-out camera poses from your trained GS model and compare those renders to ground-truth hold-out frames.
