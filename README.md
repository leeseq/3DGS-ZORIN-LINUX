# Gaussian Splatting Data Prep Pipeline (`ffmpeg` + `COLMAP`)

This repo provides an optimized pipeline to generate `.ply` files for Gaussian Splatting initialization.

## Requirements

- `ffmpeg`
- `colmap` (GPU build recommended)
  - the script now automatically falls back to `/snap/bin/colmap` if the command isn't otherwise
  - if that still fails, set `COLMAP_BIN` to the full path before running the script

Check:

```bash
ffmpeg -version
# should print help; if not on PATH specify full path or set COLMAP_BIN
colmap -h
```

## Quick Start

```bash
# if colmap isn't in PATH (snap installs are automatically handled):
#   export COLMAP_BIN=/path/to/colmap
# or run with absolute path: COLMAP_BIN=/path/to/colmap ./run_gs_pipeline.sh \
./run_gs_pipeline.sh \
  --input /path/to/video.mp4 \
  --workspace ./scene01 \
  --fps 2 \
  --max-image-size 3200 \
  --camera-model OPENCV \
  --matcher exhaustive \
  --dense
```

For denser sparse `.ply` output (recommended when your sparse cloud looks thin):

```bash
./run_gs_pipeline.sh \
  --input /path/to/video.mp4 \
  --workspace ./scene_dense \
  --fps 2 \
  --matcher exhaustive \
  --sift-max-num-features 12000 \
  --sift-peak-threshold 0.004 \
  --sift-match-max-ratio 0.85 \
  --sift-match-min-inliers 12 \
  --mapper-min-num-matches 15 \
  --mapper-init-min-num-inliers 80 \
  --mapper-abs-pose-min-inliers 20 \
  --mapper-filter-min-tri-angle 1.0 \
  --mapper-tri-min-angle 1.0
```

For image folder input:

```bash
./run_gs_pipeline.sh --input /path/to/images --workspace ./scene02 --dense
```

## Outputs

- Sparse PLY: `workspace/sparse/0/points3D_sparse.ply`
- Sparse text model: `workspace/sparse/0/text/`
- Dense PLY (if `--dense`): `workspace/dense/fused_dense.ply`

Note: by default, the script now runs a sparse densification pass (`point_triangulator`) and may write the final sparse model under `workspace/sparse/triangulated/`.

## Quality Recommendations (Important)

- Capture with high shutter speed to reduce motion blur.
- Ensure strong overlap (60-80%) between consecutive views.
- Move slowly and cover all angles around the object/scene.
- Prefer fixed exposure/focus if possible (avoid auto changes).
- Start with `--fps 2`; raise to `3-4` for fast motion, lower to `1` for slow motion.
- Use `--matcher exhaustive` for best quality (slower, more robust).
- Keep `--max-image-size` high (e.g. `3200-4000`) if VRAM allows.
- If sparse output is still thin, keep default sparse densification enabled (do not pass `--no-sparse-densify`) and lower `--sift-peak-threshold` slightly (e.g. `0.0035`).
- On non-CUDA COLMAP builds, `--dense` cannot run PatchMatch; sparse densification still works and usually improves `points3D_sparse.ply`.
- In headless shells (no usable display), the script now auto-falls back to CPU SIFT/matching. You can also force CPU explicitly with `--cpu`.

## Notes for Gaussian Splatting

Most Gaussian Splatting training code needs camera poses/intrinsics + images + optional initial point cloud.
This pipeline exports:

- COLMAP text model (`cameras.txt`, `images.txt`, `points3D.txt`)
- sparse or dense `.ply`

These are the standard inputs expected by common GS preprocess scripts.
