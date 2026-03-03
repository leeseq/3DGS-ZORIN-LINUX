# Image/Video Similarity Report

## Scope

This report summarizes quality metrics computed for the same comparison setup:

- **Reference source:** `scene.mp4` (decoded at `2 fps`)
- **Comparison frames (GS input):** `scene_dense/images/frame_%06d.png`
- **Frames compared:** `43`
- **Resolution:** `3840x2160`

`scene_hq/images` was also checked and is identical to `scene_dense/images` for this dataset.

## Results

| Metric | Value | Direction |
|---|---:|---|
| PSNR (average) | **35.096 dB** | Higher is better |
| SSIM (All) | **0.995234** | Closer to 1 is better |
| LPIPS (AlexNet) | **0.000000** | Lower is better |

Additional channel-wise outputs:

- PSNR: `Y=33.673764`, `U=43.455472`, `V=40.266521`
- SSIM: `Y=0.995522`, `U=0.992442`, `V=0.996878`

## Interpretation

- **PSNR 35.10 dB**: good/high similarity for extracted-frame comparison.
- **SSIM 0.9952**: very high structural similarity.
- **LPIPS 0.0000**: perceptual difference is effectively zero.

In this run, LPIPS is exactly zero because the compared frame pairs are byte-identical images (verified via file hash checks).

## Reproducibility

Commands used:

```bash
# PSNR
ffmpeg -i scene.mp4 -framerate 2 -start_number 1 -i scene_dense/images/frame_%06d.png \
  -filter_complex "[0:v]fps=2,format=yuv420p[ref];[1:v]format=yuv420p[gs];[ref][gs]psnr=stats_file=scene_vs_gs_psnr.log:shortest=1" \
  -an -f null -

# SSIM
ffmpeg -i scene.mp4 -framerate 2 -start_number 1 -i scene_dense/images/frame_%06d.png \
  -filter_complex "[0:v]fps=2,format=yuv420p[ref];[1:v]format=yuv420p[gs];[ref][gs]ssim=stats_file=scene_vs_gs_ssim.log:shortest=1" \
  -an -f null -

# LPIPS (alex) was computed in a Python venv with torch + lpips.
```
