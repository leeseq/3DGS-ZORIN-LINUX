# Drjohnson Export Metrics Report

> This report covers the hold-out image-quality evaluation for the `drjohnson`
> Brush export at `30000` steps.

## Scope

- **Workspace:** `drjohnson_dense_brush`
- **Export:** `drjohnson_dense_brush/brush_exports/export_30000.ply`
- **Held-out split:** every 5th image (`53` eval views, `210` training views)
- **Reference images:** `drjohnson_dense_brush/images/IMG_*.jpg`
- **Saved eval renders:** `/tmp/drjohnson_brush_30000_eval/eval_30000`

## Results

`evaluate.py` results on the saved eval renders:

| Metric | Value | Direction |
|---|---:|---|
| PSNR (average) | **33.194151 dB** | Higher is better |
| SSIM (average) | **0.937046** | Closer to 1 is better |
| LPIPS (AlexNet) | **0.157475** | Lower is better |

Brush's internal eval at iteration `30000`:

| Metric | Value | Direction |
|---|---:|---|
| PSNR | **29.965265 dB** | Higher is better |
| SSIM | **0.9023107** | Closer to 1 is better |

## Notes

- The saved Brush eval renders were `1331x875`.
- The source reference images were `1332x876`.
- For the `evaluate.py` pass, the eval renders were resized to the reference resolution before PSNR/SSIM/LPIPS computation.
- The final metrics JSON is stored at `/tmp/drjohnson_export30000_metrics.json`.

## Reproducibility

```bash
# Run Brush to step 30000 and save eval renders.
../brush-app-x86_64-unknown-linux-gnu/brush_app ./drjohnson_dense_brush \
  --total-steps 30000 \
  --eval-split-every 5 \
  --eval-every 30000 \
  --eval-save-to-disk \
  --export-every 30000 \
  --export-path /tmp/drjohnson_brush_30000_eval

# Resize saved eval renders to match the source resolution.
python3 - <<'PY'
from PIL import Image
from pathlib import Path
src = Path('/tmp/drjohnson_brush_30000_eval/eval_30000')
out = Path('/tmp/drjohnson_eval_resized_test_seq')
out.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob('*.png')):
    im = Image.open(p).convert('RGB')
    im = im.resize((1332, 876), Image.LANCZOS)
    im.save(out / p.name)
PY

# Evaluate the resized render sequence.
. /tmp/lpips_sys_venv/bin/activate
python evaluate.py \
  --ref "/tmp/drjohnson_eval_ref_seq/frame_%06d.jpg" \
  --test-pattern "/tmp/drjohnson_eval_resized_test_seq/frame_%06d.png" \
  --lpips \
  --json-out /tmp/drjohnson_export30000_metrics.json
```
