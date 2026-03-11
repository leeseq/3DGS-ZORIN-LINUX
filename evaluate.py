#!/usr/bin/env python3
"""Evaluate PSNR/SSIM/LPIPS for paired frame sequences.

Supported inputs for --ref / --test-pattern:
- video file path (frames are sampled at --fps)
- printf-style sequence pattern (e.g. scene_dense/images/frame_%06d.png)

Exit codes:
- 0: metrics computed, and any provided thresholds passed
- 1: runtime / dependency error
- 2: metrics computed but one or more thresholds failed
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def is_printf_pattern(path):
    return re.search(r"%0?\d+d", path) is not None


def expand_pattern(pattern):
    """Expand a printf-style pattern to existing file paths."""
    m = re.search(r"%(0?)([0-9]+)d", pattern)
    if not m:
        return [pattern] if os.path.exists(pattern) else []
    width = int(m.group(2))
    token = m.group(0)
    prefix, suffix = pattern.split(token)
    paths = []
    idx = 1
    while True:
        p = f"{prefix}{idx:0{width}d}{suffix}"
        if not os.path.exists(p):
            break
        paths.append(p)
        idx += 1
    return paths


def extract_video_frames(video_path, fps, out_pattern):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        out_pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def resolve_source_to_pattern(source, fps, temp_root, label):
    """Return a printf-style frame pattern for source."""
    if is_printf_pattern(source):
        return source

    source_path = Path(source)
    if source_path.is_file():
        out_dir = Path(temp_root) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pattern = str(out_dir / "frame_%06d.png")
        extract_video_frames(str(source_path), fps, out_pattern)
        return out_pattern

    raise FileNotFoundError(
        f"Unsupported source '{source}'. Use a video file or printf image pattern (e.g. frame_%06d.png)."
    )


def parse_metric_token(token):
    token_norm = token.strip().lower()
    if token_norm in {"inf", "infinity"}:
        return float("inf")
    return float(token_norm)


def run_ffmpeg_metric(ref_pattern, test_pattern, fps, metric, log_path, max_frames=0):
    left_chain = "format=yuv420p"
    right_chain = "format=yuv420p"
    if max_frames > 0:
        left_chain = f"trim=end_frame={max_frames},format=yuv420p"
        right_chain = f"trim=end_frame={max_frames},format=yuv420p"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        ref_pattern,
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        test_pattern,
        "-filter_complex",
        f"[0:v]{left_chain}[ref];[1:v]{right_chain}[test];"
        f"[ref][test]{metric}=stats_file={log_path}:shortest=1",
        "-an",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_ffmpeg_log(log_path):
    """Parse frame-wise PSNR/SSIM values and return their arithmetic mean."""
    values = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.search(r"psnr_avg:(inf|[0-9]+\.?[0-9]*)", line, flags=re.IGNORECASE)
            if m:
                values.append(parse_metric_token(m.group(1)))
                continue
            m = re.search(r"All:(inf|[0-9]+\.?[0-9]*)", line, flags=re.IGNORECASE)
            if m:
                values.append(parse_metric_token(m.group(1)))
                continue
            if "average:" in line:
                parts = line.split("average:")
                if len(parts) > 1:
                    try:
                        values.append(parse_metric_token(parts[1].split()[0]))
                    except ValueError:
                        pass
    if not values:
        return None
    return sum(values) / len(values)


def compute_lpips(ref_pattern, test_pattern, max_frames=0):
    try:
        import numpy as np
        import torch
        import lpips
        from PIL import Image
    except ImportError:
        print("LPIPS computation requires torch, lpips, numpy, and Pillow.", file=sys.stderr)
        sys.exit(1)

    ref_paths = expand_pattern(ref_pattern)
    test_paths = expand_pattern(test_pattern)
    pair_count = min(len(ref_paths), len(test_paths))
    if pair_count == 0:
        return None
    if max_frames > 0:
        pair_count = min(pair_count, max_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)

    total = 0.0
    for idx in range(pair_count):
        with Image.open(ref_paths[idx]) as im_r, Image.open(test_paths[idx]) as im_t:
            im_r = im_r.convert("RGB")
            im_t = im_t.convert("RGB")
            if im_r.size != im_t.size:
                im_t = im_t.resize(im_r.size, Image.LANCZOS)
            arr_r = torch.from_numpy(np.array(im_r).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            arr_t = torch.from_numpy(np.array(im_t).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            # LPIPS expects input in [-1, 1].
            arr_r = (arr_r * 2.0 - 1.0).to(device)
            arr_t = (arr_t * 2.0 - 1.0).to(device)
            with torch.no_grad():
                total += float(loss_fn(arr_r, arr_t).item())
    return total / pair_count


def format_val(v, precision=6):
    if v is None:
        return "n/a"
    return f"{v:.{precision}f}"


def json_safe_number(v):
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        if v > 0:
            return "inf"
        if v < 0:
            return "-inf"
        return "nan"
    return v


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS between reference and test frame sequences.")
    parser.add_argument(
        "--ref",
        required=True,
        help="Reference source: video file path or printf image pattern (e.g. ref/frame_%%06d.png).",
    )
    parser.add_argument(
        "--test-pattern",
        "--test",
        dest="test_pattern",
        required=True,
        help="Test source: video file path or printf image pattern (e.g. test/frame_%%06d.png).",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="FPS used when decoding any video source.")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (alex).")
    # Accept common typo for convenience.
    parser.add_argument("--lpsis", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit evaluation to first N pairs (0 = all).")
    parser.add_argument("--min-psnr", type=float, default=None, help="Fail if PSNR is below this value.")
    parser.add_argument("--min-ssim", type=float, default=None, help="Fail if SSIM is below this value.")
    parser.add_argument("--max-lpips", type=float, default=None, help="Fail if LPIPS is above this value.")
    parser.add_argument("--json-out", default="", help="Optional path to write machine-readable metrics JSON.")
    args = parser.parse_args()

    want_lpips = args.lpips or args.lpsis or (args.max_lpips is not None)

    with tempfile.TemporaryDirectory(prefix="eval_frames_") as temp_dir:
        try:
            ref_pattern = resolve_source_to_pattern(args.ref, args.fps, temp_dir, "ref")
            test_pattern = resolve_source_to_pattern(args.test_pattern, args.fps, temp_dir, "test")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        ref_paths = expand_pattern(ref_pattern)
        test_paths = expand_pattern(test_pattern)
        pair_count = min(len(ref_paths), len(test_paths))
        if pair_count == 0:
            print("Error: no overlapping frame pairs were found.", file=sys.stderr)
            return 1
        if args.max_frames > 0:
            pair_count = min(pair_count, args.max_frames)

        psnr_log = os.path.join(temp_dir, "psnr_eval.log")
        ssim_log = os.path.join(temp_dir, "ssim_eval.log")

        try:
            run_ffmpeg_metric(ref_pattern, test_pattern, args.fps, "psnr", psnr_log, max_frames=args.max_frames)
            run_ffmpeg_metric(ref_pattern, test_pattern, args.fps, "ssim", ssim_log, max_frames=args.max_frames)
        except subprocess.CalledProcessError as exc:
            print(f"Error: ffmpeg metric computation failed ({exc}).", file=sys.stderr)
            return 1

        psnr_val = parse_ffmpeg_log(psnr_log)
        ssim_val = parse_ffmpeg_log(ssim_log)
        lpips_val = compute_lpips(ref_pattern, test_pattern, max_frames=args.max_frames) if want_lpips else None

    print(f"pairs evaluated: {pair_count}")
    print(f"psnr average: {format_val(psnr_val)}")
    print(f"ssim average: {format_val(ssim_val)}")
    if want_lpips:
        print(f"lpips average: {format_val(lpips_val)}")

    failed_checks = []
    if args.min_psnr is not None and (psnr_val is None or psnr_val < args.min_psnr):
        failed_checks.append(f"PSNR {format_val(psnr_val)} < min-psnr {args.min_psnr:.6f}")
    if args.min_ssim is not None and (ssim_val is None or ssim_val < args.min_ssim):
        failed_checks.append(f"SSIM {format_val(ssim_val)} < min-ssim {args.min_ssim:.6f}")
    if args.max_lpips is not None and (lpips_val is None or lpips_val > args.max_lpips):
        failed_checks.append(f"LPIPS {format_val(lpips_val)} > max-lpips {args.max_lpips:.6f}")

    output = {
        "pairs_evaluated": pair_count,
        "psnr": json_safe_number(psnr_val),
        "ssim": json_safe_number(ssim_val),
        "lpips": json_safe_number(lpips_val),
        "thresholds": {
            "min_psnr": args.min_psnr,
            "min_ssim": args.min_ssim,
            "max_lpips": args.max_lpips,
        },
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")

    if failed_checks:
        print("benchmark status: FAIL")
        for check in failed_checks:
            print(f" - {check}")
        return 2

    print("benchmark status: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
