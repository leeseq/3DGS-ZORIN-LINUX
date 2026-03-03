#!/usr/bin/env python3
"""Quick evaluation helper for pipeline outputs.

Usage examples:
    # PSNR/SSIM only (requires ffmpeg on PATH)
    ./evaluate.py \
        --ref scene.mp4 \
        --test-pattern "scene_dense/images/frame_%06d.png" \
        --fps 2

    # also compute LPIPS (requires torch and lpips in the current env)
    ./evaluate.py ... --lpips

The script runs ffmpeg filters to produce PSNR/SSIM statistics, then
parses the results and prints average scores.  When ``--lpips`` is passed
the script will load each paired frame with PIL/torch and compute the
AlexNet-based LPIPS distance.

This is intended as a convenience wrapper around the commands shown in
METRICS_REPORT.md and the README, letting you evaluate a workspace in one
shot.

"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run_ffmpeg_metric(ref, test_pattern, fps, metric, log_path):
    # metric: either "psnr" or "ssim"
    # log_path: file where ffmpeg writes stats (overwritten)
    cmd = [
        "ffmpeg",
        "-i",
        ref,
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        test_pattern,
        "-filter_complex",
        f"[0:v]fps={fps},format=yuv420p[ref];"
        f"[1:v]format=yuv420p[gs];"
        f"[ref][gs]{metric}=stats_file={log_path}:shortest=1",
        "-an",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_ffmpeg_log(log_path):
    """Extract a single average value from an ffmpeg stats file.

    The format depends on which filter was used:

    * PSNR produces lines containing "psnr_avg:XX.YY".
    * SSIM produces lines with "All:0.995234".
    * Older examples used the word "average:" but modern ffmpeg avoids that.

    We scan each line and return the first match we find.
    """
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # PSNR style
            m = re.search(r"psnr_avg:([0-9]+\.?[0-9]*)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            # SSIM style
            m = re.search(r"All:([0-9]+\.?[0-9]*)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            # fallback older style
            if "average:" in line:
                parts = line.split("average:")
                if len(parts) > 1:
                    try:
                        return float(parts[1].split()[0])
                    except ValueError:
                        pass
    return None


def compute_lpips(ref_pattern, test_pattern, fps=None):
    try:
        import torch
        import lpips
        from PIL import Image
        import numpy as np
    except ImportError:
        print("LPIPS computation requires torch, lpips and PIL. install them and retry.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)

    # if the reference is a video file instead of a frame pattern, dump
    # it to a temporary folder using ffmpeg so we can compare frame-by-frame.
    if os.path.isfile(ref_pattern) and "%" not in ref_pattern:
        if fps is None:
            fps = 2
        tmpdir = "_tmp_ref_frames"
        os.makedirs(tmpdir, exist_ok=True)
        out_pat = os.path.join(tmpdir, "frame_%06d.png")
        print(f"extracting reference video to {out_pat} ...")
        cmd = [
            "ffmpeg",
            "-i",
            ref_pattern,
            "-framerate",
            str(fps),
            os.path.join(tmpdir, "frame_%06d.png"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ref_pattern = out_pat

    # helper that expands a printf-style pattern into actual paths
    def pattern_to_paths(pattern):
        m = re.search(r"%(0?)([0-9]+)d", pattern)
        if not m:
            return [pattern]  # no formatting, return literal
        width = int(m.group(2))
        prefix, suffix = pattern.split(m.group(0))
        paths = []
        idx = 1
        while True:
            p = f"{prefix}{idx:0{width}d}{suffix}"
            if not os.path.exists(p):
                break
            paths.append(p)
            idx += 1
        return paths

    ref_paths = pattern_to_paths(ref_pattern)
    test_paths = pattern_to_paths(test_pattern)
    if len(ref_paths) != len(test_paths):
        print("Warning: reference and test sequences have different lengths")

    total = 0.0
    count = 0
    for r, t in zip(ref_paths, test_paths):
        im_r = Image.open(r).convert("RGB")
        im_t = Image.open(t).convert("RGB")
        if im_r.size != im_t.size:
            im_t = im_t.resize(im_r.size, Image.LANCZOS)
        arr_r = torch.from_numpy(np.array(im_r).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        arr_t = torch.from_numpy(np.array(im_t).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        arr_r = arr_r.to(device)
        arr_t = arr_t.to(device)
        with torch.no_grad():
            d = loss_fn(arr_r, arr_t).item()
        total += d
        count += 1
    return total / count if count > 0 else None


def main():
    p = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS between reference and test frames")
    p.add_argument("--ref", required=True, help="reference video file (will be read at --fps")
    p.add_argument(
        "--test-pattern",
        required=True,
        help="printf-style pattern for test images (e.g. scene_dense/images/frame_%06d.png)",
    )
    p.add_argument("--fps", type=float, default=2.0, help="frame rate for the reference video")
    p.add_argument("--lpips", action="store_true", help="compute LPIPS (requires lpips package)")
    args = p.parse_args()

    # run ffmpeg metrics
    # use fixed temp filenames in current working directory to avoid NamedTemporaryFile issue
    psnr_log = "psnr_eval.log"
    ssim_log = "ssim_eval.log"

    print("computing PSNR...")
    run_ffmpeg_metric(args.ref, args.test_pattern, args.fps, "psnr", psnr_log)
    psnr_val = parse_ffmpeg_log(psnr_log)
    print(f"psnr average: {psnr_val}")

    print("computing SSIM...")
    run_ffmpeg_metric(args.ref, args.test_pattern, args.fps, "ssim", ssim_log)
    ssim_val = parse_ffmpeg_log(ssim_log)
    print(f"ssim average: {ssim_val}")

    if args.lpips:
        print("computing LPIPS (alex)...")
        lpips_val = compute_lpips(args.ref, args.test_pattern, fps=args.fps)
        print(f"lpips average: {lpips_val}")


if __name__ == "__main__":
    main()
