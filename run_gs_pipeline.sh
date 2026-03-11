#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_help() {
  cat <<'USAGE'
Quality-first pipeline for Gaussian Splatting data prep using ffmpeg + COLMAP.

# set COLMAP_BIN to the path of your colmap executable if it's not on your PATH

Inputs:
  - Video file (recommended for this script), OR
  - Existing image directory

Outputs:
  - Sparse model + points3D.ply
  - Optional dense fused cloud .ply

Usage:
  ./run_gs_pipeline.sh \
    --input /path/to/video_or_images \
    --workspace /path/to/workspace \
    [--profile quality|fast_hq|robust_hq] \
    [--dense-profile balanced|hq] \
    [--dense-input-type geometric|photometric] \
    [--fps 2] \
    [--max-images 0] \
    [--max-image-size 3200] \
    [--camera-model OPENCV] \
    [--matcher exhaustive|sequential|auto] \
    [--sift-max-num-features 12000] \
    [--sift-peak-threshold 0.004] \
    [--sift-match-max-ratio 0.85] \
    [--sift-match-min-inliers 12] \
    [--mapper-min-num-matches 15] \
    [--mapper-init-min-num-inliers 80] \
    [--mapper-abs-pose-min-inliers 20] \
    [--mapper-filter-min-tri-angle 1.0] \
    [--mapper-tri-min-angle 1.0] \
    [--mapper-tri-complete-max-transitivity 8] \
    [--benchmark-ref /path/to/ref_video_or_pattern] \
    [--benchmark-test-pattern /path/to/test/frame_%06d.png] \
    [--benchmark-fps 2] \
    [--benchmark-lpips] \
    [--benchmark-min-psnr 30] \
    [--benchmark-min-ssim 0.95] \
    [--benchmark-max-lpips 0.20] \
    [--benchmark-json /path/to/metrics.json] \
    [--mask-path /path/to/masks] \
    [--print-train-cmd] \
    [--train-script /path/to/train.py] \
    [--train-iterations 45000] \
    [--train-densify-grad-threshold 0.0001] \
    [--train-opacity-reset-interval 3000] \
    [--cpu] \
    [--no-sparse-densify] \
    [--dense]

Examples:
  # Best quality from video, slower
  ./run_gs_pipeline.sh --input ./capture.mp4 --workspace ./scene01 --fps 2 --dense

  # Faster high-quality preset
  ./run_gs_pipeline.sh --input ./capture.mp4 --workspace ./scene_fast --profile fast_hq

  # Use existing frames/images directory
  ./run_gs_pipeline.sh --input ./images --workspace ./scene02 --matcher exhaustive

  # End-to-end reconstruction + benchmark gate (recommended for hold-out renders)
  ./run_gs_pipeline.sh \
    --input ./capture.mp4 \
    --workspace ./scene_eval \
    --profile robust_hq \
    --dense --dense-profile hq \
    --benchmark-ref ./holdout_gt/frame_%06d.png \
    --benchmark-test-pattern ./holdout_render/frame_%06d.png \
    --benchmark-lpips \
    --benchmark-min-psnr 28 \
    --benchmark-min-ssim 0.92 \
    --benchmark-max-lpips 0.22
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

is_nonneg_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

INPUT=""
WORKSPACE=""
PROFILE="quality"
FPS="2"
MAX_IMAGES="0"
MAX_IMAGE_SIZE="3200"
CAMERA_MODEL="OPENCV"
MASK_PATH=""
MATCHER="auto"
DO_DENSE="0"
DO_SPARSE_DENSIFY="1"
DENSE_PROFILE="balanced"
DENSE_INPUT_TYPE="geometric"
SIFT_MAX_NUM_FEATURES="12000"
SIFT_PEAK_THRESHOLD="0.004"
SIFT_MATCH_MAX_RATIO="0.85"
SIFT_MATCH_MIN_INLIERS="12"
MAPPER_MIN_NUM_MATCHES="15"
MAPPER_INIT_MIN_NUM_INLIERS="80"
MAPPER_ABS_POSE_MIN_INLIERS="20"
MAPPER_FILTER_MIN_TRI_ANGLE="1.0"
MAPPER_TRI_MIN_ANGLE="1.0"
MAPPER_TRI_COMPLETE_MAX_TRANSITIVITY="8"
USE_GPU="1"
SIFT_EXTRACTION_NUM_THREADS="-1"
SIFT_MATCHING_NUM_THREADS="-1"
EXHAUSTIVE_BLOCK_SIZE="50"

PATCHMATCH_MAX_IMAGE_SIZE="-1"
PATCHMATCH_NUM_ITERATIONS="5"
PATCHMATCH_WINDOW_RADIUS="5"
PATCHMATCH_FILTER_MIN_NCC="0.1"
PATCHMATCH_FILTER_MIN_TRI_ANGLE="3"
PATCHMATCH_FILTER_MIN_NUM_CONSISTENT="2"
PATCHMATCH_GEOM_CONSISTENCY_MAX_COST="1"

FUSION_MAX_IMAGE_SIZE="-1"
FUSION_MIN_NUM_PIXELS="5"
FUSION_MAX_REPROJ_ERROR="2"
FUSION_MAX_DEPTH_ERROR="0.01"
FUSION_MAX_NORMAL_ERROR="10"

PRINT_TRAIN_CMD="0"
TRAIN_SCRIPT="train.py"
TRAIN_ITERATIONS="45000"
TRAIN_DENSIFY_GRAD_THRESHOLD="0.0001"
TRAIN_OPACITY_RESET_INTERVAL="3000"

BENCHMARK_REF=""
BENCHMARK_TEST_PATTERN=""
BENCHMARK_FPS=""
BENCHMARK_LPIPS="0"
BENCHMARK_MIN_PSNR=""
BENCHMARK_MIN_SSIM=""
BENCHMARK_MAX_LPIPS=""
BENCHMARK_JSON=""
BENCHMARK_PYTHON="python3"
BENCHMARK_EVAL_SCRIPT="$SCRIPT_DIR/evaluate.py"
RUN_BENCHMARK="0"

USER_SET_FPS="0"
USER_SET_MAX_IMAGE_SIZE="0"
USER_SET_MATCHER="0"
USER_SET_SIFT_MAX_NUM_FEATURES="0"
USER_SET_SIFT_PEAK_THRESHOLD="0"
USER_SET_SIFT_MATCH_MIN_INLIERS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --dense-profile)
      DENSE_PROFILE="$2"
      shift 2
      ;;
    --dense-input-type)
      DENSE_INPUT_TYPE="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      USER_SET_FPS="1"
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE="$2"
      USER_SET_MAX_IMAGE_SIZE="1"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --camera-model)
      CAMERA_MODEL="$2"
      shift 2
      ;;
    --mask-path)
      MASK_PATH="$2"
      shift 2
      ;;
    --matcher)
      MATCHER="$2"
      USER_SET_MATCHER="1"
      shift 2
      ;;
    --sift-max-num-features)
      SIFT_MAX_NUM_FEATURES="$2"
      USER_SET_SIFT_MAX_NUM_FEATURES="1"
      shift 2
      ;;
    --sift-peak-threshold)
      SIFT_PEAK_THRESHOLD="$2"
      USER_SET_SIFT_PEAK_THRESHOLD="1"
      shift 2
      ;;
    --sift-match-max-ratio)
      SIFT_MATCH_MAX_RATIO="$2"
      shift 2
      ;;
    --sift-match-min-inliers)
      SIFT_MATCH_MIN_INLIERS="$2"
      USER_SET_SIFT_MATCH_MIN_INLIERS="1"
      shift 2
      ;;
    --mapper-min-num-matches)
      MAPPER_MIN_NUM_MATCHES="$2"
      shift 2
      ;;
    --mapper-init-min-num-inliers)
      MAPPER_INIT_MIN_NUM_INLIERS="$2"
      shift 2
      ;;
    --mapper-abs-pose-min-inliers)
      MAPPER_ABS_POSE_MIN_INLIERS="$2"
      shift 2
      ;;
    --mapper-filter-min-tri-angle)
      MAPPER_FILTER_MIN_TRI_ANGLE="$2"
      shift 2
      ;;
    --mapper-tri-min-angle)
      MAPPER_TRI_MIN_ANGLE="$2"
      shift 2
      ;;
    --mapper-tri-complete-max-transitivity)
      MAPPER_TRI_COMPLETE_MAX_TRANSITIVITY="$2"
      shift 2
      ;;
    --benchmark-ref)
      BENCHMARK_REF="$2"
      shift 2
      ;;
    --benchmark-test-pattern)
      BENCHMARK_TEST_PATTERN="$2"
      shift 2
      ;;
    --benchmark-fps)
      BENCHMARK_FPS="$2"
      shift 2
      ;;
    --benchmark-lpips|--benchmark-lpsis)
      BENCHMARK_LPIPS="1"
      shift
      ;;
    --benchmark-min-psnr)
      BENCHMARK_MIN_PSNR="$2"
      shift 2
      ;;
    --benchmark-min-ssim)
      BENCHMARK_MIN_SSIM="$2"
      shift 2
      ;;
    --benchmark-max-lpips)
      BENCHMARK_MAX_LPIPS="$2"
      shift 2
      ;;
    --benchmark-json)
      BENCHMARK_JSON="$2"
      shift 2
      ;;
    --benchmark-python)
      BENCHMARK_PYTHON="$2"
      shift 2
      ;;
    --benchmark-eval-script)
      BENCHMARK_EVAL_SCRIPT="$2"
      shift 2
      ;;
    --print-train-cmd)
      PRINT_TRAIN_CMD="1"
      shift
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --train-iterations)
      TRAIN_ITERATIONS="$2"
      shift 2
      ;;
    --train-densify-grad-threshold)
      TRAIN_DENSIFY_GRAD_THRESHOLD="$2"
      shift 2
      ;;
    --train-opacity-reset-interval)
      TRAIN_OPACITY_RESET_INTERVAL="$2"
      shift 2
      ;;
    --cpu)
      USE_GPU="0"
      shift
      ;;
    --no-sparse-densify)
      DO_SPARSE_DENSIFY="0"
      shift
      ;;
    --dense)
      DO_DENSE="1"
      shift
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ "$PROFILE" != "quality" && "$PROFILE" != "fast_hq" && "$PROFILE" != "robust_hq" ]]; then
  echo "Error: --profile must be quality, fast_hq, or robust_hq." >&2
  exit 1
fi

if [[ "$DENSE_PROFILE" != "balanced" && "$DENSE_PROFILE" != "hq" ]]; then
  echo "Error: --dense-profile must be balanced or hq." >&2
  exit 1
fi

if [[ "$DENSE_INPUT_TYPE" != "geometric" && "$DENSE_INPUT_TYPE" != "photometric" ]]; then
  echo "Error: --dense-input-type must be geometric or photometric." >&2
  exit 1
fi

if [[ "$PROFILE" == "fast_hq" ]]; then
  if [[ "$USER_SET_FPS" -eq 0 ]]; then FPS="1.5"; fi
  if [[ "$USER_SET_MAX_IMAGE_SIZE" -eq 0 ]]; then MAX_IMAGE_SIZE="2400"; fi
  if [[ "$USER_SET_MATCHER" -eq 0 ]]; then MATCHER="sequential"; fi
  if [[ "$USER_SET_SIFT_MAX_NUM_FEATURES" -eq 0 ]]; then SIFT_MAX_NUM_FEATURES="9000"; fi
  if [[ "$USER_SET_SIFT_PEAK_THRESHOLD" -eq 0 ]]; then SIFT_PEAK_THRESHOLD="0.0045"; fi
  if [[ "$USER_SET_SIFT_MATCH_MIN_INLIERS" -eq 0 ]]; then SIFT_MATCH_MIN_INLIERS="10"; fi
fi

if [[ "$PROFILE" == "robust_hq" ]]; then
  if [[ "$USER_SET_FPS" -eq 0 ]]; then FPS="1.5"; fi
  if [[ "$USER_SET_MAX_IMAGE_SIZE" -eq 0 ]]; then MAX_IMAGE_SIZE="3000"; fi
  if [[ "$USER_SET_MATCHER" -eq 0 ]]; then MATCHER="auto"; fi
  if [[ "$USER_SET_SIFT_MAX_NUM_FEATURES" -eq 0 ]]; then SIFT_MAX_NUM_FEATURES="18000"; fi
  if [[ "$USER_SET_SIFT_PEAK_THRESHOLD" -eq 0 ]]; then SIFT_PEAK_THRESHOLD="0.0035"; fi
  if [[ "$USER_SET_SIFT_MATCH_MIN_INLIERS" -eq 0 ]]; then SIFT_MATCH_MIN_INLIERS="10"; fi
fi

# Dense presets can be used independently from sparse profile.
if [[ "$DENSE_PROFILE" == "hq" ]]; then
  PATCHMATCH_MAX_IMAGE_SIZE="$MAX_IMAGE_SIZE"
  PATCHMATCH_NUM_ITERATIONS="7"
  PATCHMATCH_WINDOW_RADIUS="5"
  PATCHMATCH_FILTER_MIN_NCC="0.08"
  PATCHMATCH_FILTER_MIN_TRI_ANGLE="2"
  PATCHMATCH_FILTER_MIN_NUM_CONSISTENT="3"
  PATCHMATCH_GEOM_CONSISTENCY_MAX_COST="1.2"

  FUSION_MAX_IMAGE_SIZE="$MAX_IMAGE_SIZE"
  FUSION_MIN_NUM_PIXELS="4"
  FUSION_MAX_REPROJ_ERROR="2.5"
  FUSION_MAX_DEPTH_ERROR="0.015"
  FUSION_MAX_NORMAL_ERROR="12"
fi

if [[ -z "$INPUT" || -z "$WORKSPACE" ]]; then
  echo "Error: --input and --workspace are required." >&2
  print_help
  exit 1
fi

if [[ ! -e "$INPUT" ]]; then
  echo "Error: input path does not exist: $INPUT" >&2
  exit 1
fi

if [[ -n "$MASK_PATH" && ! -d "$MASK_PATH" ]]; then
  echo "Error: --mask-path must be an existing directory: $MASK_PATH" >&2
  exit 1
fi

if [[ "$MATCHER" != "exhaustive" && "$MATCHER" != "sequential" && "$MATCHER" != "auto" ]]; then
  echo "Error: --matcher must be exhaustive, sequential, or auto." >&2
  exit 1
fi

if [[ ! "$MAX_IMAGES" =~ ^[0-9]+$ ]]; then
  echo "Error: --max-images must be a non-negative integer." >&2
  exit 1
fi

if [[ -n "$BENCHMARK_REF" || -n "$BENCHMARK_TEST_PATTERN" || "$BENCHMARK_LPIPS" -eq 1 || -n "$BENCHMARK_MIN_PSNR" || -n "$BENCHMARK_MIN_SSIM" || -n "$BENCHMARK_MAX_LPIPS" ]]; then
  RUN_BENCHMARK="1"
fi

if [[ "$RUN_BENCHMARK" -eq 1 ]]; then
  if [[ -z "$BENCHMARK_REF" || -z "$BENCHMARK_TEST_PATTERN" ]]; then
    echo "Error: benchmarking requires both --benchmark-ref and --benchmark-test-pattern." >&2
    exit 1
  fi
  if [[ ! -f "$BENCHMARK_EVAL_SCRIPT" ]]; then
    echo "Error: benchmark evaluator not found at $BENCHMARK_EVAL_SCRIPT" >&2
    exit 1
  fi

  if [[ -z "$BENCHMARK_FPS" ]]; then
    BENCHMARK_FPS="$FPS"
  fi

  if ! is_nonneg_number "$BENCHMARK_FPS"; then
    echo "Error: --benchmark-fps must be a positive number." >&2
    exit 1
  fi
  if ! awk -v v="$BENCHMARK_FPS" 'BEGIN{exit (v > 0)?0:1}'; then
    echo "Error: --benchmark-fps must be greater than 0." >&2
    exit 1
  fi

  if [[ -n "$BENCHMARK_MIN_PSNR" ]] && ! is_nonneg_number "$BENCHMARK_MIN_PSNR"; then
    echo "Error: --benchmark-min-psnr must be a non-negative number." >&2
    exit 1
  fi
  if [[ -n "$BENCHMARK_MIN_SSIM" ]] && ! is_nonneg_number "$BENCHMARK_MIN_SSIM"; then
    echo "Error: --benchmark-min-ssim must be a non-negative number." >&2
    exit 1
  fi
  if [[ -n "$BENCHMARK_MAX_LPIPS" ]] && ! is_nonneg_number "$BENCHMARK_MAX_LPIPS"; then
    echo "Error: --benchmark-max-lpips must be a non-negative number." >&2
    exit 1
  fi
fi

case "$CAMERA_MODEL" in
  SIMPLE_PINHOLE|PINHOLE|SIMPLE_RADIAL|RADIAL|OPENCV|OPENCV_FISHEYE|FULL_OPENCV|FOV|SIMPLE_RADIAL_FISHEYE|RADIAL_FISHEYE|THIN_PRISM_FISHEYE)
    ;;
  *)
    echo "Error: unsupported --camera-model '$CAMERA_MODEL'." >&2
    echo "Supported models include OPENCV and RADIAL (recommended for phones)." >&2
    exit 1
    ;;
esac

if [[ "$CAMERA_MODEL" != "OPENCV" && "$CAMERA_MODEL" != "RADIAL" ]]; then
  echo "Note: --camera-model=$CAMERA_MODEL. For most phone captures, OPENCV or RADIAL is usually more robust." >&2
fi

if [[ "$MAX_IMAGE_SIZE" -lt 1600 ]]; then
  echo "Warning: --max-image-size=$MAX_IMAGE_SIZE is low; this may lose detail for COLMAP/3DGS." >&2
  echo "         Prefer >=1600 (often 2000-4000) when VRAM allows." >&2
fi

# allow user to override the colmap executable (e.g. full path or custom build)
COLMAP_BIN="${COLMAP_BIN:-colmap}"

# if the user didn't set COLMAP_BIN and the plain name isn't on $PATH,
# try the snap location which is the default for snap installs.
# we can't run `command -v` directly inside [[ ]]; use a separate test.
if [[ "$COLMAP_BIN" == "colmap" ]]; then
  if ! command -v colmap >/dev/null 2>&1 && [[ -x "/snap/bin/colmap" ]]; then
    COLMAP_BIN="/snap/bin/colmap"
    echo "Note: colmap not found on PATH, using $COLMAP_BIN" >&2
  fi
fi

require_cmd ffmpeg
require_cmd "$COLMAP_BIN"  # this will check the provided path or the default name
if [[ "$RUN_BENCHMARK" -eq 1 ]]; then
  require_cmd "$BENCHMARK_PYTHON"
fi

# SiftGPU matching can crash when no working X/GL display is available.
DISPLAY_OK=1
if [[ -z "${DISPLAY:-}" ]]; then
  DISPLAY_OK=0
elif command -v xdpyinfo >/dev/null 2>&1; then
  if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    DISPLAY_OK=0
  fi
fi

if [[ "$DISPLAY_OK" -eq 0 ]]; then
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
  if [[ "$USE_GPU" -eq 1 ]]; then
    echo "Note: no usable display detected, falling back to CPU SIFT extraction/matching." >&2
    USE_GPU="0"
  fi
fi

if [[ "$USE_GPU" -eq 0 ]]; then
  # Keep memory usage manageable for CPU SIFT + exhaustive matching.
  SIFT_EXTRACTION_NUM_THREADS="8"
  SIFT_MATCHING_NUM_THREADS="8"
  EXHAUSTIVE_BLOCK_SIZE="20"
fi

# COLMAP option namespaces changed in newer releases (e.g. 3.14):
#   SiftExtraction.use_gpu -> FeatureExtraction.use_gpu
#   SiftMatching.use_gpu   -> FeatureMatching.use_gpu
#   SiftMatching.min_num_inliers -> TwoViewGeometry.min_num_inliers
FEATURE_EXTRACTOR_HELP="$("$COLMAP_BIN" feature_extractor -h 2>&1 || true)"
MATCHER_HELP="$("$COLMAP_BIN" exhaustive_matcher -h 2>&1 || true)"

if grep -q -- "--FeatureExtraction.use_gpu" <<<"$FEATURE_EXTRACTOR_HELP"; then
  FE_USE_GPU_OPT="--FeatureExtraction.use_gpu"
  FE_NUM_THREADS_OPT="--FeatureExtraction.num_threads"
  FE_MAX_IMAGE_SIZE_OPT="--FeatureExtraction.max_image_size"
else
  FE_USE_GPU_OPT="--SiftExtraction.use_gpu"
  FE_NUM_THREADS_OPT="--SiftExtraction.num_threads"
  FE_MAX_IMAGE_SIZE_OPT="--SiftExtraction.max_image_size"
fi

if grep -q -- "--FeatureMatching.use_gpu" <<<"$MATCHER_HELP"; then
  FM_USE_GPU_OPT="--FeatureMatching.use_gpu"
  FM_NUM_THREADS_OPT="--FeatureMatching.num_threads"
  FM_GUIDED_OPT="--FeatureMatching.guided_matching"
else
  FM_USE_GPU_OPT="--SiftMatching.use_gpu"
  FM_NUM_THREADS_OPT="--SiftMatching.num_threads"
  FM_GUIDED_OPT="--SiftMatching.guided_matching"
fi

if grep -q -- "--TwoViewGeometry.min_num_inliers" <<<"$MATCHER_HELP"; then
  FM_MIN_INLIERS_OPT="--TwoViewGeometry.min_num_inliers"
else
  FM_MIN_INLIERS_OPT="--SiftMatching.min_num_inliers"
fi

WORKSPACE="$(mkdir -p "$WORKSPACE" && cd "$WORKSPACE" && pwd)"
IMAGES_DIR="$WORKSPACE/images"
SPARSE_DIR="$WORKSPACE/sparse"
DENSE_DIR="$WORKSPACE/dense"
DB_PATH="$WORKSPACE/database.db"

if [[ "$RUN_BENCHMARK" -eq 1 && -z "$BENCHMARK_JSON" ]]; then
  BENCHMARK_JSON="$WORKSPACE/benchmark_metrics.json"
fi

mkdir -p "$IMAGES_DIR" "$SPARSE_DIR"

is_video=0
if [[ -f "$INPUT" ]]; then
  case "${INPUT##*.}" in
    mp4|MP4|mov|MOV|mkv|MKV|avi|AVI|webm|WEBM)
      is_video=1
      ;;
  esac
fi

if [[ "$is_video" -eq 1 ]]; then
  if command -v ffprobe >/dev/null 2>&1; then
    src_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT" | tr -d '[:space:]')
    if [[ -n "$src_width" && "$MAX_IMAGE_SIZE" -lt "$src_width" ]]; then
      echo "Note: source width is ${src_width}px and COLMAP features are clamped to ${MAX_IMAGE_SIZE}px." >&2
      echo "      Raise --max-image-size if your GPU memory allows for better seed detail." >&2
    fi
  fi

  echo "[1/9] Extracting high-quality frames with ffmpeg..."
  # PNG frames avoid JPEG artifacts and preserve detail for SIFT matching.
  ffmpeg -hide_banner -loglevel error -y \
    -i "$INPUT" \
    -vf "fps=${FPS}" \
    "$IMAGES_DIR/frame_%06d.png"
else
  echo "[1/9] Copying input images directory..."
  if [[ ! -d "$INPUT" ]]; then
    echo "Error: input is not a recognized video file or an image directory." >&2
    exit 1
  fi
  find "$INPUT" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -exec cp {} "$IMAGES_DIR" \;
fi

img_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')
if [[ "$img_count" -lt 20 ]]; then
  echo "Warning: only $img_count images found. Reconstruction quality may be limited." >&2
fi

if [[ "$MAX_IMAGES" -gt 0 && "$img_count" -gt "$MAX_IMAGES" ]]; then
  echo "Note: subsampling images uniformly from $img_count to $MAX_IMAGES (--max-images)." >&2
  mapfile -t all_images < <(find "$IMAGES_DIR" -maxdepth 1 -type f | sort)
  declare -A keep_images=()
  if [[ "$MAX_IMAGES" -eq 1 ]]; then
    keep_images["${all_images[0]}"]=1
  else
    for ((i=0; i<MAX_IMAGES; i++)); do
      idx=$(( i * (img_count - 1) / (MAX_IMAGES - 1) ))
      keep_images["${all_images[$idx]}"]=1
    done
  fi
  for img in "${all_images[@]}"; do
    if [[ -z "${keep_images[$img]:-}" ]]; then
      rm -f "$img"
    fi
  done
  img_count="$MAX_IMAGES"
fi

MATCHER_EFFECTIVE="$MATCHER"
if [[ "$MATCHER" == "auto" ]]; then
  if [[ "$img_count" -le 350 ]]; then
    MATCHER_EFFECTIVE="exhaustive"
  else
    MATCHER_EFFECTIVE="sequential"
  fi
  echo "Note: --matcher auto selected '$MATCHER_EFFECTIVE' for $img_count images." >&2
fi

if [[ -f "$DB_PATH" ]]; then
  rm -f "$DB_PATH"
fi

FEATURE_EXTRACTOR_MASK_OPTS=()
if [[ -n "$MASK_PATH" ]]; then
  FEATURE_EXTRACTOR_MASK_OPTS=(--ImageReader.mask_path "$MASK_PATH")
  echo "Note: using masks from $MASK_PATH (same relative names as images, mask out dynamic objects/shadows)." >&2
fi

echo "[2/9] Running COLMAP feature extraction (quality-first settings)..."
"$COLMAP_BIN" feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --ImageReader.camera_model "$CAMERA_MODEL" \
  --ImageReader.single_camera 1 \
  "${FEATURE_EXTRACTOR_MASK_OPTS[@]}" \
  "$FE_NUM_THREADS_OPT" "$SIFT_EXTRACTION_NUM_THREADS" \
  "$FE_USE_GPU_OPT" "$USE_GPU" \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1 \
  "$FE_MAX_IMAGE_SIZE_OPT" "$MAX_IMAGE_SIZE" \
  --SiftExtraction.max_num_features "$SIFT_MAX_NUM_FEATURES" \
  --SiftExtraction.peak_threshold "$SIFT_PEAK_THRESHOLD"

run_matching() {
  local mode="$1"
  echo "[3/9] Running COLMAP matching ($mode)..."
  if [[ "$mode" == "exhaustive" ]]; then
    "$COLMAP_BIN" exhaustive_matcher \
      --database_path "$DB_PATH" \
      "$FM_NUM_THREADS_OPT" "$SIFT_MATCHING_NUM_THREADS" \
      "$FM_USE_GPU_OPT" "$USE_GPU" \
      "$FM_GUIDED_OPT" 1 \
      --SiftMatching.max_ratio "$SIFT_MATCH_MAX_RATIO" \
      "$FM_MIN_INLIERS_OPT" "$SIFT_MATCH_MIN_INLIERS" \
      --ExhaustiveMatching.block_size "$EXHAUSTIVE_BLOCK_SIZE"
  else
    "$COLMAP_BIN" sequential_matcher \
      --database_path "$DB_PATH" \
      "$FM_NUM_THREADS_OPT" "$SIFT_MATCHING_NUM_THREADS" \
      "$FM_USE_GPU_OPT" "$USE_GPU" \
      "$FM_GUIDED_OPT" 1 \
      --SiftMatching.max_ratio "$SIFT_MATCH_MAX_RATIO" \
      "$FM_MIN_INLIERS_OPT" "$SIFT_MATCH_MIN_INLIERS"
  fi
}

run_matching "$MATCHER_EFFECTIVE"

evaluate_sparse_model() {
  local model_dir="$1"
  local eval_dir="$SPARSE_DIR/.eval_model"
  local points=0
  local images=0

  rm -rf "$eval_dir"
  mkdir -p "$eval_dir"

  if "$COLMAP_BIN" model_converter \
    --input_path "$model_dir" \
    --output_path "$eval_dir" \
    --output_type TXT >/dev/null 2>&1; then
    if [[ -f "$eval_dir/points3D.txt" ]]; then
      points="$(grep -vc '^#' "$eval_dir/points3D.txt" || true)"
    fi
    if [[ -f "$eval_dir/images.txt" ]]; then
      images="$(awk 'BEGIN{c=0} !/^#/ && NF>0 {c++} END{print int(c/2)}' "$eval_dir/images.txt")"
    fi
  fi

  rm -rf "$eval_dir"
  echo "$images $points"
}

run_mapper_attempt() {
  local label="$1"
  local min_matches="$2"
  local init_inliers="$3"
  local abs_inliers="$4"
  local filter_tri="$5"
  local tri_angle="$6"
  local transitivity="$7"
  local model_dir="$SPARSE_DIR/0"
  local stats=""
  local images=0
  local points=0

  rm -rf "$SPARSE_DIR/0"
  echo "  - Mapper attempt: $label (min_matches=$min_matches, init_inliers=$init_inliers, abs_inliers=$abs_inliers, tri_angle=$tri_angle)"
  "$COLMAP_BIN" mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.min_num_matches "$min_matches" \
    --Mapper.init_min_num_inliers "$init_inliers" \
    --Mapper.abs_pose_min_num_inliers "$abs_inliers" \
    --Mapper.filter_min_tri_angle "$filter_tri" \
    --Mapper.tri_min_angle "$tri_angle" \
    --Mapper.tri_complete_max_transitivity "$transitivity" \
    --Mapper.tri_ignore_two_view_tracks 0 \
    --Mapper.multiple_models 0

  if [[ ! -d "$model_dir" ]]; then
    echo "    -> No sparse model produced."
    return 1
  fi

  stats="$(evaluate_sparse_model "$model_dir")"
  images="$(awk '{print $1}' <<<"$stats")"
  points="$(awk '{print $2}' <<<"$stats")"
  echo "    -> Sparse model stats: images=$images points=$points"

  if [[ "$images" -ge 8 && "$points" -ge 100 ]]; then
    return 0
  fi
  return 1
}

echo "[4/9] Running sparse mapping..."
mapper_ok=0
if run_mapper_attempt \
  "primary" \
  "$MAPPER_MIN_NUM_MATCHES" \
  "$MAPPER_INIT_MIN_NUM_INLIERS" \
  "$MAPPER_ABS_POSE_MIN_INLIERS" \
  "$MAPPER_FILTER_MIN_TRI_ANGLE" \
  "$MAPPER_TRI_MIN_ANGLE" \
  "$MAPPER_TRI_COMPLETE_MAX_TRANSITIVITY"; then
  mapper_ok=1
else
  echo "Warning: sparse model is weak/empty, retrying mapper with safer thresholds..." >&2
  if run_mapper_attempt \
    "retry_relaxed" \
    "10" \
    "50" \
    "12" \
    "0.7" \
    "0.7" \
    "12"; then
    mapper_ok=1
  fi
fi

if [[ "$mapper_ok" -ne 1 && "$MATCHER_EFFECTIVE" != "sequential" ]]; then
  echo "Warning: sparse mapping failed with $MATCHER_EFFECTIVE matches; retrying with sequential matcher..." >&2
  run_matching "sequential"
  if run_mapper_attempt \
    "sequential_retry_primary" \
    "$MAPPER_MIN_NUM_MATCHES" \
    "$MAPPER_INIT_MIN_NUM_INLIERS" \
    "$MAPPER_ABS_POSE_MIN_INLIERS" \
    "$MAPPER_FILTER_MIN_TRI_ANGLE" \
    "$MAPPER_TRI_MIN_ANGLE" \
    "$MAPPER_TRI_COMPLETE_MAX_TRANSITIVITY"; then
    mapper_ok=1
    MATCHER_EFFECTIVE="sequential"
  else
    if run_mapper_attempt \
      "sequential_retry_relaxed" \
      "10" \
      "50" \
      "12" \
      "0.7" \
      "0.7" \
      "12"; then
      mapper_ok=1
      MATCHER_EFFECTIVE="sequential"
    fi
  fi
fi

MODEL_PATH="$SPARSE_DIR/0"
if [[ "$mapper_ok" -ne 1 || ! -d "$MODEL_PATH" ]]; then
  echo "Error: no usable sparse model generated (expected $MODEL_PATH)." >&2
  echo "Try: --matcher sequential --fps 1.5 (or 1), stable camera motion around object, and less motion blur." >&2
  exit 1
fi

if [[ "$DO_SPARSE_DENSIFY" -eq 1 ]]; then
  TRIANGULATED_MODEL_PATH="$SPARSE_DIR/triangulated"
  mkdir -p "$TRIANGULATED_MODEL_PATH"
  echo "[5/9] Running sparse point densification (point_triangulator)..."
  if "$COLMAP_BIN" point_triangulator \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --input_path "$MODEL_PATH" \
    --output_path "$TRIANGULATED_MODEL_PATH" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.min_num_matches "$MAPPER_MIN_NUM_MATCHES" \
    --Mapper.init_min_num_inliers "$MAPPER_INIT_MIN_NUM_INLIERS" \
    --Mapper.abs_pose_min_num_inliers "$MAPPER_ABS_POSE_MIN_INLIERS" \
    --Mapper.filter_min_tri_angle "$MAPPER_FILTER_MIN_TRI_ANGLE" \
    --Mapper.tri_min_angle "$MAPPER_TRI_MIN_ANGLE" \
    --Mapper.tri_complete_max_transitivity "$MAPPER_TRI_COMPLETE_MAX_TRANSITIVITY" \
    --Mapper.tri_ignore_two_view_tracks 0; then
    if [[ -d "$TRIANGULATED_MODEL_PATH" ]]; then
      MODEL_PATH="$TRIANGULATED_MODEL_PATH"
    fi
  else
    echo "Warning: point triangulation densification failed; continuing with mapper output." >&2
  fi
else
  echo "[5/9] Sparse densification skipped (--no-sparse-densify)."
fi

final_stats="$(evaluate_sparse_model "$MODEL_PATH")"
final_images="$(awk '{print $1}' <<<"$final_stats")"
final_points="$(awk '{print $2}' <<<"$final_stats")"
if [[ "$final_points" -lt 1 ]]; then
  echo "Error: sparse model has 0 points; stopping before writing empty PLY outputs." >&2
  echo "Try: --matcher sequential --fps 1.5, or capture with stronger parallax and less blur." >&2
  exit 1
fi

echo "[6/9] Converting sparse model to PLY..."
"$COLMAP_BIN" model_converter \
  --input_path "$MODEL_PATH" \
  --output_path "$MODEL_PATH/points3D_sparse.ply" \
  --output_type PLY

echo "[7/9] Exporting COLMAP text model (for gs training tools that require txt)..."
mkdir -p "$MODEL_PATH/text"
"$COLMAP_BIN" model_converter \
  --input_path "$MODEL_PATH" \
  --output_path "$MODEL_PATH/text" \
  --output_type TXT

if [[ "$DO_DENSE" -eq 1 ]]; then
  if "$COLMAP_BIN" patch_match_stereo -h >/dev/null 2>&1; then
    mkdir -p "$DENSE_DIR"
    echo "[8/9] Running image undistortion + PatchMatch stereo..."
    "$COLMAP_BIN" image_undistorter \
      --image_path "$IMAGES_DIR" \
      --input_path "$MODEL_PATH" \
      --output_path "$DENSE_DIR" \
      --output_type COLMAP

    "$COLMAP_BIN" patch_match_stereo \
      --workspace_path "$DENSE_DIR" \
      --workspace_format COLMAP \
      --PatchMatchStereo.max_image_size "$PATCHMATCH_MAX_IMAGE_SIZE" \
      --PatchMatchStereo.num_iterations "$PATCHMATCH_NUM_ITERATIONS" \
      --PatchMatchStereo.window_radius "$PATCHMATCH_WINDOW_RADIUS" \
      --PatchMatchStereo.geom_consistency true \
      --PatchMatchStereo.filter_min_ncc "$PATCHMATCH_FILTER_MIN_NCC" \
      --PatchMatchStereo.filter_min_triangulation_angle "$PATCHMATCH_FILTER_MIN_TRI_ANGLE" \
      --PatchMatchStereo.filter_min_num_consistent "$PATCHMATCH_FILTER_MIN_NUM_CONSISTENT" \
      --PatchMatchStereo.filter_geom_consistency_max_cost "$PATCHMATCH_GEOM_CONSISTENCY_MAX_COST"

    echo "[9/9] Fusing dense depth maps to PLY..."
    "$COLMAP_BIN" stereo_fusion \
      --workspace_path "$DENSE_DIR" \
      --workspace_format COLMAP \
      --input_type "$DENSE_INPUT_TYPE" \
      --StereoFusion.max_image_size "$FUSION_MAX_IMAGE_SIZE" \
      --StereoFusion.min_num_pixels "$FUSION_MIN_NUM_PIXELS" \
      --StereoFusion.max_reproj_error "$FUSION_MAX_REPROJ_ERROR" \
      --StereoFusion.max_depth_error "$FUSION_MAX_DEPTH_ERROR" \
      --StereoFusion.max_normal_error "$FUSION_MAX_NORMAL_ERROR" \
      --output_path "$DENSE_DIR/fused_dense.ply"
  else
    echo "[8/9] Dense step requested but skipped: this COLMAP build requires CUDA for patch_match_stereo." >&2
    echo "[9/9] Done (sparse outputs only)."
    DO_DENSE="0"
  fi
else
  echo "[8/9] Dense step skipped (use --dense for densest PLY on CUDA-enabled COLMAP)."
  echo "[9/9] Done."
fi

echo "\nPipeline complete. Key outputs:"
echo "  Sparse PLY: $MODEL_PATH/points3D_sparse.ply"
echo "  Sparse TXT: $MODEL_PATH/text"
if [[ "$DO_DENSE" -eq 1 ]]; then
  echo "  Dense PLY : $DENSE_DIR/fused_dense.ply"
fi

if [[ "$RUN_BENCHMARK" -eq 1 ]]; then
  echo
  echo "[benchmark] Evaluating paired outputs with PSNR/SSIM/LPIPS..."
  BENCHMARK_CMD=(
    "$BENCHMARK_PYTHON" "$BENCHMARK_EVAL_SCRIPT"
    --ref "$BENCHMARK_REF"
    --test-pattern "$BENCHMARK_TEST_PATTERN"
    --fps "$BENCHMARK_FPS"
    --json-out "$BENCHMARK_JSON"
  )
  if [[ "$BENCHMARK_LPIPS" -eq 1 ]]; then
    BENCHMARK_CMD+=(--lpips)
  fi
  if [[ -n "$BENCHMARK_MIN_PSNR" ]]; then
    BENCHMARK_CMD+=(--min-psnr "$BENCHMARK_MIN_PSNR")
  fi
  if [[ -n "$BENCHMARK_MIN_SSIM" ]]; then
    BENCHMARK_CMD+=(--min-ssim "$BENCHMARK_MIN_SSIM")
  fi
  if [[ -n "$BENCHMARK_MAX_LPIPS" ]]; then
    BENCHMARK_CMD+=(--max-lpips "$BENCHMARK_MAX_LPIPS")
  fi

  set +e
  "${BENCHMARK_CMD[@]}"
  bench_status=$?
  set -e

  if [[ "$bench_status" -eq 2 ]]; then
    echo "Benchmark thresholds failed. Metrics were written to: $BENCHMARK_JSON" >&2
    exit 2
  fi
  if [[ "$bench_status" -ne 0 ]]; then
    echo "Benchmark execution failed. Check input paths and metric dependencies." >&2
    exit 1
  fi
  echo "Benchmark metrics JSON: $BENCHMARK_JSON"
fi

if [[ "$PRINT_TRAIN_CMD" -eq 1 ]]; then
  cat <<EOF

Suggested 3DGS training command (original train.py style):
  python $TRAIN_SCRIPT \\
    -s $WORKSPACE \\
    -m $WORKSPACE/gs_model \\
    --iterations $TRAIN_ITERATIONS \\
    --densify_grad_threshold $TRAIN_DENSIFY_GRAD_THRESHOLD \\
    --opacity_reset_interval $TRAIN_OPACITY_RESET_INTERVAL

Notes:
  - Lower --densify_grad_threshold can recover fine details but increases Gaussian count.
  - Higher --iterations can improve complex scenes but may overfit.
  - Keep --opacity_reset_interval enabled to prune floaters/junk Gaussians.
EOF
fi
