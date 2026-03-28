#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/monitor_pipeline.py" \
  --workspace "$SCRIPT_DIR/demo" \
  -- \
  --input "$SCRIPT_DIR/scene.mp4" \
  --workspace "$SCRIPT_DIR/demo" \
  --fps 1 \
  --max-images 20 \
  --dense \
  "$@"
