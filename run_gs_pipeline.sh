#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_SCRIPT="$SCRIPT_DIR/run_gs_pipeline_core.sh"
GUI_SCRIPT="$SCRIPT_DIR/gui_monitor.py"

if [[ ! -x "$CORE_SCRIPT" ]]; then
  echo "Error: missing core pipeline script: $CORE_SCRIPT" >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec "$CORE_SCRIPT" "$@"
  fi
done

if [[ "${RUN_GS_PIPELINE_NO_GUI:-0}" == "1" ]]; then
  exec "$CORE_SCRIPT" "$@"
fi

if [[ -n "${DISPLAY:-}" ]] && [[ -f "$GUI_SCRIPT" ]] && [[ -x /usr/bin/python3 ]]; then
  exec /usr/bin/python3 "$GUI_SCRIPT" -- "$@"
fi

exec "$CORE_SCRIPT" "$@"
