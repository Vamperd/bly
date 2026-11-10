#!/usr/bin/env bash
set -Eeuo pipefail
# Run inside an allocated GPU job, after the committed DAgger stage finishes.
# Usage: render_mlp_cvae11_128_final_ghost4.sh C_RUN D_RUN OUTPUT_STATE_DIR MIG
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
exec "$PYTHON" "$SCRIPT_DIR/mlp_cvae11_128_ghost_stage.py" render \
  "${1:?C_RUN required}" "${2:?D_RUN required}" "${3:?OUTPUT_STATE_DIR required}" "${4:?MIG required}"
