#!/usr/bin/env bash
# Export the 70-D reference table for the Physics v5 motions (no DAgger lock: safe beside a running DAgger job).

set -Eeuo pipefail

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$HOME/bly/state-action-cvae}"
PHYSICS_DATASET="${CVAE_V5_PHYSICS_DATASET:-$SONIC_RUNS_ROOT/cvae_physics_dataset_v5}"
if [[ -z "${CVAE_V5_MOTION_FILE:-}" ]]; then
  POINTER="$SONIC_RUNS_ROOT/latest_overfit_reference_motion_subset_run_dir.txt"
  test -s "$POINTER" || {
    echo "Set CVAE_V5_MOTION_FILE or create $POINTER" >&2
    exit 2
  }
  CVAE_V5_MOTION_FILE="$(<"$POINTER")/data/robot_filtered"
fi
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
test -x "$PYTHON" || { echo "Python is missing: $PYTHON" >&2; exit 2; }
test -d "$PHYSICS_DATASET" || { echo "Physics v5 dataset is missing: $PHYSICS_DATASET" >&2; exit 2; }
test -d "$CVAE_V5_MOTION_FILE" || { echo "Motion directory is missing: $CVAE_V5_MOTION_FILE" >&2; exit 2; }

export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$SONIC_RUNS_ROOT/logs"
LOG_PATH="$SONIC_RUNS_ROOT/logs/export_reference_table_$(date +%Y%m%d_%H%M%S).log"
echo "[export_reference_table] dataset=$PHYSICS_DATASET motion_file=$CVAE_V5_MOTION_FILE log=$LOG_PATH"
set +e
"$PYTHON" -m dagger.export_reference_table \
  --physics-dataset "$PHYSICS_DATASET" \
  --motion-file "$CVAE_V5_MOTION_FILE" \
  --sonic-root "$SONIC_REPO_ROOT" \
  --output-root "$SONIC_RUNS_ROOT" \
  --device "${DAGGER_DEVICE:-cuda:0}" 2>&1 | tee -a "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e
echo "[export_reference_table] exit_code=$STATUS log=$LOG_PATH"
exit "$STATUS"
