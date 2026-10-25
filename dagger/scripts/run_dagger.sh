#!/usr/bin/env bash

set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in
  smoke|pilot) CONFIG_MODE="$MODE" ;;
  cvae-smoke) CONFIG_MODE="cvae_smoke" ;;
  cvae-pilot) CONFIG_MODE="cvae_pilot" ;;
  mlp-smoke) CONFIG_MODE="mlp_smoke" ;;
  mlp-pilot) CONFIG_MODE="mlp_pilot" ;;
  mlp-debug) CONFIG_MODE="mlp_debug" ;;
  *)
    echo "Usage: $0 smoke|pilot|cvae-smoke|cvae-pilot|mlp-smoke|mlp-pilot|mlp-debug" >&2
  exit 2
    ;;
esac

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$HOME/bly/state-action-cvae}"
DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered}"

export DAGGER_ROOT SONIC_REPO_ROOT SONIC_TEACHER_CHECKPOINT SONIC_RUNS_ROOT STATE_ACTION_CVAE_ROOT DAGGER_MOTION_FILE
export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
[[ -x "$PYTHON" ]] || {
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 2
}

test -d "$SONIC_REPO_ROOT" || {
  echo "SONIC repository is missing: $SONIC_REPO_ROOT" >&2
  exit 2
}

test -s "$SONIC_TEACHER_CHECKPOINT" || {
  echo "Teacher checkpoint is missing or empty: $SONIC_TEACHER_CHECKPOINT" >&2
  exit 2
}

CONFIG="$DAGGER_ROOT/configs/${CONFIG_MODE}.yaml"
test -f "$CONFIG" || {
  echo "DAgger config is missing: $CONFIG" >&2
  exit 2
}

mkdir -p "$SONIC_RUNS_ROOT"

if [[ "$MODE" == mlp-smoke || "$MODE" == mlp-pilot || "$MODE" == mlp-debug ]]; then
  test -d "$DAGGER_MOTION_FILE" || {
    echo "DAGGER_MOTION_FILE is missing or not a directory: $DAGGER_MOTION_FILE" >&2
    exit 2
  }
fi

if [[ "$MODE" == cvae-smoke || "$MODE" == cvae-pilot ]]; then
  : "${CVAE_STUDENT_CHECKPOINT:?Set CVAE_STUDENT_CHECKPOINT to a Stage-C CVAE checkpoint}"
  : "${CVAE_BASE_DATASET_RUN:?Set CVAE_BASE_DATASET_RUN to a validated CVAE dataset run}"
  test -s "$CVAE_STUDENT_CHECKPOINT" || {
    echo "CVAE student checkpoint is missing or empty: $CVAE_STUDENT_CHECKPOINT" >&2
    exit 2
  }
  test -d "$CVAE_BASE_DATASET_RUN" || {
    echo "CVAE base dataset run is missing: $CVAE_BASE_DATASET_RUN" >&2
    exit 2
  }
fi

LOG_PATH="$SONIC_RUNS_ROOT/dagger_${MODE}_command_$(date +%Y%m%d_%H%M%S).log"
echo "[run_dagger] CONFIG=$CONFIG" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_MOTION_FILE=$DAGGER_MOTION_FILE" >> "$LOG_PATH"
echo "[run_dagger] SONIC_TEACHER_CHECKPOINT=$SONIC_TEACHER_CHECKPOINT" >> "$LOG_PATH"
set +e
if [[ "${DAGGER_TEE:-0}" == "1" ]]; then
  "$PYTHON" -m dagger.cli --config "$CONFIG" 2>&1 | tee -a "$LOG_PATH"
  STATUS=${PIPESTATUS[0]}
else
  "$PYTHON" -m dagger.cli --config "$CONFIG" >> "$LOG_PATH" 2>&1
  STATUS=$?
fi
set -e
echo "[run_dagger] dagger_exit_code=$STATUS" >> "$LOG_PATH"
echo "[run_dagger] log=$LOG_PATH"
echo "[run_dagger] dagger_exit_code=$STATUS"
exit "$STATUS"
