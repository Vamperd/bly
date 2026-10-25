#!/usr/bin/env bash

set -Eeuo pipefail

MOTION_KEY="${1:-${DAGGER_RENDER_MOTION_KEY:-}}"
STEPS="${2:-${DAGGER_RENDER_STEPS:-192}}"

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"

if [[ -z "$MOTION_KEY" && -n "${DAGGER_RENDER_REFERENCE_RUN:-}" ]]; then
  REF_MANIFEST="$DAGGER_RENDER_REFERENCE_RUN/manifests/dagger_mlp_manifest.json"
  [[ -f "$REF_MANIFEST" ]] || REF_MANIFEST="$DAGGER_RENDER_REFERENCE_RUN/manifests/dagger_mlp_manifest.partial.json"
  if [[ -f "$REF_MANIFEST" ]]; then
    MOTION_KEY="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); keys=d.get("motion_keys_loaded") or d.get("motion", {}).get("loaded_keys") or []; print(keys[0] if keys else "")' "$REF_MANIFEST")"
  fi
fi
if [[ -z "$MOTION_KEY" ]]; then
  echo "Usage: $0 <motion_key> [steps]" >&2
  echo "Or set DAGGER_RENDER_MOTION_KEY / DAGGER_RENDER_REFERENCE_RUN." >&2
  exit 2
fi

test -d "$SONIC_REPO_ROOT" || { echo "SONIC_REPO_ROOT is missing: $SONIC_REPO_ROOT" >&2; exit 2; }
test -d "$DAGGER_MOTION_FILE" || { echo "DAGGER_MOTION_FILE is missing: $DAGGER_MOTION_FILE" >&2; exit 2; }
test -s "$SONIC_TEACHER_CHECKPOINT" || { echo "Teacher checkpoint is missing: $SONIC_TEACHER_CHECKPOINT" >&2; exit 2; }
test -x "$PYTHON" || { echo "Python is missing: $PYTHON" >&2; exit 2; }
test -f "$SONIC_KIT_ROOT/render_mujoco_trajectory.py" || {
  echo "MuJoCo renderer is missing: $SONIC_KIT_ROOT/render_mujoco_trajectory.py" >&2
  exit 2
}
MODEL="$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
test -f "$MODEL" || { echo "G1 MuJoCo model is missing: $MODEL" >&2; exit 2; }

export SONIC_REPO_ROOT SONIC_RUNS_ROOT DAGGER_MOTION_FILE SONIC_TEACHER_CHECKPOINT
export DAGGER_RENDER_POLICY=sonic
export DAGGER_RENDER_STEPS="$STEPS"
export DAGGER_RENDER_MOTION_KEY="$MOTION_KEY"

echo "[render-sonic] motion_key=$MOTION_KEY steps=$STEPS"

set +e
bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render
RUN_STATUS=$?
set -e

RENDER_RUN="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
if [[ -z "$RENDER_RUN" ]]; then
  echo "Render run was not created (status=$RUN_STATUS)." >&2
  exit 1
fi
TRAJECTORY="$RENDER_RUN/data/000000.trajectory.pkl"
if [[ ! -s "$TRAJECTORY" ]]; then
  echo "Trajectory was not written (run status=$RUN_STATUS): $TRAJECTORY" >&2
  exit 1
fi

mkdir -p "$RENDER_RUN/videos" "$RENDER_RUN/markers"
VIDEO="$RENDER_RUN/videos/sonic_policy.mp4"

MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_mujoco_trajectory.py" \
  --trajectory "$TRAJECTORY" \
  --model "$MODEL" \
  --output "$VIDEO" \
  --manifest-dir "$RENDER_RUN/manifests" \
  --width 960 \
  --height 540 \
  --gl egl \
  --camera-distance 2.0

test -s "$VIDEO" || { echo "MP4 was not produced: $VIDEO" >&2; exit 1; }
printf 'PASS\n' > "$RENDER_RUN/markers/render_sonic.ok"
echo "[render-sonic] trajectory=$TRAJECTORY"
echo "[render-sonic] video=$VIDEO"
