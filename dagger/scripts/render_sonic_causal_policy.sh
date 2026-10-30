#!/usr/bin/env bash
# Usage: render_sonic_causal_policy.sh <checkpoint> [motion_key] [steps]
# An empty motion_key picks a seeded-random motion from the checkpoint's train_motion_keys.

set -Eeuo pipefail

CHECKPOINT="${1:-${DAGGER_RENDER_CHECKPOINT:-}}"
MOTION_KEY="${2:-${DAGGER_RENDER_MOTION_KEY:-}}"
STEPS="${3:-${DAGGER_RENDER_STEPS:-192}}"

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 <checkpoint> [motion_key] [steps]" >&2
  exit 2
fi
test -s "$CHECKPOINT" || { echo "Checkpoint is missing: $CHECKPOINT" >&2; exit 2; }
test -x "$PYTHON" || { echo "Python is missing: $PYTHON" >&2; exit 2; }
test -f "$SONIC_KIT_ROOT/render_mujoco_trajectory.py" || {
  echo "MuJoCo renderer is missing: $SONIC_KIT_ROOT/render_mujoco_trajectory.py" >&2
  exit 2
}
MODEL="$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
test -f "$MODEL" || { echo "G1 MuJoCo model is missing: $MODEL" >&2; exit 2; }

export SONIC_REPO_ROOT SONIC_RUNS_ROOT SONIC_TEACHER_CHECKPOINT
export DAGGER_RENDER_POLICY=sonic_causal
export DAGGER_RENDER_CHECKPOINT="$CHECKPOINT"
export DAGGER_RENDER_STEPS="$STEPS"
export DAGGER_RENDER_MOTION_KEY="$MOTION_KEY"

echo "[render-sonic-causal] checkpoint=$CHECKPOINT motion_key=${MOTION_KEY:-<random C-trained>} steps=$STEPS latent=${DAGGER_RENDER_LATENT_MODE:-fixed_standard_normal}"

STARTED_AT="$(date +%s)"
set +e
bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render
RUN_STATUS=$?
set -e

RENDER_RUN="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
if [[ -z "$RENDER_RUN" || "$(stat -c %Y "$RENDER_RUN")" -lt "$STARTED_AT" ]]; then
  echo "Render run was not created (status=$RUN_STATUS)." >&2
  exit 1
fi
TRAJECTORY="$RENDER_RUN/data/000000.trajectory.pkl"
if [[ ! -s "$TRAJECTORY" ]]; then
  echo "Trajectory was not written (run status=$RUN_STATUS): $TRAJECTORY" >&2
  exit 1
fi

mkdir -p "$RENDER_RUN/videos" "$RENDER_RUN/markers"
VIDEO="$RENDER_RUN/videos/sonic_causal_policy.mp4"

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
printf 'PASS\n' > "$RENDER_RUN/markers/render_sonic_causal.ok"
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("[render-sonic-causal] motion=%s frames=%s/%s terminated=%s" % (m["motion_keys"], m["frames"], m["requested_steps"]+1, m["terminated"]))' "$RENDER_RUN/manifests/dagger_policy_render_manifest.json"
echo "[render-sonic-causal] trajectory=$TRAJECTORY"
echo "[render-sonic-causal] video=$VIDEO"
