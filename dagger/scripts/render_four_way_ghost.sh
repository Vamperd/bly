#!/usr/bin/env bash
# Usage: render_four_way_ghost.sh <mlp_ckpt> <cvae_c_ckpt> <cvae_dagger_ckpt> [motion_key] [steps]
# Layers: sonic (opaque, white) / mlp_dagger (blue) / cvae_c (green) / cvae_dagger (orange).
# All four rollouts use the same reference-subset motion file and motion key.
# MLP_RENDER_CONFIG selects the MLP architecture config (e.g. dagger/configs/mlp_render_intensive.yaml).

set -Eeuo pipefail

MLP_CKPT="${1:-}"
C_CKPT="${2:-}"
DAGGER_CKPT="${3:-}"
MOTION_KEY="${4:-}"
STEPS="${5:-192}"

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MLP_RENDER_CONFIG="${MLP_RENDER_CONFIG:-$DAGGER_ROOT/configs/mlp_render.yaml}"

if [[ -z "$MLP_CKPT" || -z "$C_CKPT" || -z "$DAGGER_CKPT" ]]; then
  echo "Usage: $0 <mlp_ckpt> <cvae_c_ckpt> <cvae_dagger_ckpt> [motion_key] [steps]" >&2
  exit 2
fi
for f in "$MLP_CKPT" "$C_CKPT" "$DAGGER_CKPT" "$SONIC_TEACHER_CHECKPOINT" "$MLP_RENDER_CONFIG"; do
  test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Python is missing: $PYTHON" >&2; exit 2; }
test -f "$SONIC_KIT_ROOT/render_policy_ghost.py" || { echo "Missing $SONIC_KIT_ROOT/render_policy_ghost.py" >&2; exit 2; }
MODEL="$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
test -f "$MODEL" || { echo "G1 MuJoCo model is missing: $MODEL" >&2; exit 2; }

if [[ -z "${DAGGER_MOTION_FILE:-}" ]]; then
  POINTER="$SONIC_RUNS_ROOT/latest_overfit_reference_motion_subset_run_dir.txt"
  test -s "$POINTER" || { echo "Set DAGGER_MOTION_FILE or create $POINTER" >&2; exit 2; }
  DAGGER_MOTION_FILE="$(<"$POINTER")/data/robot_filtered"
fi
test -d "$DAGGER_MOTION_FILE" || { echo "DAGGER_MOTION_FILE is not a directory: $DAGGER_MOTION_FILE" >&2; exit 2; }

if [[ -z "$MOTION_KEY" ]]; then
  # First motion shared by the MLP checkpoint and both CVAE checkpoints.
  MOTION_KEY="$("$PYTHON" -c 'import sys,torch; l=lambda p: torch.load(p,map_location="cpu",weights_only=False); m=set(l(sys.argv[1]).get("motion_keys") or []); c=set(l(sys.argv[2]).get("train_motion_keys") or []); d=set(l(sys.argv[3]).get("train_motion_keys") or []); s=sorted(m&c&d); print(s[0] if s else "")' "$MLP_CKPT" "$C_CKPT" "$DAGGER_CKPT")"
  [[ -n "$MOTION_KEY" ]] || { echo "No motion shared by all three checkpoints; pass motion_key explicitly" >&2; exit 2; }
fi

export SONIC_REPO_ROOT SONIC_RUNS_ROOT SONIC_TEACHER_CHECKPOINT DAGGER_MOTION_FILE
export DAGGER_RENDER_STEPS="$STEPS"
export DAGGER_RENDER_MOTION_KEY="$MOTION_KEY"
echo "[ghost4] motion_key=$MOTION_KEY steps=$STEPS motion_file=$DAGGER_MOTION_FILE"

render_one() {
  local name="$1" policy="$2" ckpt="$3" config="$4" started run status
  started="$(date +%s)"
  echo "[ghost4] rollout $name policy=$policy ckpt=${ckpt:-<teacher>}" >&2
  set +e
  DAGGER_RENDER_POLICY="$policy" DAGGER_RENDER_CHECKPOINT="$ckpt" DAGGER_RENDER_CONFIG="$config" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >&2
  status=$?
  set -e
  run="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
  if [[ -z "$run" || "$(stat -c %Y "$run")" -lt "$started" || ! -s "$run/data/000000.trajectory.pkl" ]]; then
    echo "[ghost4] $name rollout failed (status=$status, run=${run:-none})" >&2
    return 1
  fi
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("[ghost4] %s frames=%s/%s terminated=%s" % (sys.argv[2], m["frames"], m["requested_steps"]+1, m["terminated"]), file=sys.stderr)' "$run/manifests/dagger_policy_render_manifest.json" "$name"
  echo "$run/data/000000.trajectory.pkl"
}

SONIC_TRAJ="$(render_one sonic sonic "" "")"
sleep 5
MLP_TRAJ="$(render_one mlp_dagger mlp "$MLP_CKPT" "$MLP_RENDER_CONFIG")"
sleep 5
C_TRAJ="$(render_one cvae_c sonic_causal "$C_CKPT" "")"
sleep 5
DAGGER_TRAJ="$(render_one cvae_dagger sonic_causal "$DAGGER_CKPT" "")"

SAFE_KEY="$(printf '%s' "$MOTION_KEY" | tr -c 'A-Za-z0-9_.-' '_')"
OUT="$SONIC_RUNS_ROOT/policy_ghost4_${SAFE_KEY}_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
  --layer "sonic=$SONIC_TRAJ" \
  --layer "mlp_dagger=$MLP_TRAJ" \
  --layer "cvae_c=$C_TRAJ" \
  --layer "cvae_dagger=$DAGGER_TRAJ" \
  --model "$MODEL" \
  --output-run "$OUT" \
  --title "$SAFE_KEY" \
  --gl egl

printf 'sonic=%s\nmlp_dagger=%s\ncvae_c=%s\ncvae_dagger=%s\n' "$SONIC_TRAJ" "$MLP_TRAJ" "$C_TRAJ" "$DAGGER_TRAJ" > "$OUT/manifests/ghost4_sources.txt"
echo "[ghost4] output=$OUT"
ls "$OUT/videos"
