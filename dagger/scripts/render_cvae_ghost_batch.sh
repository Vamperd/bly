#!/usr/bin/env bash
# Usage: render_cvae_ghost_batch.sh <cvae_kl2_dagger_run_dir> [mlp_run_dir] [num_motions=0(all)] [steps=192]
# For every training motion of the 65-token CVAE DAgger checkpoint:
#   rollouts: sonic, cvae_dagger, and mlp_best_survival when the MLP was trained on that motion.
#   solo video per rollout + one ghost (sonic opaque / mlp / cvae_dagger).
# All rollouts use the KL2 motion file so every layer tracks the same reference.
# Overrides: CVAE_DAGGER_CKPT (default best_survival.pt), DAGGER_MOTION_FILE, LIST_ONLY=1.

set -Eeuo pipefail

CVAE_RUN="${1:-}"
MLP_RUN="${2:-}"
NUM_MOTIONS="${3:-0}"
STEPS="${4:-192}"

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MLP_RENDER_CONFIG="${MLP_RENDER_CONFIG:-$DAGGER_ROOT/configs/mlp_render.yaml}"
MODEL="$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-${CVAE_KL2_MOTION_FILE:-/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered}}"

if [[ -z "$CVAE_RUN" ]]; then
  echo "Usage: $0 <cvae_kl2_dagger_run_dir> [mlp_run_dir] [num_motions] [steps]" >&2
  exit 2
fi
CVAE_DAGGER_CKPT="${CVAE_DAGGER_CKPT:-$CVAE_RUN/checkpoints/best_survival.pt}"
MLP_CKPT=""
[[ -n "$MLP_RUN" ]] && MLP_CKPT="$MLP_RUN/checkpoints/best_survival.pt"
for f in "$CVAE_DAGGER_CKPT" "$SONIC_TEACHER_CHECKPOINT" "$MLP_RENDER_CONFIG" "$MODEL" ${MLP_CKPT:+"$MLP_CKPT"} \
         "$SONIC_KIT_ROOT/render_policy_ghost.py" "$SONIC_KIT_ROOT/render_mujoco_trajectory.py"; do
  test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }
done
test -d "$DAGGER_MOTION_FILE" || { echo "DAGGER_MOTION_FILE is not a directory: $DAGGER_MOTION_FILE" >&2; exit 2; }

# Each line: <motion_key> <1 if the MLP was trained on it else 0>
mapfile -t MOTIONS < <("$PYTHON" -c '
import sys, torch
load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
c = load(sys.argv[1])
keys = list(c.get("motion_keys") or []) or sorted(c.get("motion_coverage") or {})
m = set(load(sys.argv[2]).get("motion_keys") or []) if sys.argv[2] else set()
print(f"cvae={len(keys)} mlp={len(m)} shared={len(set(keys) & m)}", file=sys.stderr)
n = int(sys.argv[3])
for key in (keys[:n] if n > 0 else keys):
    print(key, int(key in m))
' "$CVAE_DAGGER_CKPT" "$MLP_CKPT" "$NUM_MOTIONS")
[[ ${#MOTIONS[@]} -gt 0 && -n "${MOTIONS[0]}" ]] || { echo "CVAE DAgger checkpoint has no motion_keys" >&2; exit 2; }
printf '[cvae-ghost] motion %s\n' "${MOTIONS[@]}"
[[ "${LIST_ONLY:-0}" == 1 ]] && exit 0

export SONIC_REPO_ROOT SONIC_RUNS_ROOT SONIC_TEACHER_CHECKPOINT DAGGER_MOTION_FILE
export DAGGER_RENDER_STEPS="$STEPS"
OUT_ROOT="$SONIC_RUNS_ROOT/cvae_ghost_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT/videos/solo" "$OUT_ROOT/videos/ghost"
{
  echo "cvae_dagger=$CVAE_DAGGER_CKPT"; echo "mlp=${MLP_CKPT:-<none>}"
  echo "motion_file=$DAGGER_MOTION_FILE"; echo "steps=$STEPS"; printf 'motion=%s\n' "${MOTIONS[@]}"
} > "$OUT_ROOT/sources.txt"
echo "[cvae-ghost] out=$OUT_ROOT"

# Prints the trajectory path on stdout; logs go to stderr.
rollout() {
  local name="$1" policy="$2" ckpt="$3" config="$4" started run status
  started="$(date +%s)"
  echo "[cvae-ghost] rollout $name policy=$policy ckpt=${ckpt:-<teacher>}" >&2
  set +e
  DAGGER_RENDER_POLICY="$policy" DAGGER_RENDER_CHECKPOINT="$ckpt" DAGGER_RENDER_CONFIG="$config" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >&2
  status=$?
  set -e
  run="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
  if [[ -z "$run" || "$(stat -c %Y "$run")" -lt "$started" || ! -s "$run/data/000000.trajectory.pkl" ]]; then
    echo "[cvae-ghost] $name rollout failed (status=$status, run=${run:-none})" >&2
    return 1
  fi
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("[cvae-ghost] %s frames=%s/%s terminated=%s" % (sys.argv[2], m["frames"], m["requested_steps"]+1, m["terminated"]), file=sys.stderr)' "$run/manifests/dagger_policy_render_manifest.json" "$name" >&2
  sleep 5
  echo "$run/data/000000.trajectory.pkl"
}

solo_video() {
  local traj="$1" video="$2" dir
  dir="$(dirname "$video")"
  MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_mujoco_trajectory.py" \
    --trajectory "$traj" --model "$MODEL" --output "$video" --manifest-dir "$dir/manifests" \
    --width 960 --height 540 --gl egl --camera-distance 2.0 >&2 || return 1
  test -s "$video"
}

render_motion() {
  local key="$1" with_mlp="$2" safe dir name out
  local -A trajs=()
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  dir="$OUT_ROOT/$safe"
  mkdir -p "$dir/solo/manifests"
  export DAGGER_RENDER_MOTION_KEY="$key"
  echo "[cvae-ghost] ===== motion=$key mlp=$with_mlp ====="
  trajs[sonic]="$(rollout sonic sonic "" "")" || return 1
  if [[ "$with_mlp" == 1 ]]; then
    trajs[mlp_best_survival]="$(rollout mlp_best_survival mlp "$MLP_CKPT" "$MLP_RENDER_CONFIG")" || return 1
  fi
  trajs[cvae_dagger]="$(rollout cvae_dagger cvae "$CVAE_DAGGER_CKPT" "")" || return 1
  for name in "${!trajs[@]}"; do
    echo "$name=${trajs[$name]}" >> "$dir/trajectories.txt"
    solo_video "${trajs[$name]}" "$dir/solo/$name.mp4" || { echo "[cvae-ghost] solo video failed: $name" >&2; return 1; }
    cp "$dir/solo/$name.mp4" "$OUT_ROOT/videos/solo/${safe}__${name}.mp4"
  done
  out="$dir/ghost"
  local layers=(--layer "sonic=${trajs[sonic]}")
  [[ "$with_mlp" == 1 ]] && layers+=(--layer "mlp_best_survival=${trajs[mlp_best_survival]}")
  layers+=(--layer "cvae_dagger=${trajs[cvae_dagger]}")
  MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" "${layers[@]}" \
    --model "$MODEL" --output-run "$out" --title "$safe" --gl egl || return 1
  cp "$out"/videos/*.mp4 "$OUT_ROOT/videos/ghost/${safe}.mp4"
}

FAILED=()
for line in "${MOTIONS[@]}"; do
  key="${line% *}"
  render_motion "$key" "${line##* }" || { echo "[cvae-ghost] motion failed: $key" >&2; FAILED+=("$key"); }
done

echo "[cvae-ghost] solo videos: $(ls "$OUT_ROOT/videos/solo" | wc -l)  ghost videos: $(ls "$OUT_ROOT/videos/ghost" | wc -l)"
echo "[cvae-ghost] out=$OUT_ROOT/videos"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '[cvae-ghost] FAILED motion %s\n' "${FAILED[@]}"
  exit 1
fi
