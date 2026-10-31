#!/usr/bin/env bash
# Usage: render_ghost4_batch.sh <mlp_run_dir> <sonic_causal_dagger_run_dir> [num_motions=0(all)] [steps=192]
# For every training motion shared by MLP / C / DAgger checkpoints:
#   rollouts: sonic, cvae_c_before, cvae_dagger_after, mlp_best_survival
#   solo video per rollout + one 4-way ghost (sonic / mlp_best_survival / cvae_c_before / cvae_dagger_after).
# C_CKPT is the pre-DAgger C checkpoint (sonic_causal_train output), CVAE_DAGGER_CKPT the post-DAgger one.
# LIST_ONLY=1 prints the shared motions and exits without Isaac.
# Overrides: C_CKPT (default: student_checkpoint in the DAgger run manifest), CVAE_DAGGER_CKPT (default: best.pt).

set -Eeuo pipefail

MLP_RUN="${1:-}"
CVAE_RUN="${2:-}"
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
MLP_TAGS=(best_survival)

if [[ -z "$MLP_RUN" || -z "$CVAE_RUN" ]]; then
  echo "Usage: $0 <mlp_run_dir> <sonic_causal_dagger_run_dir> [num_motions] [steps]" >&2
  exit 2
fi
if [[ -n "${C_CKPT:-}" && "$(realpath -m "$C_CKPT")" == "$(realpath -m "${CVAE_DAGGER_CKPT:-$CVAE_RUN/checkpoints/best.pt}")" ]]; then
  echo "C_CKPT must be the pre-DAgger C checkpoint, not the DAgger best.pt" >&2
  exit 2
fi
CVAE_MANIFEST="$CVAE_RUN/manifests/sonic_causal_manifest.json"
if [[ -z "${C_CKPT:-}" ]]; then
  test -s "$CVAE_MANIFEST" || { echo "Missing $CVAE_MANIFEST; set C_CKPT" >&2; exit 2; }
  C_CKPT="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_checkpoint"])' "$CVAE_MANIFEST")"
fi
CVAE_DAGGER_CKPT="${CVAE_DAGGER_CKPT:-$CVAE_RUN/checkpoints/best.pt}"
for f in "$C_CKPT" "$CVAE_DAGGER_CKPT" "$SONIC_TEACHER_CHECKPOINT" "$MLP_RENDER_CONFIG" "$MODEL" \
         "$SONIC_KIT_ROOT/render_policy_ghost.py" "$SONIC_KIT_ROOT/render_mujoco_trajectory.py"; do
  test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }
done
for tag in "${MLP_TAGS[@]}"; do
  test -s "$MLP_RUN/checkpoints/$tag.pt" || { echo "Missing $MLP_RUN/checkpoints/$tag.pt; run keep_best_last.py first" >&2; exit 2; }
done

if [[ -z "${DAGGER_MOTION_FILE:-}" ]]; then
  POINTER="$SONIC_RUNS_ROOT/latest_overfit_reference_motion_subset_run_dir.txt"
  test -s "$POINTER" || { echo "Set DAGGER_MOTION_FILE or create $POINTER" >&2; exit 2; }
  DAGGER_MOTION_FILE="$(<"$POINTER")/data/robot_filtered"
fi
test -d "$DAGGER_MOTION_FILE" || { echo "DAGGER_MOTION_FILE is not a directory: $DAGGER_MOTION_FILE" >&2; exit 2; }

mapfile -t MOTION_KEYS < <("$PYTHON" -c '
import sys, torch
load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
m = set(load(sys.argv[1]).get("motion_keys") or [])
c = set(load(sys.argv[2]).get("train_motion_keys") or [])
d = set(load(sys.argv[3]).get("train_motion_keys") or [])
shared = sorted(m & c & d)
print(f"mlp={len(m)} c={len(c)} dagger={len(d)} shared={len(shared)}", file=sys.stderr)
n = int(sys.argv[4])
print("\n".join(shared[:n] if n > 0 else shared))
' "$MLP_RUN/checkpoints/last.pt" "$C_CKPT" "$CVAE_DAGGER_CKPT" "$NUM_MOTIONS")
[[ ${#MOTION_KEYS[@]} -gt 0 && -n "${MOTION_KEYS[0]}" ]] || { echo "No motion shared by MLP / C / DAgger checkpoints" >&2; exit 2; }
printf '[ghost4-batch] motion %s\n' "${MOTION_KEYS[@]}"
[[ "${LIST_ONLY:-0}" == 1 ]] && exit 0

export SONIC_REPO_ROOT SONIC_RUNS_ROOT SONIC_TEACHER_CHECKPOINT DAGGER_MOTION_FILE
export DAGGER_RENDER_STEPS="$STEPS"
OUT_ROOT="$SONIC_RUNS_ROOT/ghost4_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT/videos/solo" "$OUT_ROOT/videos/ghost4"
{
  echo "mlp_run=$MLP_RUN"; echo "cvae_c=$C_CKPT"; echo "cvae_dagger=$CVAE_DAGGER_CKPT"
  echo "motion_file=$DAGGER_MOTION_FILE"; echo "steps=$STEPS"; printf 'motion=%s\n' "${MOTION_KEYS[@]}"
} > "$OUT_ROOT/sources.txt"
echo "[ghost4-batch] out=$OUT_ROOT"

# Prints the trajectory path on stdout; logs go to stderr.
rollout() {
  local name="$1" policy="$2" ckpt="$3" config="$4" started run status
  started="$(date +%s)"
  echo "[ghost4-batch] rollout $name policy=$policy ckpt=${ckpt:-<teacher>}" >&2
  set +e
  DAGGER_RENDER_POLICY="$policy" DAGGER_RENDER_CHECKPOINT="$ckpt" DAGGER_RENDER_CONFIG="$config" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >&2
  status=$?
  set -e
  run="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
  if [[ -z "$run" || "$(stat -c %Y "$run")" -lt "$started" || ! -s "$run/data/000000.trajectory.pkl" ]]; then
    echo "[ghost4-batch] $name rollout failed (status=$status, run=${run:-none})" >&2
    return 1
  fi
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("[ghost4-batch] %s frames=%s/%s terminated=%s" % (sys.argv[2], m["frames"], m["requested_steps"]+1, m["terminated"]), file=sys.stderr)' "$run/manifests/dagger_policy_render_manifest.json" "$name" >&2
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
  local key="$1" safe dir name traj out
  local -A trajs=()
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  dir="$OUT_ROOT/$safe"
  mkdir -p "$dir/solo/manifests"
  export DAGGER_RENDER_MOTION_KEY="$key"
  echo "[ghost4-batch] ===== motion=$key ====="
  trajs[sonic]="$(rollout sonic sonic "" "")" || return 1
  trajs[cvae_c_before]="$(rollout cvae_c_before sonic_causal "$C_CKPT" "")" || return 1
  trajs[cvae_dagger_after]="$(rollout cvae_dagger_after sonic_causal "$CVAE_DAGGER_CKPT" "")" || return 1
  for tag in "${MLP_TAGS[@]}"; do
    trajs[mlp_$tag]="$(rollout "mlp_$tag" mlp "$MLP_RUN/checkpoints/$tag.pt" "$MLP_RENDER_CONFIG")" || return 1
  done
  for name in "${!trajs[@]}"; do
    echo "$name=${trajs[$name]}" >> "$dir/trajectories.txt"
    solo_video "${trajs[$name]}" "$dir/solo/$name.mp4" || { echo "[ghost4-batch] solo video failed: $name" >&2; return 1; }
    cp "$dir/solo/$name.mp4" "$OUT_ROOT/videos/solo/${safe}__${name}.mp4"
  done
  for tag in "${MLP_TAGS[@]}"; do
    out="$dir/ghost4_mlp_$tag"
    MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
      --layer "sonic=${trajs[sonic]}" \
      --layer "mlp_${tag}=${trajs[mlp_$tag]}" \
      --layer "cvae_c_before=${trajs[cvae_c_before]}" \
      --layer "cvae_dagger_after=${trajs[cvae_dagger_after]}" \
      --model "$MODEL" --output-run "$out" --title "${safe}_mlp_${tag}" --gl egl || return 1
    cp "$out"/videos/*.mp4 "$OUT_ROOT/videos/ghost4/${safe}__mlp_${tag}.mp4"
  done
}

FAILED=()
for key in "${MOTION_KEYS[@]}"; do
  render_motion "$key" || { echo "[ghost4-batch] motion failed: $key" >&2; FAILED+=("$key"); }
done

echo "[ghost4-batch] solo videos: $(ls "$OUT_ROOT/videos/solo" | wc -l)  ghost4 videos: $(ls "$OUT_ROOT/videos/ghost4" | wc -l)"
echo "[ghost4-batch] out=$OUT_ROOT/videos"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '[ghost4-batch] FAILED motion %s\n' "${FAILED[@]}"
  exit 1
fi
