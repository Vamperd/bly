#!/usr/bin/env bash
# Usage: render_cvae_ghost4.sh <c_ckpt> <v5ref_dagger_run|-> [mlp_run|-] [num_motions=0(all)] [steps=192]
# Per motion: rollouts sonic, mlp_best_survival (only if the MLP trained on it), cvae_before (raw C),
# cvae_after (v5ref DAgger best_survival.pt); a solo video per rollout and one 4-layer ghost.
# Motions: DAgger motion_keys (or the v5 manifest when no DAgger run) restricted to C train_motion_keys.
# Every layer uses the Physics v5 reference subset with motion augmentation disabled.
# Overrides: CVAE_DAGGER_CKPT, DAGGER_MOTION_FILE, CVAE_MODEL_CONFIG, CVAE_KL2_DATASET_RUN, LIST_ONLY=1.

set -Eeuo pipefail

C_CKPT="${1:-}"
DAGGER_RUN="${2:--}"
MLP_RUN="${3:--}"
NUM_MOTIONS="${4:-0}"
STEPS="${5:-192}"
[[ "$DAGGER_RUN" == - ]] && DAGGER_RUN=""
[[ "$MLP_RUN" == - ]] && MLP_RUN=""

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${BLY_PROJECT_ROOT:-$(cd "$DAGGER_ROOT/.." && pwd)}"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$PROJECT_ROOT/state-action-cvae}"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
RENDER_CONFIG="${RENDER_CONFIG:-$DAGGER_ROOT/configs/mlp_render.yaml}"
MANIFEST="${CVAE_GHOST_MANIFEST:-$DAGGER_ROOT/configs/cvae_kl2_motion_manifest_v5.json}"
MODEL="$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
CVAE_MODEL_CONFIG="${CVAE_MODEL_CONFIG:-$STATE_ACTION_CVAE_ROOT/configs/posterior_hierarchical_standard_cvae_v5_reference_fp32.json}"
CVAE_KL2_DATASET_RUN="${CVAE_KL2_DATASET_RUN:-$SONIC_RUNS_ROOT/cvae_reference_mixed_dataset_20261103_090945}"
REFERENCE_SUBSET_POINTER="$SONIC_RUNS_ROOT/latest_overfit_reference_motion_subset_run_dir.txt"
if [[ -z "${DAGGER_MOTION_FILE:-}" ]]; then
  test -s "$REFERENCE_SUBSET_POINTER" || { echo "Set DAGGER_MOTION_FILE or create $REFERENCE_SUBSET_POINTER" >&2; exit 2; }
  DAGGER_MOTION_FILE="$(<"$REFERENCE_SUBSET_POINTER")/data/robot_filtered"
fi

if [[ -z "$C_CKPT" ]]; then
  echo "Usage: $0 <c_ckpt> <v5ref_dagger_run|-> [mlp_run|-] [num_motions] [steps]" >&2
  exit 2
fi
CVAE_DAGGER_CKPT="${CVAE_DAGGER_CKPT:-${DAGGER_RUN:+$DAGGER_RUN/checkpoints/best_survival.pt}}"
MLP_CKPT="${MLP_RUN:+$MLP_RUN/checkpoints/best_survival.pt}"
for f in "$C_CKPT" "$SONIC_TEACHER_CHECKPOINT" "$RENDER_CONFIG" "$MANIFEST" "$MODEL" "$CVAE_MODEL_CONFIG" \
         ${CVAE_DAGGER_CKPT:+"$CVAE_DAGGER_CKPT"} ${MLP_CKPT:+"$MLP_CKPT"} \
         "$SONIC_KIT_ROOT/render_policy_ghost.py" "$SONIC_KIT_ROOT/render_mujoco_trajectory.py"; do
  test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }
done
test -d "$DAGGER_MOTION_FILE" || { echo "DAGGER_MOTION_FILE is not a directory: $DAGGER_MOTION_FILE" >&2; exit 2; }
test -d "$CVAE_KL2_DATASET_RUN" || { echo "CVAE_KL2_DATASET_RUN is not a directory: $CVAE_KL2_DATASET_RUN" >&2; exit 2; }

# Each line: <motion_key> <1 if the MLP was trained on it else 0>
mapfile -t MOTIONS < <("$PYTHON" -c '
import json, sys, torch
load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
c_keys = set(load(sys.argv[1]).get("train_motion_keys") or [])
if not c_keys:
    sys.exit("C checkpoint has no train_motion_keys")
if sys.argv[2]:
    d = load(sys.argv[2])
    keys = list(d.get("motion_keys") or []) or sorted(d.get("motion_coverage") or {})
else:
    keys = list(json.load(open(sys.argv[3]))["motion_keys"])
m = set(load(sys.argv[4]).get("motion_keys") or []) if sys.argv[4] else set()
dropped = [k for k in keys if k not in c_keys]
keys = [k for k in keys if k in c_keys]
print(f"motions={len(keys)} dropped_not_in_c={len(dropped)} mlp={len(m)} shared_mlp={len(set(keys) & m)}", file=sys.stderr)
n = int(sys.argv[5])
for key in (keys[:n] if n > 0 else keys):
    print(key, int(key in m))
' "$C_CKPT" "$CVAE_DAGGER_CKPT" "$MANIFEST" "$MLP_CKPT" "$NUM_MOTIONS")
[[ ${#MOTIONS[@]} -gt 0 && -n "${MOTIONS[0]}" ]] || { echo "No motion shared by the motion list and C train_motion_keys" >&2; exit 2; }
printf '[ghost4] motion %s\n' "${MOTIONS[@]}"
[[ "${LIST_ONLY:-0}" == 1 ]] && exit 0

export SONIC_REPO_ROOT SONIC_RUNS_ROOT STATE_ACTION_CVAE_ROOT SONIC_TEACHER_CHECKPOINT DAGGER_MOTION_FILE
export CVAE_MODEL_CONFIG CVAE_KL2_DATASET_RUN
export DAGGER_RENDER_STEPS="$STEPS"
export DAGGER_RENDER_DISABLE_MOTION_AUG=true
OUT_ROOT="$SONIC_RUNS_ROOT/cvae_ghost4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT/videos/solo" "$OUT_ROOT/videos/ghost"
{
  echo "cvae_before=$C_CKPT"; echo "cvae_after=${CVAE_DAGGER_CKPT:-<none>}"; echo "mlp=${MLP_CKPT:-<none>}"
  echo "model_config=$CVAE_MODEL_CONFIG"; echo "normalization_run=$CVAE_KL2_DATASET_RUN"
  echo "motion_file=$DAGGER_MOTION_FILE"; echo "steps=$STEPS"; printf 'motion=%s\n' "${MOTIONS[@]}"
} > "$OUT_ROOT/sources.txt"
echo "[ghost4] out=$OUT_ROOT"

# Prints the trajectory path on stdout; logs go to stderr.
rollout() {
  local name="$1" policy="$2" ckpt="$3" started run status
  started="$(date +%s)"
  echo "[ghost4] rollout $name policy=$policy ckpt=${ckpt:-<teacher>}" >&2
  set +e
  DAGGER_RENDER_POLICY="$policy" DAGGER_RENDER_CHECKPOINT="$ckpt" DAGGER_RENDER_CONFIG="$RENDER_CONFIG" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >&2
  status=$?
  set -e
  run="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
  if [[ -z "$run" || "$(stat -c %Y "$run")" -lt "$started" || ! -s "$run/data/000000.trajectory.pkl" ]]; then
    echo "[ghost4] $name rollout failed (status=$status, run=${run:-none})" >&2
    return 1
  fi
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("[ghost4] %s frames=%s/%s terminated=%s" % (sys.argv[2], m["frames"], m["requested_steps"]+1, m["terminated"]), file=sys.stderr)' "$run/manifests/dagger_policy_render_manifest.json" "$name" >&2
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
  local order=(sonic)
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  dir="$OUT_ROOT/$safe"
  mkdir -p "$dir/solo/manifests"
  export DAGGER_RENDER_MOTION_KEY="$key"
  echo "[ghost4] ===== motion=$key mlp=$with_mlp ====="
  trajs[sonic]="$(rollout sonic sonic "")" || return 1
  if [[ "$with_mlp" == 1 ]]; then
    trajs[mlp_best_survival]="$(rollout mlp_best_survival mlp "$MLP_CKPT")" || return 1
    order+=(mlp_best_survival)
  fi
  trajs[cvae_before]="$(rollout cvae_before cvae "$C_CKPT")" || return 1
  order+=(cvae_before)
  if [[ -n "$CVAE_DAGGER_CKPT" ]]; then
    trajs[cvae_after]="$(rollout cvae_after cvae "$CVAE_DAGGER_CKPT")" || return 1
    order+=(cvae_after)
  fi
  local layers=()
  for name in "${order[@]}"; do
    echo "$name=${trajs[$name]}" >> "$dir/trajectories.txt"
    solo_video "${trajs[$name]}" "$dir/solo/$name.mp4" || { echo "[ghost4] solo video failed: $name" >&2; return 1; }
    cp "$dir/solo/$name.mp4" "$OUT_ROOT/videos/solo/${safe}__${name}.mp4"
    layers+=(--layer "$name=${trajs[$name]}")
  done
  out="$dir/ghost"
  MUJOCO_GL=egl "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" "${layers[@]}" \
    --model "$MODEL" --output-run "$out" --title "$safe" --gl egl || return 1
  cp "$out"/videos/*.mp4 "$OUT_ROOT/videos/ghost/${safe}.mp4"
}

FAILED=()
for line in "${MOTIONS[@]}"; do
  key="${line% *}"
  render_motion "$key" "${line##* }" || { echo "[ghost4] motion failed: $key" >&2; FAILED+=("$key"); }
done

echo "[ghost4] solo videos: $(ls "$OUT_ROOT/videos/solo" | wc -l)  ghost videos: $(ls "$OUT_ROOT/videos/ghost" | wc -l)"
echo "[ghost4] out=$OUT_ROOT/videos"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '[ghost4] FAILED motion %s\n' "${FAILED[@]}"
  exit 1
fi
