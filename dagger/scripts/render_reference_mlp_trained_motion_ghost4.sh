#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: render_reference_mlp_trained_motion_ghost4.sh C.pt AFTER.pt motion_keys.json motion_dir [steps=192]
DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}"
C_CHECKPOINT="${1:?C checkpoint is required}"
AFTER_CHECKPOINT="${2:?after checkpoint is required}"
MOTIONS="${3:?motion key manifest is required}"
MOTION_FILE="${4:?motion directory is required}"
STEPS="${5:-192}"
TEACHER="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MODEL="${GHOST_MODEL:-$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
CONFIG="$DAGGER_ROOT/configs/mlp_render.yaml"
OUT_ROOT="$SONIC_RUNS_ROOT/reference_mlp_trained_motion_ghost4_$(date +%Y%m%d_%H%M%S)"
for f in "$C_CHECKPOINT" "$AFTER_CHECKPOINT" "$MOTIONS" "$TEACHER" "$MODEL" "$CONFIG" "$SONIC_KIT_ROOT/render_policy_ghost.py"; do test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }; done
test -d "$MOTION_FILE"
[[ "$STEPS" == 192 ]] || { echo "This contract requires 192 actions" >&2; exit 2; }
export PYTHONPATH="$DAGGER_ROOT/src:$PROJECT_ROOT/state-action-cvae/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$OUT_ROOT/cases"
mapfile -t KEYS < <("$PYTHON" - "$MOTIONS" <<'PY'
import json,sys
keys=json.load(open(sys.argv[1]))
if not isinstance(keys,list) or len(keys)!=4: raise SystemExit('motion key manifest must contain exactly four keys')
print('\n'.join(str(x) for x in keys))
PY
)
run_policy() {
  local label="$1" checkpoint="$2" key="$3" root="$4"
  mkdir -p "$root"
  set +e
  BLY_PROJECT_ROOT="$PROJECT_ROOT" SONIC_RUNS_ROOT="$root" SONIC_REPO_ROOT="$SONIC_REPO_ROOT" SONIC_KIT_ROOT="$SONIC_KIT_ROOT" \
    STATE_ACTION_CVAE_ROOT="$PROJECT_ROOT/state-action-cvae" SONIC_TEACHER_CHECKPOINT="$TEACHER" \
    DAGGER_MOTION_FILE="$MOTION_FILE" DAGGER_RENDER_POLICY="$label" DAGGER_RENDER_CHECKPOINT="$checkpoint" \
    DAGGER_RENDER_STEPS=192 DAGGER_RENDER_MOTION_KEY="$key" DAGGER_RENDER_DISABLE_MOTION_AUG=true \
    DAGGER_RENDER_KEEP_RUNNING=true DAGGER_RENDER_CONFIG="$CONFIG" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >"$root/render.log" 2>&1
  local status=$?
  set -e
  local run
  run="$(find "$root" -maxdepth 1 -type d -name 'dagger_mlp_render_*' -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"
  test "$status" = 0 && test -s "$run/data/000000.trajectory.pkl" || { echo "$label rollout failed for $key; see $root/render.log" >&2; return 1; }
  printf '%s' "$run/data/000000.trajectory.pkl"
}
for index in 0 1 2 3; do
  key="${KEYS[$index]}"
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  case_out="$OUT_ROOT/cases/case_${index}_${safe}"
  mkdir -p "$case_out"
  sonic_traj="$(run_policy sonic '' "$key" "$case_out/sonic")"
  before_traj="$(run_policy sonic_causal "$C_CHECKPOINT" "$key" "$case_out/before")"
  after_traj="$(run_policy sonic_causal "$AFTER_CHECKPOINT" "$key" "$case_out/after")"
  "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
    --layer "sonic=$sonic_traj" --layer "dagger_before=$before_traj" --layer "dagger_after=$after_traj" \
    --model "$MODEL" --ground-style black --title "${safe}_trained_threeway" --output-run "$case_out/ghost"
  "$PYTHON" - "$case_out/ghost/manifests/policy_ghost.json" "$case_out/case.json" "$key" "$C_CHECKPOINT" "$AFTER_CHECKPOINT" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if int(m.get('frames',0)) != 193: raise SystemExit(f"expected 193 frames, got {m.get('frames')}")
json.dump({'motion_key':sys.argv[3], 'c_checkpoint':sys.argv[4], 'after_checkpoint':sys.argv[5], 'ghost':m}, open(sys.argv[2],'w'), indent=2)
PY
done
"$PYTHON" - "$OUT_ROOT" "$C_CHECKPOINT" "$AFTER_CHECKPOINT" "$MOTIONS" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cases=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('case_*/case.json'))]
if len(cases)!=4: raise SystemExit(f'expected four cases, got {len(cases)}')
(root/'markers').mkdir(exist_ok=True)
json.dump({'format_version':'sonic_reference_mlp_trained_motion_ghost4_v1','c_checkpoint':str(pathlib.Path(sys.argv[2]).resolve()),'after_checkpoint':str(pathlib.Path(sys.argv[3]).resolve()),'motion_keys_manifest':str(pathlib.Path(sys.argv[4]).resolve()),'horizon_actions':192,'horizon_states':193,'layers':['sonic','dagger_before','dagger_after'],'cases':cases}, (root/'manifest.json').open('w'), indent=2)
(root/'markers/reference_mlp_trained_motion_ghost4.ok').write_text('PASS\n')
print(root)
PY
