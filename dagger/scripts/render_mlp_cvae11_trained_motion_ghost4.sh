#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: render_mlp_cvae11_trained_motion_ghost4.sh BEFORE.pt AFTER.pt motion_keys.json motion_dir dataset_run [steps=192]
DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}"
BEFORE="${1:?before checkpoint is required}"
AFTER="${2:?after checkpoint is required}"
MOTIONS="${3:?motion key manifest is required}"
MOTION_FILE="${4:?motion directory is required}"
DATASET="${5:?dataset run is required}"
STEPS="${6:-192}"
TEACHER="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MODEL="${GHOST_MODEL:-$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
CONFIG="$DAGGER_ROOT/configs/mlp_cvae11_render.yaml"
OUT_ROOT="${GHOST_OUTPUT_ROOT:-$SONIC_RUNS_ROOT/mlp_cvae11_trained_motion_ghost4_$(date +%Y%m%d_%H%M%S)}"
for f in "$BEFORE" "$AFTER" "$MOTIONS" "$TEACHER" "$MODEL" "$CONFIG" "$SONIC_KIT_ROOT/render_policy_ghost.py"; do test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }; done
test -d "$MOTION_FILE" && test -d "$DATASET"
[[ "$STEPS" == 192 ]] || { echo "This contract requires 192 actions" >&2; exit 2; }
export PYTHONPATH="$DAGGER_ROOT/src:$PROJECT_ROOT/state-action-cvae/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"
if [[ -e "$OUT_ROOT" ]] && [[ -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Ghost output must be new and empty: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT/cases"

mapfile -t KEYS < <("$PYTHON" - "$MOTIONS" <<'PY'
import json,sys
keys=json.load(open(sys.argv[1]))
if not isinstance(keys,list) or len(keys)!=4 or len(set(keys))!=4 or any(not isinstance(x,str) or not x or '\n' in x for x in keys): raise SystemExit('motion key manifest must contain exactly four distinct nonempty keys')
print('\n'.join(str(x) for x in keys))
PY
)
[[ "${#KEYS[@]}" == 4 ]] || { echo "Invalid motion key manifest" >&2; exit 2; }
run_policy() {
  local label="$1" checkpoint="$2" key="$3" root="$4"
  mkdir -p "$root"
  set +e
  SONIC_RUNS_ROOT="$root" BLY_PROJECT_ROOT="$PROJECT_ROOT" SONIC_REPO_ROOT="$SONIC_REPO_ROOT" SONIC_KIT_ROOT="$SONIC_KIT_ROOT" \
    STATE_ACTION_CVAE_ROOT="$PROJECT_ROOT/state-action-cvae" SONIC_TEACHER_CHECKPOINT="$TEACHER" \
    DAGGER_MOTION_FILE="$MOTION_FILE" CVAE_DATASET_RUN="$DATASET" \
    DAGGER_RENDER_POLICY="$label" DAGGER_RENDER_CHECKPOINT="$checkpoint" DAGGER_RENDER_STEPS=192 \
    DAGGER_RENDER_MOTION_KEY="$key" DAGGER_RENDER_DISABLE_MOTION_AUG=true \
    DAGGER_RENDER_KEEP_RUNNING=true DAGGER_RENDER_CONFIG="$CONFIG" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >"$root/render.log" 2>&1
  local status=$?
  set -e
  local run
  run="$("$PYTHON" - "$root" <<'PY'
import pathlib,sys
runs=list(pathlib.Path(sys.argv[1]).glob('dagger_mlp_render_*'))
if len(runs)!=1: raise SystemExit(f'expected one rollout in {sys.argv[1]}, got {len(runs)}')
print(runs[0])
PY
)"
  test "$status" = 0 && test -s "$run/data/000000.trajectory.pkl" && test -s "$run/markers/dagger_mlp_render.ok" || { echo "$label rollout failed for $key; see $root/render.log" >&2; return 1; }
  "$PYTHON" - "$run" "$key" "$label" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
m=json.loads((root/'manifests/dagger_policy_render_manifest.json').read_text())
if m.get('execution_pass') is not True or m.get('render_kind')!='isaac_rollout_then_mujoco_kinematic_render': raise SystemExit('missing Isaac rollout engine evidence')
if m.get('motion_keys') != [sys.argv[2]] or m.get('policy_kind') != sys.argv[3]: raise SystemExit('rollout motion/policy mismatch')
if not 1 <= int(m.get('frames',0)) <= 193: raise SystemExit('invalid original rollout frame count')
print(root/'data/000000.trajectory.pkl')
PY
}
for index in 0 1 2 3; do
  key="${KEYS[$index]}"
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  case_out="$OUT_ROOT/cases/case_${index}_${safe}"
  mkdir -p "$case_out"
  sonic_traj="$(run_policy sonic '' "$key" "$case_out/sonic")"
  before_traj="$(run_policy mlp_cvae11 "$BEFORE" "$key" "$case_out/before")"
  after_traj="$(run_policy mlp_cvae11 "$AFTER" "$key" "$case_out/after")"
  "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
    --layer "sonic=$sonic_traj" --layer "dagger_before=$before_traj" --layer "dagger_after=$after_traj" \
    --model "$MODEL" --ground-style black --title "${safe}_trained_threeway" --display-frames 193 --output-run "$case_out/ghost"
  "$PYTHON" - "$case_out/ghost/manifests/policy_ghost.json" "$case_out/case.json" "$key" "$BEFORE" "$AFTER" <<'PY'
import json,pathlib,sys
m=json.load(open(sys.argv[1]))
if int(m.get('frames',0)) != 193: raise SystemExit(f"expected 193 frames, got {m.get('frames')}")
rollouts={}
for layer in m['layers']:
    run=pathlib.Path(layer['trajectory']).parent.parent
    evidence=json.loads((run/'manifests/dagger_policy_render_manifest.json').read_text())
    if int(evidence['frames']) != int(layer['frames']): raise SystemExit('original frame count mismatch')
    rollouts[layer['name']]=evidence
json.dump({'motion_key':sys.argv[3], 'before_checkpoint':sys.argv[4], 'after_checkpoint':sys.argv[5], 'rollouts':rollouts, 'ghost':m}, open(sys.argv[2],'w'), indent=2)
PY
done
"$PYTHON" - "$OUT_ROOT" "$BEFORE" "$AFTER" "$MOTIONS" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cases=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('case_*/case.json'))]
if len(cases)!=4: raise SystemExit(f'expected four cases, got {len(cases)}')
(root/'markers').mkdir(exist_ok=True)
json.dump({'format_version':'sonic_mlp_cvae11_trained_motion_ghost4_v1','before_checkpoint':str(pathlib.Path(sys.argv[2]).resolve()),'after_checkpoint':str(pathlib.Path(sys.argv[3]).resolve()),'motion_keys_manifest':str(pathlib.Path(sys.argv[4]).resolve()),'horizon_actions':192,'horizon_states':193,'layers':['sonic','dagger_before','dagger_after'],'cases':cases}, (root/'manifest.json').open('w'), indent=2)
(root/'markers/mlp_cvae11_trained_motion_ghost4.ok').write_text('PASS\n')
print(root)
PY
