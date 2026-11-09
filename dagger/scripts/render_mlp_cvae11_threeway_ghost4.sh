#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: render_mlp_cvae11_threeway_ghost4.sh BEFORE.pt AFTER.pt cases.json env.yaml dataset_run [steps=192]
DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}"
BEFORE="${1:?before checkpoint is required}"
AFTER="${2:?after checkpoint is required}"
CASES="${3:?cases manifest is required}"
ENV_CONFIG="${4:?Isaac config is required}"
DATASET="${5:?dataset run is required}"
STEPS="${6:-192}"
MOTION_FILE="${DAGGER_MOTION_FILE:?DAGGER_MOTION_FILE is required}"
TEACHER="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MODEL="${GHOST_MODEL:-$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
OUT_ROOT="${SONIC_RUNS_ROOT}/mlp_cvae11_threeway_ghost4_$(date +%Y%m%d_%H%M%S)"

for f in "$BEFORE" "$AFTER" "$CASES" "$ENV_CONFIG" "$TEACHER" "$MODEL" "$SONIC_KIT_ROOT/render_policy_ghost.py"; do test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }; done
test -d "$DATASET" && test -d "$MOTION_FILE"
[[ "$STEPS" == 192 ]] || { echo "This contract requires 192 actions" >&2; exit 2; }
export PYTHONPATH="$DAGGER_ROOT/src:$PROJECT_ROOT/state-action-cvae/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$OUT_ROOT/cases"

COUNT="$($PYTHON - "$CASES" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if len(x.get('cases',[])) != 4: raise SystemExit('cases.json must contain exactly four cases')
print(4)
PY
)"
[[ "$COUNT" == 4 ]]

run_replay() {
  local label="$1" checkpoint="$2" case_json="$3" out="$4"
  "$PYTHON" -m dagger.mlp_cvae11_recorded_state_replay \
    --cases-manifest "$CASES" --case-index "$case_json" --config "$ENV_CONFIG" \
    --checkpoint "$checkpoint" --dataset-run "$DATASET" --output-run "$out" --device "${DAGGER_MLP_DEVICE:-cuda:0}"
  test -s "$out/data/mlp_cvae11.trajectory.pkl"
}

run_sonic() {
  local key="$1" root="$2"
  mkdir -p "$root"
  set +e
  SONIC_RUNS_ROOT="$root" DAGGER_MOTION_FILE="$MOTION_FILE" \
    SONIC_REPO_ROOT="$SONIC_REPO_ROOT" SONIC_TEACHER_CHECKPOINT="$TEACHER" \
    DAGGER_RENDER_POLICY=sonic DAGGER_RENDER_CHECKPOINT= DAGGER_RENDER_STEPS=192 \
    DAGGER_RENDER_MOTION_KEY="$key" DAGGER_RENDER_DISABLE_MOTION_AUG=true \
    DAGGER_RENDER_KEEP_RUNNING=true bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >"$root/sonic.log" 2>&1
  local status=$?
  set -e
  local run
  run="$(find "$root" -maxdepth 1 -type d -name 'dagger_mlp_render_*' -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"
  test "$status" = 0 && test -s "$run/data/000000.trajectory.pkl" || { echo "SONIC rollout failed for $key" >&2; return 1; }
  printf '%s' "$run/data/000000.trajectory.pkl"
}

for INDEX in 0 1 2 3; do
  KEY="$($PYTHON - "$CASES" "$INDEX" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['cases'][int(sys.argv[2])]['motion_key'])
PY
)"
  SAFE="$(printf '%s' "$KEY" | tr -c 'A-Za-z0-9_.-' '_')"
  CASE_OUT="$OUT_ROOT/cases/case_${INDEX}_${SAFE}"
  mkdir -p "$CASE_OUT"
  run_replay before "$BEFORE" "$INDEX" "$CASE_OUT/before"
  run_replay after "$AFTER" "$INDEX" "$CASE_OUT/after"
  SONIC_TRAJ="$(run_sonic "$KEY" "$CASE_OUT/sonic_runs")"
  BEFORE_TRAJ="$CASE_OUT/before/data/mlp_cvae11.trajectory.pkl"
  AFTER_TRAJ="$CASE_OUT/after/data/mlp_cvae11.trajectory.pkl"
  "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
    --layer "sonic=$SONIC_TRAJ" --layer "dagger_before=$BEFORE_TRAJ" --layer "dagger_after=$AFTER_TRAJ" \
    --model "$MODEL" --ground-style black --title "${SAFE}_threeway" \
    --output-run "$CASE_OUT/ghost"
  "$PYTHON" - "$CASE_OUT/ghost/manifests/policy_ghost.json" "$CASE_OUT/case.json" "$KEY" "$BEFORE" "$AFTER" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if int(m.get('frames',0)) != 193: raise SystemExit(f"expected 193 frames, got {m.get('frames')}")
json.dump({'motion_key':sys.argv[3], 'before_checkpoint':sys.argv[4], 'after_checkpoint':sys.argv[5], 'ghost':m}, open(sys.argv[2],'w'), indent=2)
PY
done

"$PYTHON" - "$OUT_ROOT" "$BEFORE" "$AFTER" "$CASES" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cases=[]
for p in sorted((root/'cases').glob('case_*/case.json')): cases.append(json.loads(p.read_text()))
if len(cases)!=4: raise SystemExit(f'expected four cases, got {len(cases)}')
(root/'markers').mkdir(exist_ok=True)
json.dump({'format_version':'sonic_mlp_cvae11_threeway_ghost4_v1','before_checkpoint':str(pathlib.Path(sys.argv[2]).resolve()),'after_checkpoint':str(pathlib.Path(sys.argv[3]).resolve()),'cases_manifest':str(pathlib.Path(sys.argv[4]).resolve()),'horizon_actions':192,'horizon_states':193,'layers':['sonic','dagger_before','dagger_after'],'cases':cases}, (root/'manifest.json').open('w'), indent=2)
(root/'markers/mlp_cvae11_threeway_ghost4.ok').write_text('PASS\n')
print(root)
PY
