#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: render_reference_mlp_threeway_ghost4.sh C.pt AFTER.pt cases.json motion_dir [steps=192]
DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}"
C_CHECKPOINT="${1:?C checkpoint is required}"
AFTER_CHECKPOINT="${2:?after checkpoint is required}"
CASES="${3:?cases manifest is required}"
MOTION_FILE="${4:?motion directory is required}"
STEPS="${5:-192}"
TEACHER="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MODEL="${GHOST_MODEL:-$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
OUT_ROOT="${SONIC_RUNS_ROOT}/reference_mlp_threeway_ghost4_$(date +%Y%m%d_%H%M%S)"
RENDER_CONFIG="$DAGGER_ROOT/configs/mlp_render.yaml"
for f in "$C_CHECKPOINT" "$AFTER_CHECKPOINT" "$CASES" "$TEACHER" "$MODEL" "$RENDER_CONFIG" "$SONIC_KIT_ROOT/render_policy_ghost.py"; do test -s "$f" || { echo "Missing file: $f" >&2; exit 2; }; done
test -d "$MOTION_FILE"
[[ "$STEPS" == 192 ]] || { echo "This contract requires 192 actions" >&2; exit 2; }
export PYTHONPATH="$DAGGER_ROOT/src:$PROJECT_ROOT/state-action-cvae/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$OUT_ROOT/cases"

$PYTHON - "$CASES" <<'PY'
import json,sys
if len(json.load(open(sys.argv[1])).get('cases',[])) != 4: raise SystemExit('cases.json must contain exactly four cases')
PY

run_policy() {
  local label="$1" checkpoint="$2" key="$3" root="$4"
  mkdir -p "$root"
  set +e
  SONIC_RUNS_ROOT="$root" SONIC_REPO_ROOT="$SONIC_REPO_ROOT" SONIC_TEACHER_CHECKPOINT="$TEACHER" \
    DAGGER_MOTION_FILE="$MOTION_FILE" DAGGER_RENDER_POLICY="$label" \
    DAGGER_RENDER_CHECKPOINT="$checkpoint" DAGGER_RENDER_STEPS=192 \
    DAGGER_RENDER_MOTION_KEY="$key" DAGGER_RENDER_DISABLE_MOTION_AUG=true \
    DAGGER_RENDER_KEEP_RUNNING=true DAGGER_RENDER_CONFIG="$RENDER_CONFIG" \
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >"$root/render.log" 2>&1
  local status=$?
  set -e
  local run
  run="$(find "$root" -maxdepth 1 -type d -name 'dagger_mlp_render_*' -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"
  test "$status" = 0 && test -s "$run/data/000000.trajectory.pkl" || { echo "$label rollout failed for $key; see $root/render.log" >&2; return 1; }
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
  SONIC_TRAJ="$(run_policy sonic '' "$KEY" "$CASE_OUT/sonic")"
  BEFORE_TRAJ="$(run_policy sonic_causal "$C_CHECKPOINT" "$KEY" "$CASE_OUT/before")"
  AFTER_TRAJ="$(run_policy sonic_causal "$AFTER_CHECKPOINT" "$KEY" "$CASE_OUT/after")"
  "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
    --layer "sonic=$SONIC_TRAJ" --layer "dagger_before=$BEFORE_TRAJ" --layer "dagger_after=$AFTER_TRAJ" \
    --model "$MODEL" --ground-style black --title "${SAFE}_threeway" --output-run "$CASE_OUT/ghost"
  "$PYTHON" - "$CASE_OUT/ghost/manifests/policy_ghost.json" "$CASE_OUT/case.json" "$KEY" "$C_CHECKPOINT" "$AFTER_CHECKPOINT" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if int(m.get('frames',0)) != 193: raise SystemExit(f"expected 193 frames, got {m.get('frames')}")
json.dump({'motion_key':sys.argv[3], 'c_checkpoint':sys.argv[4], 'after_checkpoint':sys.argv[5], 'ghost':m}, open(sys.argv[2],'w'), indent=2)
PY
done

"$PYTHON" - "$OUT_ROOT" "$C_CHECKPOINT" "$AFTER_CHECKPOINT" "$CASES" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cases=[]
for p in sorted((root/'cases').glob('case_*/case.json')): cases.append(json.loads(p.read_text()))
if len(cases)!=4: raise SystemExit(f'expected four cases, got {len(cases)}')
(root/'markers').mkdir(exist_ok=True)
json.dump({'format_version':'sonic_reference_mlp_threeway_ghost4_v1','c_checkpoint':str(pathlib.Path(sys.argv[2]).resolve()),'after_checkpoint':str(pathlib.Path(sys.argv[3]).resolve()),'cases_manifest':str(pathlib.Path(sys.argv[4]).resolve()),'horizon_actions':192,'horizon_states':193,'layers':['sonic','dagger_before','dagger_after'],'cases':cases}, (root/'manifest.json').open('w'), indent=2)
(root/'markers/reference_mlp_threeway_ghost4.ok').write_text('PASS\n')
print(root)
PY
