#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   render_mlp_action11_ghost4.sh MLP_CHECKPOINT CASES_JSON ENV_CONFIG [STEPS=192]
#
# CASES_JSON must contain four exact-init/source-replay cases.  The MLP lane
# uses one Action per model call with real simulator feedback.  The SONIC lane
# is the unchanged release teacher on the same motion key.  Both trajectories
# are rendered into a two-layer 193-frame ghost when available.

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$PROJECT_ROOT/state-action-cvae}"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
MLP_CHECKPOINT="${1:?MLP checkpoint is required}"
CASES_MANIFEST="${2:?cases manifest is required}"
ENV_CONFIG="${3:?Isaac environment config is required}"
STEPS="${4:-192}"
MOTION_FILE="${DAGGER_MOTION_FILE:?DAGGER_MOTION_FILE must point to the motion library}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
MODEL="${GHOST_MODEL:-$SONIC_REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
OUT_ROOT="${SONIC_RUNS_ROOT:-$PROJECT_ROOT/runs}/mlp_action11_ghost4_$(date +%Y%m%d_%H%M%S)"

[[ -s "$MLP_CHECKPOINT" ]] || { echo "Missing MLP checkpoint: $MLP_CHECKPOINT" >&2; exit 2; }
[[ -s "$CASES_MANIFEST" ]] || { echo "Missing cases manifest: $CASES_MANIFEST" >&2; exit 2; }
[[ -s "$SONIC_TEACHER_CHECKPOINT" ]] || { echo "Missing SONIC teacher: $SONIC_TEACHER_CHECKPOINT" >&2; exit 2; }
[[ -s "$MODEL" ]] || { echo "Missing MuJoCo model: $MODEL" >&2; exit 2; }
[[ -f "$SONIC_KIT_ROOT/render_policy_ghost.py" ]] || { echo "Missing ghost renderer" >&2; exit 2; }
[[ "$STEPS" == 192 ]] || { echo "This contract requires exactly 192 transitions" >&2; exit 2; }

export CVAE_KL2_MOTION_FILE="$MOTION_FILE"
export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$OUT_ROOT/tmp"
mkdir -p "$TMPDIR"
mkdir -p "$OUT_ROOT/cases"

CASE_COUNT="$($PYTHON - "$CASES_MANIFEST" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
cases = value.get("cases")
if not isinstance(cases, list) or len(cases) != 4:
    raise SystemExit("cases manifest must contain exactly four cases")
print(len(cases))
PY
)"
[[ "$CASE_COUNT" == 4 ]] || exit 2

for INDEX in 0 1 2 3; do
  MOTION_KEY="$($PYTHON - "$CASES_MANIFEST" "$INDEX" <<'PY'
import json, sys
cases = json.load(open(sys.argv[1], encoding="utf-8"))["cases"]
print(cases[int(sys.argv[2])]["motion_key"])
PY
)"
  CASE_OUT="$OUT_ROOT/cases/case_${INDEX}_${MOTION_KEY}"
  mkdir -p "$CASE_OUT"
  echo "[mlp-action11-ghost4] case=$INDEX motion=$MOTION_KEY"

  "$PYTHON" -m dagger.mlp_action11_recorded_state_replay \
    --cases-manifest "$CASES_MANIFEST" \
    --case-index "$INDEX" \
    --config "$ENV_CONFIG" \
    --checkpoint "$MLP_CHECKPOINT" \
    --output-run "$CASE_OUT/mlp" \
    --device "${DAGGER_MLP_DEVICE:-cuda:0}"
  MLP_TRAJ="$CASE_OUT/mlp/data/mlp_action11.trajectory.pkl"
  [[ -s "$MLP_TRAJ" ]] || { echo "MLP trajectory missing: $MLP_TRAJ" >&2; exit 1; }

  SONIC_RUN_ROOT="$CASE_OUT/sonic_runs"
  mkdir -p "$SONIC_RUN_ROOT"
  set +e
  SONIC_RUNS_ROOT="$SONIC_RUN_ROOT" \
  DAGGER_MOTION_FILE="$MOTION_FILE" \
  SONIC_REPO_ROOT="$SONIC_REPO_ROOT" \
  SONIC_TEACHER_CHECKPOINT="$SONIC_TEACHER_CHECKPOINT" \
  DAGGER_RENDER_POLICY=sonic \
  DAGGER_RENDER_CHECKPOINT= \
  DAGGER_RENDER_STEPS="$STEPS" \
  DAGGER_RENDER_MOTION_KEY="$MOTION_KEY" \
  DAGGER_RENDER_DISABLE_MOTION_AUG=true \
  DAGGER_RENDER_KEEP_RUNNING=true \
  bash "$DAGGER_ROOT/scripts/run_dagger.sh" mlp-render >"$CASE_OUT/sonic.log" 2>&1
  SONIC_STATUS=$?
  set -e
  SONIC_RUN="$(find "$SONIC_RUN_ROOT" -maxdepth 1 -type d -name 'dagger_mlp_render_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"
  SONIC_TRAJ="$SONIC_RUN/data/000000.trajectory.pkl"
  if [[ "$SONIC_STATUS" != 0 || ! -s "$SONIC_TRAJ" ]]; then
    echo "SONIC rollout failed; preserving MLP output: status=$SONIC_STATUS run=${SONIC_RUN:-none}" >&2
    exit 1
  fi

  GHOST_OUT="$CASE_OUT/ghost"
  "$PYTHON" "$SONIC_KIT_ROOT/render_policy_ghost.py" \
    --layer "mlp_action11=$MLP_TRAJ" \
    --layer "sonic=$SONIC_TRAJ" \
    --model "$MODEL" \
    --ground-style black \
    --title "${MOTION_KEY}_mlp_action11_vs_sonic" \
    --output-run "$GHOST_OUT"
  "$PYTHON" - "$GHOST_OUT/manifests/policy_ghost.json" "$CASE_OUT/case.json" "$MOTION_KEY" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if int(manifest.get("frames", 0)) != 193:
    raise SystemExit(f"ghost renderer did not produce 193 frames: {manifest.get('frames')}")
json.dump({"motion_key": sys.argv[3], "ghost": manifest}, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
PY
done

"$PYTHON" - "$OUT_ROOT" "$MLP_CHECKPOINT" "$CASES_MANIFEST" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cases = []
for path in sorted((root / "cases").glob("case_*/case.json")):
    cases.append(json.load(path.open(encoding="utf-8")))
if len(cases) != 4:
    raise SystemExit(f"expected four completed cases, found {len(cases)}")
json.dump({
    "format_version": "sonic_mlp_action11_ghost4_v1",
    "mlp_checkpoint": str(pathlib.Path(sys.argv[2]).resolve()),
    "cases_manifest": str(pathlib.Path(sys.argv[3]).resolve()),
    "horizon_actions": 192,
    "horizon_states": 193,
    "train_only_model": True,
    "cases": cases,
}, (root / "manifests.json").open("w", encoding="utf-8"), indent=2)
(root / "markers" ).mkdir(exist_ok=True)
(root / "markers/mlp_action11_ghost4.ok").write_text("PASS\n", encoding="utf-8")
print(root)
PY

echo "$OUT_ROOT"
