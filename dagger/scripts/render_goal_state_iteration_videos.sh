#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: render_goal_state_iteration_videos.sh FIXED_POINT_RUN [MODEL]
DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$DAGGER_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python}"
KIT_ROOT="${SONIC_KIT_ROOT:-$PROJECT_ROOT/sonic-repro-kit}"
RUN="${1:?fixed-point run is required}"
MODEL="${2:-$PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml}"
STEPS="${STEPS:-192}"
OUTPUT="$RUN/videos_goal_state_iterations"
MANIFEST="$RUN/manifests/goal_state_fixed_point.json"
[[ "$STEPS" == 192 ]] || { echo "This contract requires 192 transitions" >&2; exit 2; }
test -s "$MANIFEST" && test -s "$MODEL"
mkdir -p "$OUTPUT/iterations" "$OUTPUT/final_ghost"

mapfile -t KEYS < <("$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); print('\n'.join(m['motion_keys_loaded']))
if int(m.get('max_iterations',0)) < 21: raise SystemExit('fixed-point run must contain iteration 0..20')
PY
)
[[ ${#KEYS[@]} -eq 8 ]] || { echo "expected eight loaded motion keys, found ${#KEYS[@]}" >&2; exit 2; }

for iteration in $(seq -w 0 20); do
  iter_dir="$RUN/data/iterations/iteration_$iteration"
  test -d "$iter_dir" || { echo "missing $iter_dir" >&2; exit 2; }
  for index in $(seq 0 7); do
    key="${KEYS[$index]}"
    safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
    trajectory="$iter_dir/$(printf '%02d' "$index")_${safe}.trajectory.pkl"
    test -s "$trajectory" || { echo "missing trajectory $trajectory" >&2; exit 2; }
    out="$OUTPUT/iterations/iteration_$iteration/${safe}.mp4"
    mkdir -p "$(dirname "$out")"
    MUJOCO_GL=egl "$PYTHON" "$KIT_ROOT/render_mujoco_trajectory.py" \
      --trajectory "$trajectory" --model "$MODEL" --output "$out" \
      --manifest-dir "$(dirname "$out")/manifests" --width 960 --height 540 \
      --gl egl --ground-style black
  done
done

for index in $(seq 0 7); do
  key="${KEYS[$index]}"
  safe="$(printf '%s' "$key" | tr -c 'A-Za-z0-9_.-' '_')"
  base="$OUTPUT/final_ghost/$safe"
  mkdir -p "$base"
  layers=(
    "goal=$(find "$RUN/data/iterations/iteration_00" -name "$(printf '%02d' "$index")_${safe}.trajectory.pkl" -print -quit)"
    "state_goal_10=$(find "$RUN/data/iterations/iteration_10" -name "$(printf '%02d' "$index")_${safe}.trajectory.pkl" -print -quit)"
    "state_goal_20=$(find "$RUN/data/iterations/iteration_20" -name "$(printf '%02d' "$index")_${safe}.trajectory.pkl" -print -quit)"
  )
  "$PYTHON" "$KIT_ROOT/render_policy_ghost.py" \
    --layer "${layers[0]}" --layer "${layers[1]}" --layer "${layers[2]}" \
    --model "$MODEL" --ground-style black --title "${safe}_goal_state_iterations" \
    --output-run "$base/ghost"
done

"$PYTHON" - "$RUN" "$OUTPUT" <<'PY'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); fixed=json.loads((run/'manifests/goal_state_fixed_point.json').read_text())
checks=[]
for p in sorted((out/'final_ghost').glob('*/ghost/manifests/policy_ghost.json')):
 m=json.loads(p.read_text()); checks.append({'motion_key':p.parent.parent.name,'frames':m.get('frames'),'video':m.get('video'),'layers':[x['name'] for x in m.get('layers',[])]})
if len(checks)!=8 or any(x['frames']!=193 for x in checks): raise SystemExit(f'final ghost validation failed: {checks}')
json.dump({'format_version':'sonic_goal_state_iteration_videos_v1','source_run':str(run.resolve()),'motion_keys':fixed['motion_keys_loaded'],'iterations_saved':21,'horizon_actions':192,'horizon_states':193,'final_ghost_layers':['goal','state_goal_10','state_goal_20'],'final_ghosts':checks},(out/'manifest.json').open('w'),indent=2)
(out/'markers').mkdir(exist_ok=True); (out/'markers/sonic_goal_state_iteration_videos.ok').write_text('PASS\n')
print(out)
PY
