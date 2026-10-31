#!/usr/bin/env bash
# Stop any DAgger run, then resume mlp-causal-6h with a wall-clock deadline stored in the run dir.
# Usage: bash resume_mlp_causal_6h.sh [run_dir] [hours_if_no_deadline_yet]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
RUN="${1:-$RUNS/dagger_mlp_causal_6h_20261031_133715}"
HOURS="${2:-6}"
CK="$RUNS/dagger_sonic_causal_8h_20261031_032743/checkpoints/best.pt"
PY="$HOME/bly/sonic-repro/.venv-sonic/bin/python"

test -d "$RUN" || { echo "run dir missing: $RUN" >&2; exit 2; }
test -s "$CK" || { echo "motion source checkpoint missing: $CK" >&2; exit 2; }

echo "== stop"
pkill -f run_dagger_auto.sh
pkill -INT -f 'python -m dagger.cli'
for _ in $(seq 36); do pgrep -f 'python -m dagger.cli' >/dev/null || break; sleep 5; done
if pgrep -f 'python -m dagger.cli' >/dev/null; then
  echo "dagger.cli still running after 3 min; run: pkill -9 -f 'python -m dagger.cli'" >&2
  pgrep -af 'python -m dagger.cli' >&2
  exit 2
fi

DL="$RUN/deadline_epoch.txt"
[[ -s "$DL" ]] || echo $(( $(date +%s) + HOURS * 3600 )) > "$DL"

export MLP_MOTION_SOURCE_CHECKPOINT="$CK" DAGGER_RESUME_RUN="$RUN" DAGGER_DEADLINE_EPOCH="$(<"$DL")"
export DAGGER_FAULTHANDLER_SECONDS=0 DAGGER_MAX_SHARDS=300 DAGGER_SHARD_COOLDOWN=30 DAGGER_MAX_HOURS=24

echo "== check"
echo "run=$RUN"
echo "deadline=$(date -d @"$DAGGER_DEADLINE_EPOCH" '+%F %T')"
echo "motion_source=$CK"
echo "cli_has_deadline=$(grep -c DAGGER_DEADLINE_EPOCH "$SCRIPT_DIR/../src/dagger/cli.py")"
grep "rounds:" "$SCRIPT_DIR/../configs/mlp_causal_6h.yaml"
echo "completed_rounds=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$RUN/manifests/dagger_mlp_metrics.partial.json")"
echo "round_checkpoints=$(ls "$RUN"/checkpoints/round_*.pt | wc -l) round_data=$(ls "$RUN"/data/dagger/round_*.pt | wc -l)"
df -h "$RUNS" | tail -1

echo "== launch"
mkdir -p "$RUNS/logs"
nohup bash "$SCRIPT_DIR/run_dagger_auto.sh" mlp-causal-6h >> "$RUNS/logs/mlp6h.txt" 2>&1 &
echo "auto_pid=$!"
