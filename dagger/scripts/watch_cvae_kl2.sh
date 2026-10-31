#!/usr/bin/env bash
# Usage: watch -n 60 bash watch_cvae_kl2.sh [mode=cvae-kl2-pilot-future-ref-chunk-12h] [run_dir]
# Prints the live round/phase/loss from the newest shard log and the per-round eval table.

MODE="${1:-cvae-kl2-pilot-future-ref-chunk-12h}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
RUN="${2:-$(ls -dt "$SONIC_RUNS_ROOT"/dagger_${MODE//-/_}_* 2>/dev/null | head -n 1)}"
LOG="$(ls -t "$SONIC_RUNS_ROOT"/logs/dagger_${MODE}_command_*.log 2>/dev/null | head -n 1)"

echo "run=${RUN:-<none>}"
echo "log=${LOG:-<none>}"
if pgrep -f 'python -m dagger.cli' >/dev/null; then echo "process=running"; else echo "process=NOT running"; fi
if [[ -n "$LOG" ]]; then
  grep -E "cvae shard start_round" "$LOG" | tail -n 1
  grep -E "cvae (collect|train|eval)" "$LOG" | tail -n 1
  grep -E "Traceback|Error|SIGSEGV|exit_code" "$LOG" | tail -n 3
fi
[[ -f "$RUN/markers/dagger_cvae_kl2_early_stop.ok" ]] && echo "EARLY STOP: $RUN/manifests/dagger_cvae_kl2_early_stop.json"
[[ -f "$RUN/manifests/dagger_cvae_kl2_manifest.json" ]] && echo "FINAL manifest exists"
METRICS="$RUN/manifests/dagger_cvae_kl2_metrics.partial.json"
[[ -f "$METRICS" ]] || { echo "no completed round yet"; exit 0; }
"$PYTHON" - "$METRICS" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print(f"{'round':>5} {'beta':>5} {'min':>6} {'loss':>9} {'rmse':>8} {'surv':>5} {'term':>7}")
total = 0.0
for m in rows:
    s, t = m.get("student_only") or {}, m.get("train") or {}
    total += float(m.get("round_seconds", 0.0))
    print(f"{m['round']:>5} {m.get('beta', 0):>5.2f} {m.get('round_seconds', 0) / 60:>6.1f} "
          f"{t.get('train_loss', float('nan')):>9.4g} {s.get('teacher_action_rmse', float('nan')):>8.4f} "
          f"{s.get('survival_ratio', float('nan')):>5.2f} {s.get('termination_rate', float('nan')):>7.4f}")
print(f"completed={len(rows)} elapsed={total / 3600:.2f}h")
PY
