#!/usr/bin/env bash

set -u

MODE="${1:-mlp-pilot}"
case "$MODE" in
  mlp-pilot|mlp-long|mlp-intensive|mlp-dynamic2|cvae-kl2-pilot|cvae-kl2-pilot-intensive|cvae-kl2-pilot-intensive-8h|cvae-kl2-pilot-dagger-v2|cvae-kl2-pilot-dagger-v2-2m|cvae-kl2-pilot-dagger-v2-2m-2h|cvae-kl2-formal) ;;
  *)
    echo "Usage: $0 mlp-pilot|mlp-long|mlp-intensive|mlp-dynamic2|cvae-kl2-pilot|cvae-kl2-pilot-intensive|cvae-kl2-pilot-intensive-8h|cvae-kl2-pilot-dagger-v2|cvae-kl2-pilot-dagger-v2-2m|cvae-kl2-pilot-dagger-v2-2m-2h|cvae-kl2-formal" >&2
    exit 2
    ;;
esac

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
if [[ -n "${DAGGER_SHARD_ROUNDS:-}" ]]; then
  SHARD_ROUNDS="$DAGGER_SHARD_ROUNDS"
elif [[ "$MODE" == cvae-kl2-* ]]; then
  SHARD_ROUNDS=1
else
  SHARD_ROUNDS=2
fi
if [[ -n "${DAGGER_MAX_RETRIES:-}" ]]; then
  MAX_RETRIES="$DAGGER_MAX_RETRIES"
elif [[ "$MODE" == cvae-kl2-* ]]; then
  MAX_RETRIES=5
else
  MAX_RETRIES=3
fi
MAX_SHARDS="${DAGGER_MAX_SHARDS:-20}"
if [[ -n "${DAGGER_SHARD_TIMEOUT:-}" ]]; then
  SHARD_TIMEOUT="$DAGGER_SHARD_TIMEOUT"
elif [[ "$MODE" == cvae-kl2-* ]]; then
  SHARD_TIMEOUT="${DAGGER_CVAE_SHARD_TIMEOUT:-2400}"
else
  SHARD_TIMEOUT=0
fi
RUN_PREFIX="dagger_${MODE//-/_}"
RUN="${DAGGER_RESUME_RUN:-}"

test -x "$PYTHON" || {
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 2
}
mkdir -p "$SONIC_RUNS_ROOT"

ATTEMPTS=0
PREV=-1

for ((shard = 1; shard <= MAX_SHARDS; shard++)); do
  if [[ -n "$RUN" ]]; then
    export DAGGER_RESUME_RUN="$RUN"
    echo "[auto] shard=$shard attempt=$ATTEMPTS resume=$RUN"
  else
    unset DAGGER_RESUME_RUN
    echo "[auto] shard=$shard attempt=$ATTEMPTS new_run"
  fi
  export DAGGER_ROUNDS_PER_PROCESS="$SHARD_ROUNDS"

  if [[ "$SHARD_TIMEOUT" -gt 0 ]]; then
    timeout --signal=INT --kill-after=60 "$SHARD_TIMEOUT" \
      bash "$DAGGER_ROOT/scripts/run_dagger.sh" "$MODE"
    STATUS=$?
  else
    bash "$DAGGER_ROOT/scripts/run_dagger.sh" "$MODE"
    STATUS=$?
  fi

  if [[ -z "$RUN" ]]; then
    RUN="$(ls -dt "$SONIC_RUNS_ROOT"/${RUN_PREFIX}_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$RUN" ]]; then
    echo "[auto] no run directory was created" >&2
    exit 1
  fi

  if [[ "$MODE" == cvae-kl2-* ]]; then
    METRICS="$RUN/manifests/dagger_cvae_kl2_metrics.partial.json"
    FINAL_MANIFEST="$RUN/manifests/dagger_cvae_kl2_manifest.json"
  else
    METRICS="$RUN/manifests/dagger_mlp_metrics.partial.json"
    FINAL_MANIFEST="$RUN/manifests/dagger_mlp_manifest.json"
  fi
  if [[ -f "$RUN/markers/dagger_cvae_kl2_early_stop.ok" ]]; then
    echo "[auto] early stop run=$RUN"
    cat "$RUN/manifests/dagger_cvae_kl2_early_stop.json" 2>/dev/null || true
    exit 0
  fi
  COMPLETED=0
  if [[ -f "$METRICS" ]]; then
    COMPLETED="$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$METRICS" 2>/dev/null || echo 0)"
  fi

  if [[ -f "$FINAL_MANIFEST" ]]; then
    COMPLETE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("complete", False))' "$FINAL_MANIFEST" 2>/dev/null || echo False)"
    if [[ "$COMPLETE" == "True" ]]; then
      echo "[auto] complete run=$RUN"
      if [[ "$MODE" == cvae-kl2-* ]]; then
        "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("quality_pass=", d.get("quality_pass")); print("rounds=", len(d.get("rounds", [])))' "$FINAL_MANIFEST"
      else
        "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("memorization_pass=", d.get("memorization_pass")); print("rounds=", len(d.get("rounds", [])))' "$FINAL_MANIFEST"
      fi
      exit 0
    fi
  fi

  echo "[auto] status=$STATUS completed_rounds=$COMPLETED run=$RUN"

  if [[ "$STATUS" -eq 2 ]]; then
    echo "[auto] stopping: configuration/lock/process conflict (status=2)" >&2
    exit 2
  fi

  # Give the previous Kit/Isaac process time to release the GPU and files
  # before the next shard is launched.
  sleep "${DAGGER_SHARD_COOLDOWN:-5}"

  if [[ "${DAGGER_STOP_ON_CRASH:-0}" == "1" && "$STATUS" -ne 0 ]]; then
    echo "[auto] stopping: shard exited with status=$STATUS and DAGGER_STOP_ON_CRASH=1" >&2
    exit 1
  fi

  if [[ "$COMPLETED" -gt "$PREV" ]]; then
    PREV="$COMPLETED"
    ATTEMPTS=0
  else
    ATTEMPTS=$((ATTEMPTS + 1))
  fi

  if [[ "$ATTEMPTS" -ge "$MAX_RETRIES" ]]; then
    echo "[auto] stopping: no progress after $MAX_RETRIES attempts" >&2
    exit 1
  fi

  export DAGGER_RESUME_RUN="$RUN"
done

echo "[auto] stopping: reached DAGGER_MAX_SHARDS=$MAX_SHARDS" >&2
exit 1
