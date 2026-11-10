#!/usr/bin/env bash
# Invoke once OUTSIDE a GPU allocation: submits one Isaac shard at a time and
# blocks until 12h of committed DAgger work is saved (queue/startup time excluded).
# Usage: bash run_mlp_cvae11_128_dagger.sh C_RUN MIG OUTPUT_STATE_DIR
# Restart with the same arguments to reattach/resume; dagger_stage.json is atomic.
# DAGGER_STAGE_SMOKE=1 retains all 128 motions/envs, but defaults to two tiny
# one-round processes. Smoke-only overrides: DAGGER_STAGE_ROUNDS/HOURS,
# DAGGER_STAGE_ROLLOUT_STEPS/TRAIN_STEPS/EVAL_STEPS. Resource overrides:
# DAGGER_SHARD_ROUNDS (2), DAGGER_STAGE_LEASE_SECONDS (7200),
# DAGGER_STAGE_CPUS (16), DAGGER_STAGE_RAM (48G), DAGGER_STAGE_POLL_SECONDS (15).
set -Eeuo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$CODE/sonic-repro/.venv-sonic/bin/python}"
exec "$PYTHON" "$CODE/dagger/scripts/run_mlp_cvae11_128_dagger.py" \
  "${1:?C_RUN required}" "${2:?MIG required}" "${3:?OUTPUT_STATE_DIR required}"
