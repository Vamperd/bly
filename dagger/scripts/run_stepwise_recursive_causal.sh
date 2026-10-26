#!/usr/bin/env bash
set -Eeuo pipefail

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$HOME/bly/state-action-cvae}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"

export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SONIC_REPO_ROOT STATE_ACTION_CVAE_ROOT SONIC_KIT_ROOT

CONFIG="${1:?usage: $0 CONFIG OUTPUT_RUN}"
OUTPUT_RUN="${2:?usage: $0 CONFIG OUTPUT_RUN}"

"$PYTHON" -m dagger.stepwise_recursive_causal \
  --config "$CONFIG" \
  --output-run "$OUTPUT_RUN"
