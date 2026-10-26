#!/usr/bin/env bash
set -Eeuo pipefail

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$HOME/bly/state-action-cvae}"
SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$HOME/bly/sonic-repro-kit}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"

export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT:$SONIC_KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${CVAE_ORIGINAL_CHECKPOINT:?Set CVAE_ORIGINAL_CHECKPOINT}"
: "${CVAE_KL2_DATASET_RUN:?Set CVAE_KL2_DATASET_RUN}"
OUTPUT_RUN="$1"
HORIZON_STEPS="${STEPWISE_HORIZON_STEPS:-192}"
MASK_SEED="${STEPWISE_MASK_SEED:-20261016}"
SAMPLE_SEED="${STEPWISE_SAMPLE_SEED:-20261023}"

shift
if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 OUTPUT_RUN MOTION_KEY [MOTION_KEY ...]" >&2
  exit 2
fi

motion_args=()
for key in "$@"; do
  motion_args+=(--motion-key "$key")
done

"$PYTHON" -m dagger.stepwise_prepare \
  --dataset-run "$CVAE_KL2_DATASET_RUN" \
  --checkpoint "$CVAE_ORIGINAL_CHECKPOINT" \
  --output-run "$OUTPUT_RUN" \
  --cvae-root "$STATE_ACTION_CVAE_ROOT" \
  --horizon-steps "$HORIZON_STEPS" \
  --mask-seed "$MASK_SEED" \
  --sample-seed "$SAMPLE_SEED" \
  "${motion_args[@]}"
