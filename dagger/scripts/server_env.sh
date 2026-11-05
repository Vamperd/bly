#!/usr/bin/env bash
# Source after activating the server's existing Isaac/SONIC Python environment.
# Set BLY_MIGRATED_RUNS explicitly if more than one migration directory exists.

_bly_server_env() {
  local project migration
  project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return
  export BLY_PROJECT_ROOT="${BLY_PROJECT_ROOT:-$project}"
  export SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$BLY_PROJECT_ROOT/runs}"
  export SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$BLY_PROJECT_ROOT/sonic-repro/GR00T-WholeBodyControl}"
  export STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$project/state-action-cvae}"
  export SONIC_KIT_ROOT="${SONIC_KIT_ROOT:-$project/sonic-repro-kit}"
  export SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
  if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$BLY_PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python" ]]; then
      export PYTHON="$BLY_PROJECT_ROOT/sonic-repro/.venv-sonic/bin/python"
    else
      export PYTHON="$(command -v python)"
    fi
  fi
  [[ -x "$PYTHON" ]] || { echo "Activate the Isaac/SONIC environment or set PYTHON to its interpreter." >&2; return 2; }
  export PYTHONPATH="$project/dagger/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  migration="${BLY_MIGRATED_RUNS:-}"
  if [[ -z "$migration" ]]; then
    local -a candidates=()
    local item
    for item in "$BLY_PROJECT_ROOT"/runs/source_migration_*; do
      [[ -d "$item/cvae_physics_dataset_v5" ]] && candidates+=("$item")
    done
    [[ ${#candidates[@]} -eq 1 ]] || {
      echo "Set BLY_MIGRATED_RUNS to the completed source_migration directory (found ${#candidates[@]} candidates)." >&2
      return 2
    }
    migration="${candidates[0]}"
  fi
  migration="$(cd "$migration" && pwd)" || return
  export BLY_MIGRATED_RUNS="$migration"
  if [[ -z "${CVAE_DATASET_PATH_MAP:-}" ]]; then
    CVAE_DATASET_PATH_MAP="$("$PYTHON" -c 'import json,sys; print(json.dumps({"/home/helloworld/bly/runs":sys.argv[1]}))' "$migration")" || return
    export CVAE_DATASET_PATH_MAP
  fi
  export CVAE_STUDENT_CHECKPOINT="${CVAE_STUDENT_CHECKPOINT:-$SONIC_RUNS_ROOT/cvae_reference_train_20261103_095438/checkpoints/best.pt}"
  export CVAE_KL2_DATASET_RUN="${CVAE_KL2_DATASET_RUN:-$migration/cvae_reference_mixed_dataset_20261103_090945}"
  export CVAE_KL2_MOTION_FILE="${CVAE_KL2_MOTION_FILE:-$migration/prepare_overfit_reference_subset_20261030_174517/data/robot_filtered}"
  export DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-$CVAE_KL2_MOTION_FILE}"
  export DAGGER_MOTION_MANIFEST="${DAGGER_MOTION_MANIFEST:-$project/dagger/configs/cvae_kl2_motion_manifest_v5.json}"
  export CVAE_MODEL_CONFIG="${CVAE_MODEL_CONFIG:-$STATE_ACTION_CVAE_ROOT/configs/posterior_hierarchical_standard_cvae_v5_reference_fp32.json}"
  export DAGGER_DEVICE="${DAGGER_DEVICE:-cuda:0}"
  export DAGGER_CVAE_SHARD_TIMEOUT="${DAGGER_CVAE_SHARD_TIMEOUT:-3600}"
  export DAGGER_MAX_SHARDS="${DAGGER_MAX_SHARDS:-40}"
  printf 'project=%s\npython=%s\nmigrated_runs=%s\nstudent=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$BLY_PROJECT_ROOT" "$PYTHON" "$migration" "$CVAE_STUDENT_CHECKPOINT" "${CUDA_VISIBLE_DEVICES:-not set}"
}
_bly_server_env
