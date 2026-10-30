#!/usr/bin/env bash

set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in
  smoke|pilot) CONFIG_MODE="$MODE" ;;
  cvae-smoke) CONFIG_MODE="cvae_smoke" ;;
  cvae-pilot) CONFIG_MODE="cvae_pilot" ;;
  cvae-kl2-smoke) CONFIG_MODE="cvae_kl2_smoke" ;;
  cvae-kl2-pilot) CONFIG_MODE="cvae_kl2_pilot" ;;
  cvae-kl2-pilot-intensive) CONFIG_MODE="cvae_kl2_pilot_intensive" ;;
  cvae-kl2-pilot-intensive-8h) CONFIG_MODE="cvae_kl2_pilot_intensive_8h" ;;
  cvae-kl2-pilot-dagger-v2) CONFIG_MODE="cvae_kl2_pilot_dagger_v2" ;;
  cvae-kl2-pilot-dagger-v2-2m) CONFIG_MODE="cvae_kl2_pilot_dagger_v2_2m" ;;
  cvae-kl2-pilot-dagger-v2-2m-2h) CONFIG_MODE="cvae_kl2_pilot_dagger_v2_2m_2h" ;;
  cvae-kl2-formal) CONFIG_MODE="cvae_kl2_formal" ;;
  sonic-causal-smoke) CONFIG_MODE="sonic_causal_smoke" ;;
  sonic-causal-pilot) CONFIG_MODE="sonic_causal_pilot" ;;
  mlp-smoke) CONFIG_MODE="mlp_smoke" ;;
  mlp-pilot) CONFIG_MODE="mlp_pilot" ;;
  mlp-debug) CONFIG_MODE="mlp_debug" ;;
  mlp-scale) CONFIG_MODE="mlp_scale" ;;
  mlp-repro) CONFIG_MODE="mlp_repro" ;;
  mlp-long) CONFIG_MODE="mlp_long" ;;
  mlp-intensive) CONFIG_MODE="mlp_intensive" ;;
  mlp-dynamic2) CONFIG_MODE="mlp_dynamic2" ;;
  mlp-causal-motions) CONFIG_MODE="mlp_causal_motions" ;;
  mlp-render) CONFIG_MODE="mlp_render" ;;
  *)
    echo "Usage: $0 smoke|pilot|cvae-smoke|cvae-pilot|cvae-kl2-smoke|cvae-kl2-pilot|cvae-kl2-pilot-intensive|cvae-kl2-pilot-intensive-8h|cvae-kl2-pilot-dagger-v2|cvae-kl2-pilot-dagger-v2-2m|cvae-kl2-pilot-dagger-v2-2m-2h|cvae-kl2-formal|sonic-causal-smoke|sonic-causal-pilot|mlp-smoke|mlp-pilot|mlp-debug|mlp-scale|mlp-repro|mlp-long|mlp-intensive|mlp-dynamic2|mlp-causal-motions|mlp-render" >&2
  exit 2
    ;;
esac

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
STATE_ACTION_CVAE_ROOT="${STATE_ACTION_CVAE_ROOT:-$HOME/bly/state-action-cvae}"
if [[ ( "$MODE" == sonic-causal-* || "$MODE" == mlp-causal-motions || ( "$MODE" == mlp-render && "${DAGGER_RENDER_POLICY:-}" == sonic_causal ) ) && -z "${DAGGER_MOTION_FILE:-}" ]]; then
  # C training data comes from the Physics v5 reference subset; DAgger motions must be loadable from it.
  REFERENCE_SUBSET_POINTER="$SONIC_RUNS_ROOT/latest_overfit_reference_motion_subset_run_dir.txt"
  test -s "$REFERENCE_SUBSET_POINTER" || {
    echo "Set DAGGER_MOTION_FILE or create $REFERENCE_SUBSET_POINTER" >&2
    exit 2
  }
  DAGGER_MOTION_FILE="$(<"$REFERENCE_SUBSET_POINTER")/data/robot_filtered"
fi
DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered}"
DAGGER_RESUME_RUN="${DAGGER_RESUME_RUN:-}"
if [[ "$MODE" == cvae-kl2-* ]]; then
  # CVAE KL2 must not recreate an Isaac env inside one process; one round per
  # shard is enforced. Use run_dagger_auto.sh for multi-round training.
  DAGGER_ROUNDS_PER_PROCESS="${DAGGER_ROUNDS_PER_PROCESS:-1}"
else
  DAGGER_ROUNDS_PER_PROCESS="${DAGGER_ROUNDS_PER_PROCESS:-0}"
fi
DAGGER_RENDER_CHECKPOINT="${DAGGER_RENDER_CHECKPOINT:-}"
DAGGER_RENDER_STEPS="${DAGGER_RENDER_STEPS:-192}"
DAGGER_RENDER_MOTION_KEY="${DAGGER_RENDER_MOTION_KEY:-}"
DAGGER_RENDER_POLICY="${DAGGER_RENDER_POLICY:-mlp}"
DAGGER_RENDER_CONFIG="${DAGGER_RENDER_CONFIG:-}"
DAGGER_RENDER_LATENT_MODE="${DAGGER_RENDER_LATENT_MODE:-fixed_standard_normal}"

export DAGGER_ROOT SONIC_REPO_ROOT SONIC_TEACHER_CHECKPOINT SONIC_RUNS_ROOT STATE_ACTION_CVAE_ROOT DAGGER_MOTION_FILE DAGGER_RESUME_RUN DAGGER_ROUNDS_PER_PROCESS DAGGER_RENDER_CHECKPOINT DAGGER_RENDER_STEPS DAGGER_RENDER_MOTION_KEY DAGGER_RENDER_POLICY DAGGER_RENDER_CONFIG DAGGER_RENDER_LATENT_MODE
export PYTHONPATH="$DAGGER_ROOT/src:$STATE_ACTION_CVAE_ROOT/src:$SONIC_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
[[ -x "$PYTHON" ]] || {
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 2
}

test -d "$SONIC_REPO_ROOT" || {
  echo "SONIC repository is missing: $SONIC_REPO_ROOT" >&2
  exit 2
}

test -s "$SONIC_TEACHER_CHECKPOINT" || {
  echo "Teacher checkpoint is missing or empty: $SONIC_TEACHER_CHECKPOINT" >&2
  exit 2
}

CONFIG="$DAGGER_ROOT/configs/${CONFIG_MODE}.yaml"
if [[ "$MODE" == "mlp-render" && -n "$DAGGER_RENDER_CONFIG" ]]; then
  CONFIG="$DAGGER_RENDER_CONFIG"
fi
test -f "$CONFIG" || {
  echo "DAgger config is missing: $CONFIG" >&2
  exit 2
}

mkdir -p "$SONIC_RUNS_ROOT"

if pgrep -f 'python -m dagger.cli' >/dev/null; then
  echo "Another dagger.cli process is already running:" >&2
  pgrep -af 'python -m dagger.cli' >&2
  echo "Stop it before starting a new run." >&2
  exit 2
fi

LOCK_PATH="$SONIC_RUNS_ROOT/.dagger_active.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another DAgger runner holds the lock: $LOCK_PATH" >&2
  exit 2
fi
echo "[run_dagger] acquired $LOCK_PATH"

if [[ "$MODE" == mlp-smoke || "$MODE" == mlp-pilot || "$MODE" == mlp-debug || "$MODE" == mlp-scale || "$MODE" == mlp-repro || "$MODE" == mlp-long || "$MODE" == mlp-intensive || "$MODE" == mlp-dynamic2 || "$MODE" == mlp-causal-motions || "$MODE" == mlp-render ]]; then
  test -d "$DAGGER_MOTION_FILE" || {
    echo "DAGGER_MOTION_FILE is missing or not a directory: $DAGGER_MOTION_FILE" >&2
    exit 2
  }
fi

if [[ "$MODE" == mlp-render ]]; then
  if [[ "$DAGGER_RENDER_POLICY" == "sonic" ]]; then
    test -s "$SONIC_TEACHER_CHECKPOINT" || {
      echo "SONIC teacher checkpoint is missing or empty: $SONIC_TEACHER_CHECKPOINT" >&2
      exit 2
    }
  else
    : "${DAGGER_RENDER_CHECKPOINT:?Set DAGGER_RENDER_CHECKPOINT to an MLP or sonic_causal checkpoint}"
    test -s "$DAGGER_RENDER_CHECKPOINT" || {
      echo "Render checkpoint is missing or empty: $DAGGER_RENDER_CHECKPOINT" >&2
      exit 2
    }
  fi
fi

if [[ "$MODE" == cvae-smoke || "$MODE" == cvae-pilot ]]; then
  : "${CVAE_STUDENT_CHECKPOINT:?Set CVAE_STUDENT_CHECKPOINT to a Stage-C CVAE checkpoint}"
  : "${CVAE_BASE_DATASET_RUN:?Set CVAE_BASE_DATASET_RUN to a validated CVAE dataset run}"
  test -s "$CVAE_STUDENT_CHECKPOINT" || {
    echo "CVAE student checkpoint is missing or empty: $CVAE_STUDENT_CHECKPOINT" >&2
    exit 2
  }
  test -d "$CVAE_BASE_DATASET_RUN" || {
    echo "CVAE base dataset run is missing: $CVAE_BASE_DATASET_RUN" >&2
    exit 2
  }
fi

if [[ "$MODE" == cvae-kl2-smoke || "$MODE" == cvae-kl2-pilot || "$MODE" == cvae-kl2-pilot-intensive || "$MODE" == cvae-kl2-pilot-intensive-8h || "$MODE" == cvae-kl2-pilot-dagger-v2 || "$MODE" == cvae-kl2-pilot-dagger-v2-2m || "$MODE" == cvae-kl2-pilot-dagger-v2-2m-2h || "$MODE" == cvae-kl2-formal ]]; then
  export CVAE_KL2_MOTION_FILE="${CVAE_KL2_MOTION_FILE:-/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered}"
  test -d "$CVAE_KL2_MOTION_FILE" || {
    echo "CVAE KL2 motion directory is missing: $CVAE_KL2_MOTION_FILE" >&2
    exit 2
  }
  if [[ "$MODE" == cvae-kl2-pilot-dagger-v2-2m* ]]; then
    export DAGGER_MOTION_MANIFEST="${DAGGER_MOTION_MANIFEST:-$DAGGER_ROOT/configs/cvae_kl2_motion_manifest_2m.json}"
  else
    export DAGGER_MOTION_MANIFEST="${DAGGER_MOTION_MANIFEST:-$DAGGER_ROOT/configs/cvae_kl2_motion_manifest.json}"
  fi
  test -s "$DAGGER_MOTION_MANIFEST" || {
    echo "CVAE KL2 motion manifest is missing: $DAGGER_MOTION_MANIFEST" >&2
    exit 2
  }
  if [[ -n "${CVAE_STUDENT_CHECKPOINT:-}" ]]; then
    test -s "$CVAE_STUDENT_CHECKPOINT" || {
      echo "CVAE student checkpoint is missing or empty: $CVAE_STUDENT_CHECKPOINT" >&2
      exit 2
    }
  fi
  if [[ -n "${CVAE_KL2_DATASET_RUN:-}" ]]; then
    test -d "$CVAE_KL2_DATASET_RUN" || {
      echo "CVAE KL2 dataset run is missing: $CVAE_KL2_DATASET_RUN" >&2
      exit 2
    }
  fi
fi

if [[ "$MODE" == mlp-causal-motions ]]; then
  : "${MLP_MOTION_SOURCE_CHECKPOINT:=${SONIC_CAUSAL_STUDENT_CHECKPOINT:-}}"
  export MLP_MOTION_SOURCE_CHECKPOINT
  test -s "$MLP_MOTION_SOURCE_CHECKPOINT" || {
    echo "Set MLP_MOTION_SOURCE_CHECKPOINT (or SONIC_CAUSAL_STUDENT_CHECKPOINT) to the C checkpoint used by sonic-causal-pilot" >&2
    exit 2
  }
  echo "[run_dagger] MLP_MOTION_SOURCE_CHECKPOINT=$MLP_MOTION_SOURCE_CHECKPOINT"
fi

if [[ "$MODE" == sonic-causal-smoke || "$MODE" == sonic-causal-pilot ]]; then
  : "${SONIC_CAUSAL_STUDENT_CHECKPOINT:=${CVAE_STUDENT_CHECKPOINT:-}}"
  export SONIC_CAUSAL_STUDENT_CHECKPOINT
  test -s "$SONIC_CAUSAL_STUDENT_CHECKPOINT" || {
    echo "SONIC_CAUSAL_STUDENT_CHECKPOINT is missing or empty" >&2
    exit 2
  }
  test -d "$DAGGER_MOTION_FILE" || {
    echo "DAGGER_MOTION_FILE is missing or not a directory: $DAGGER_MOTION_FILE" >&2
    exit 2
  }
fi

LOG_PATH="$SONIC_RUNS_ROOT/dagger_${MODE}_command_$(date +%Y%m%d_%H%M%S).log"
echo "[run_dagger] CONFIG=$CONFIG" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_MOTION_FILE=$DAGGER_MOTION_FILE" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RESUME_RUN=$DAGGER_RESUME_RUN" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_ROUNDS_PER_PROCESS=$DAGGER_ROUNDS_PER_PROCESS" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_CHECKPOINT=$DAGGER_RENDER_CHECKPOINT" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_STEPS=$DAGGER_RENDER_STEPS" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_MOTION_KEY=$DAGGER_RENDER_MOTION_KEY" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_POLICY=$DAGGER_RENDER_POLICY" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_CONFIG=$DAGGER_RENDER_CONFIG" >> "$LOG_PATH"
echo "[run_dagger] DAGGER_RENDER_LATENT_MODE=$DAGGER_RENDER_LATENT_MODE" >> "$LOG_PATH"
echo "[run_dagger] SONIC_TEACHER_CHECKPOINT=$SONIC_TEACHER_CHECKPOINT" >> "$LOG_PATH"
set +e
if [[ "${DAGGER_TEE:-0}" == "1" ]]; then
  "$PYTHON" -m dagger.cli --config "$CONFIG" 2>&1 | tee -a "$LOG_PATH"
  STATUS=${PIPESTATUS[0]}
else
  "$PYTHON" -m dagger.cli --config "$CONFIG" >> "$LOG_PATH" 2>&1
  STATUS=$?
fi
set -e
echo "[run_dagger] dagger_exit_code=$STATUS" >> "$LOG_PATH"
echo "[run_dagger] log=$LOG_PATH"
echo "[run_dagger] dagger_exit_code=$STATUS"
exit "$STATUS"
