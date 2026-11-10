#!/usr/bin/env bash
set -Eeuo pipefail

C_JOB_REF="${1:?C job ref required}"
C_RUN="${2:?C run required}"
MIG="${3:?migration root required}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D_CFG="$CODE/dagger/configs/sonic_causal_mlp_cvae11_128_12h.yaml"

while :; do
  state="$(gpu ps --json 2>/dev/null | python3 -c '
import json, sys
payload = json.load(sys.stdin)
job = next((item for item in payload["jobs"] if item["job_ref"] == sys.argv[1]), None)
print("MISSING" if job is None else job["state"])
' "$C_JOB_REF" || true)"
  printf '%s C_JOB=%s\n' "$(date -Is)" "$state"
  case "$state" in
    SUCCEEDED)
      test -s "$C_RUN/checkpoints/best.pt"
      test -s "$C_RUN/manifests/mlp_cvae11_summary.json"
      break
      ;;
    FAILED|CANCELLED|PREEMPTED|MISSING)
      printf 'C job did not complete successfully: %s\n' "$state" >&2
      exit 1
      ;;
  esac
  sleep 60
done

MAP="$(python3 -c 'import json,sys; print(json.dumps({"/home/helloworld/bly/runs":sys.argv[1]}))' "$MIG")"
SONIC_REPO_ROOT="$CODE/sonic-repro/GR00T-WholeBodyControl"
TMP_ROOT="$CODE/../isaac_tmp"
export BLY_PROJECT_ROOT="$CODE"
export SONIC_RUNS_ROOT="$CODE/../runs"
export SONIC_REPO_ROOT
export SONIC_KIT_ROOT="$CODE/sonic-repro-kit"
export SONIC_TEACHER_CHECKPOINT="$SONIC_REPO_ROOT/sonic_release/last.pt"
export DAGGER_MOTION_FILE="$MIG/bones_seed_ingest_2048_20260825_132447/data/robot_filtered"
export MLP_CVAE11_CHECKPOINT="$C_RUN/checkpoints/best.pt"
export CVAE_DATASET_RUN="$MIG/cvae_physics_dataset_20260825_235244"
export CVAE_DATASET_PATH_MAP="$MAP"
export TMPDIR="$TMP_ROOT"
export XDG_CACHE_HOME="$TMP_ROOT"
export PYTHONPATH="$CODE/dagger/src:$CODE/state-action-cvae/src:$CODE/sonic-repro-kit:$SONIC_REPO_ROOT"
mkdir -p "$TMP_ROOT"

exec gpu 1 -c16 -m48G -t12h -n mlp-cvae11-128-dagger-12h env \
  BLY_PROJECT_ROOT="$BLY_PROJECT_ROOT" SONIC_RUNS_ROOT="$SONIC_RUNS_ROOT" \
  SONIC_REPO_ROOT="$SONIC_REPO_ROOT" SONIC_KIT_ROOT="$SONIC_KIT_ROOT" \
  SONIC_TEACHER_CHECKPOINT="$SONIC_TEACHER_CHECKPOINT" DAGGER_MOTION_FILE="$DAGGER_MOTION_FILE" \
  MLP_CVAE11_CHECKPOINT="$MLP_CVAE11_CHECKPOINT" CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
  CVAE_DATASET_PATH_MAP="$CVAE_DATASET_PATH_MAP" TMPDIR="$TMPDIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  PYTHONPATH="$PYTHONPATH" "$CODE/sonic-repro/.venv-sonic/bin/python" -m dagger.cli --config "$D_CFG"
