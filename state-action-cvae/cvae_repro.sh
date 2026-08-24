#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${CVAE_RUNS_ROOT:-$HOME/bly/runs}"
PYTHON="${CVAE_PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
DEFAULT_CONFIG="$SCRIPT_DIR/configs/default.json"
SMOKE_CONFIG="$SCRIPT_DIR/configs/smoke.json"
SEED="${CVAE_SEED:-20260824}"
SONIC_DIR="${SONIC_DIR:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/bly/sonic-repro/IsaacLab}"
SONIC_KIT_DIR="${SONIC_KIT_DIR:-$HOME/bly/sonic-repro-kit}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

die() {
  printf '[%s] ERROR: %s\n' "$(date -Is)" "$*" >&2
  exit 1
}

resolved_runs_root() {
  local allowed resolved
  allowed="$(realpath -m -- "$HOME/bly/runs")"
  resolved="$(realpath -m -- "$RUNS_ROOT")"
  [[ "$resolved" == "$allowed" || "$resolved" == "$allowed/"* ]] \
    || die "CVAE_RUNS_ROOT must remain under $allowed; found $resolved"
  printf '%s\n' "$resolved"
}

new_run_dir() {
  local prefix="$1" root candidate
  root="$(resolved_runs_root)"
  mkdir -p -- "$root"
  candidate="${CVAE_RUN_DIR:-$root/${prefix}_$(timestamp)}"
  candidate="$(realpath -m -- "$candidate")"
  [[ "$candidate" == "$root/"* ]] || die "run directory escapes $root: $candidate"
  [[ ! -e "$candidate" ]] || die "run directory already exists; refusing overwrite: $candidate"
  mkdir -p -- "$candidate"/{logs,videos,data,checkpoints,manifests,markers}
  printf '%s\n' "$candidate"
}

capture_environment() {
  local run_dir="$1"
  {
    printf 'invocation='; printf '%q ' "$0" "${ORIGINAL_ARGS[@]}"; printf '\n'
    printf 'cwd=%q\n' "$PWD"
    printf 'python=%q\n' "$PYTHON"
    printf 'seed=%q\n' "$SEED"
  } > "$run_dir/manifests/command.txt"
  git -C "$SCRIPT_DIR" rev-parse HEAD > "$run_dir/manifests/source_commit.txt" 2>&1 || true
  git -C "$SCRIPT_DIR" status --short --branch > "$run_dir/manifests/source_status.txt" 2>&1 || true
  git -C "$SONIC_DIR" rev-parse HEAD > "$run_dir/manifests/sonic_commit.txt" 2>&1 || true
  git -C "$SONIC_DIR" status --short --branch > "$run_dir/manifests/sonic_status.txt" 2>&1 || true
  git -C "$ISAACLAB_DIR" rev-parse HEAD > "$run_dir/manifests/isaaclab_commit.txt" 2>&1 || true
  git -C "$ISAACLAB_DIR" status --short --branch > "$run_dir/manifests/isaaclab_status.txt" 2>&1 || true
  if command -v uv >/dev/null 2>&1; then
    uv pip freeze --python "$PYTHON" > "$run_dir/manifests/environment_freeze.txt" 2>&1 || true
  else
    "$PYTHON" -m pip freeze > "$run_dir/manifests/environment_freeze.txt" 2>&1 || true
  fi
  nvidia-smi > "$run_dir/manifests/nvidia-smi.txt" 2>&1 || true
}

run_logged() {
  local run_dir="$1" log_name="$2"
  shift 2
  set +e
  "$@" 2>&1 | tee "$run_dir/logs/$log_name"
  local code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$run_dir/manifests/exit_code.txt"
  if (( code != 0 )); then
    printf 'FAIL exit_code=%s\n' "$code" > "$run_dir/markers/cvae.failed"
    return "$code"
  fi
}

update_latest() {
  local kind="$1" run_dir="$2" root temporary target
  root="$(resolved_runs_root)"
  target="$root/latest_${kind}_run_dir.txt"
  temporary="$root/.latest_${kind}_run_dir.tmp.$$"
  printf '%s\n' "$run_dir" > "$temporary"
  mv -f -- "$temporary" "$target"
}

build_index() {
  local run_dir
  run_dir="$(new_run_dir cvae_dataset)"
  capture_environment "$run_dir"
  local default_sources=(
    "/home/helloworld/bly/runs/collect_state_action_20260823_210152"
    "/home/helloworld/bly/runs/collect_state_action_20260823_235356"
    "/home/helloworld/bly/runs/collect_state_action_20260823_224638"
    "/home/helloworld/bly/runs/collect_state_action_20260824_005108"
  )
  local sources=("${default_sources[@]}")
  if [[ -n "${CVAE_SOURCE_RUNS:-}" ]]; then
    IFS=':' read -r -a sources <<< "$CVAE_SOURCE_RUNS"
  fi
  local source_args=()
  local source
  for source in "${sources[@]}"; do
    source_args+=(--source-run "$source")
  done
  run_logged "$run_dir" build_index.log \
    "$PYTHON" -m cvae_sa.indexer \
      "${source_args[@]}" \
      --output-run "$run_dir" \
      --expected-motions "${CVAE_EXPECTED_MOTIONS:-768}" \
      --expected-episodes "${CVAE_EXPECTED_EPISODES:-5120}" \
      --split-counts "${CVAE_SPLIT_COUNTS:-616,76,76}" \
      --seed "$SEED"
  [[ -f "$run_dir/markers/cvae_dataset.ok" ]] \
    || die "indexer exited without cvae_dataset.ok: $run_dir"
  update_latest cvae_dataset "$run_dir"
  printf '%s\n' "$run_dir"
}

train_model() {
  local smoke="$1" dataset_run="${CVAE_DATASET_RUN:-}" prefix config run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  if [[ "$smoke" == "true" ]]; then
    prefix="cvae_smoke"
    config="${CVAE_CONFIG:-$SMOKE_CONFIG}"
  else
    prefix="cvae_train"
    config="${CVAE_CONFIG:-$DEFAULT_CONFIG}"
  fi
  run_dir="$(new_run_dir "$prefix")"
  capture_environment "$run_dir"
  local smoke_args=()
  [[ "$smoke" == "true" ]] && smoke_args+=(--smoke)
  run_logged "$run_dir" train.log \
    "$PYTHON" -m cvae_sa.trainer \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --config "$config" \
      --model-kind "${CVAE_MODEL_KIND:-transformer}" \
      --seed "$SEED" \
      "${smoke_args[@]}"
  local marker="cvae_train.ok"
  [[ "$smoke" == "true" ]] && marker="cvae_smoke_train.ok"
  [[ -f "$run_dir/markers/$marker" ]] || die "training marker is missing: $marker"
  update_latest "$prefix" "$run_dir"
  printf '%s\n' "$run_dir"
}

evaluate_model() {
  local dataset_run="${CVAE_DATASET_RUN:-}" checkpoint="${CVAE_CHECKPOINT:-}" run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -n "$checkpoint" ]] || die "CVAE_CHECKPOINT is required"
  run_dir="$(new_run_dir cvae_eval)"
  capture_environment "$run_dir"
  local limit_args=()
  [[ -n "${CVAE_EVAL_MAX_BATCHES:-}" ]] && limit_args+=(--max-batches "$CVAE_EVAL_MAX_BATCHES")
  run_logged "$run_dir" evaluate.log \
    "$PYTHON" -m cvae_sa.evaluator \
      --dataset-run "$dataset_run" \
      --checkpoint "$checkpoint" \
      --output-run "$run_dir" \
      "${limit_args[@]}"
  [[ -f "$run_dir/markers/cvae_eval.ok" ]] || die "evaluation failed its physical gate"
  update_latest cvae_eval "$run_dir"
  printf '%s\n' "$run_dir"
}

sample_model() {
  local dataset_run="${CVAE_DATASET_RUN:-}" checkpoint="${CVAE_CHECKPOINT:-}" run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -n "$checkpoint" ]] || die "CVAE_CHECKPOINT is required"
  run_dir="$(new_run_dir cvae_sample)"
  capture_environment "$run_dir"
  local input_args=()
  [[ -n "${CVAE_SAMPLE_INPUT:-}" ]] && input_args+=(--input "$CVAE_SAMPLE_INPUT")
  [[ -n "${CVAE_SAMPLE_EPISODE:-}" ]] && input_args+=(--episode "$CVAE_SAMPLE_EPISODE")
  run_logged "$run_dir" sample.log \
    "$PYTHON" -m cvae_sa.sampler \
      --dataset-run "$dataset_run" \
      --checkpoint "$checkpoint" \
      --output-run "$run_dir" \
      --start "${CVAE_SAMPLE_START:-0}" \
      --task "${CVAE_SAMPLE_TASK:-completion}" \
      --completion "${CVAE_SAMPLE_COMPLETION:-step}" \
      --latent-samples "${CVAE_LATENT_SAMPLES:-8}" \
      "${input_args[@]}"
  [[ -f "$run_dir/markers/cvae_sample.ok" ]] || die "sample marker is missing"
  update_latest cvae_sample "$run_dir"
  printf '%s\n' "$run_dir"
}

validate_action_mask_replay() {
  local dataset_run="${CVAE_DATASET_RUN:-}" checkpoint="${CVAE_CHECKPOINT:-}"
  local split="${CVAE_REPLAY_SPLIT:-validation}" package="${CVAE_REPLAY_PACKAGE:-Locomotion}"
  local motion_key="${CVAE_REPLAY_MOTION_KEY:-auto}"
  local preset="${CVAE_MASK_PRESET:-all_action_masks_v1}"
  local latent_mode="${CVAE_REPLAY_LATENT_MODE:-prior_mean}"
  local latent_samples="${CVAE_REPLAY_LATENT_SAMPLES:-8}"
  local render_mode="${CVAE_REPLAY_RENDER:-representatives}"
  local replay_seed="${CVAE_REPLAY_SEED:-$SEED}" run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -n "$checkpoint" ]] || die "CVAE_CHECKPOINT is required"
  [[ -d "$SONIC_KIT_DIR" && -f "$SONIC_KIT_DIR/sonic_repro.sh" ]] \
    || die "SONIC reproduction kit is unavailable: $SONIC_KIT_DIR"
  [[ "$split" == "validation" ]] \
    || die "Action-mask replay is currently restricted to CVAE_REPLAY_SPLIT=validation"
  [[ "$latent_mode" == "prior_mean" ]] \
    || die "The primary replay result must use CVAE_REPLAY_LATENT_MODE=prior_mean"
  [[ "$latent_samples" =~ ^[1-9][0-9]*$ ]] \
    || die "CVAE_REPLAY_LATENT_SAMPLES must be a positive integer"
  [[ "$replay_seed" =~ ^[0-9]+$ ]] \
    || die "CVAE_REPLAY_SEED must be a non-negative integer"
  [[ "$render_mode" == "representatives" || "$render_mode" == "all" \
      || "$render_mode" == "none" ]] \
    || die "CVAE_REPLAY_RENDER must be representatives, all, or none"
  run_dir="$(new_run_dir cvae_action_mask_eval)"
  capture_environment "$run_dir"

  local -a scenario_args=()
  if [[ -n "${CVAE_MASK_SCENARIOS:-}" ]]; then
    scenario_args=(--custom-scenarios "$CVAE_MASK_SCENARIOS")
  fi
  run_logged "$run_dir" action_mask_prepare.log \
    "$PYTHON" -m cvae_sa.action_mask_eval prepare \
      --dataset-run "$dataset_run" \
      --checkpoint "$checkpoint" \
      --output-run "$run_dir" \
      --split "$split" \
      --package "$package" \
      --motion-key "$motion_key" \
      --seed "$replay_seed" \
      --preset "$preset" \
      --latent-mode "$latent_mode" \
      --latent-samples "$latent_samples" \
      --render-mode "$render_mode" \
      "${scenario_args[@]}"
  run_logged "$run_dir" action_mask_source_orchestration.log \
    env ACTION_MASK_RUN_DIR="$run_dir" \
      bash "$SONIC_KIT_DIR/sonic_repro.sh" capture-action-mask-source
  run_logged "$run_dir" action_mask_completion.log \
    "$PYTHON" -m cvae_sa.action_mask_eval complete \
      --dataset-run "$dataset_run" \
      --checkpoint "$checkpoint" \
      --output-run "$run_dir" \
      --latent-samples "$latent_samples" \
      --seed "$replay_seed" \
      "${scenario_args[@]}"
  run_logged "$run_dir" action_mask_replay_orchestration.log \
    env ACTION_MASK_RUN_DIR="$run_dir" \
      bash "$SONIC_KIT_DIR/sonic_repro.sh" replay-action-mask
  run_logged "$run_dir" action_mask_physics_metrics.log \
    "$PYTHON" -m cvae_sa.action_mask_eval physics-metrics \
      --output-run "$run_dir"
  if [[ "$render_mode" != "none" ]]; then
    run_logged "$run_dir" action_mask_render_orchestration.log \
      env ACTION_MASK_RUN_DIR="$run_dir" \
        bash "$SONIC_KIT_DIR/sonic_repro.sh" render-action-mask
  fi
  run_logged "$run_dir" action_mask_finalize.log \
    "$PYTHON" -m cvae_sa.action_mask_eval finalize \
      --output-run "$run_dir" \
      --render-mode "$render_mode"
  [[ -f "$run_dir/markers/cvae_action_mask_replay.ok" ]] \
    || die "Action-mask replay evaluation marker is missing"
  update_latest cvae_action_mask_eval "$run_dir"
  printf '%s\n' "$run_dir"
}

ORIGINAL_ARGS=("$@")
[[ -x "$PYTHON" ]] || die "Python environment is unavailable: $PYTHON"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  build-index) build_index ;;
  smoke-train) train_model true ;;
  train) train_model false ;;
  evaluate) evaluate_model ;;
  sample) sample_model ;;
  validate-action-mask-replay) validate_action_mask_replay ;;
  *) die "usage: bash ./cvae_repro.sh {build-index|smoke-train|train|evaluate|sample|validate-action-mask-replay}" ;;
esac
