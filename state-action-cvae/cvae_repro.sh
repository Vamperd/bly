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

build_physics_index() {
  [[ -n "${CVAE_SOURCE_RUNS:-}" ]] \
    || die "CVAE_SOURCE_RUNS is required for build-physics-index"
  local run_dir
  run_dir="$(new_run_dir cvae_physics_dataset)"
  capture_environment "$run_dir"
  local sources=()
  IFS=':' read -r -a sources <<< "$CVAE_SOURCE_RUNS"
  [[ "${#sources[@]}" -ge 1 ]] \
    || die "build-physics-index requires at least one collection run"
  local source_args=()
  local source
  for source in "${sources[@]}"; do
    source_args+=(--source-run "$source")
  done
  run_logged "$run_dir" build_physics_index.log \
    "$PYTHON" -m cvae_sa.physics_indexer \
      "${source_args[@]}" \
      --output-run "$run_dir" \
      --expected-motions "${CVAE_EXPECTED_MOTIONS:-2048}" \
      --expected-episodes "${CVAE_EXPECTED_EPISODES:-16384}" \
      --split-counts "${CVAE_SPLIT_COUNTS:-1638,205,205}" \
      --seed "$SEED"
  [[ -f "$run_dir/markers/cvae_physics_dataset.ok" ]] \
    || die "physics indexer exited without cvae_physics_dataset.ok: $run_dir"
  update_latest cvae_physics_dataset "$run_dir"
  printf '%s\n' "$run_dir"
}

build_overfit_subset() {
  local parent="${CVAE_DATASET_RUN:-}" run_dir seed="${CVAE_SEED:-20260828}"
  SEED="$seed"
  [[ -n "$parent" ]] || die "CVAE_DATASET_RUN is required for build-overfit-subset"
  [[ -f "$parent/markers/cvae_physics_dataset.ok" ]] \
    || die "parent Physics dataset marker is missing: $parent"
  run_dir="$(new_run_dir cvae_overfit_subset)"
  capture_environment "$run_dir"
  run_logged "$run_dir" build_overfit_subset.log \
    "$PYTHON" -m cvae_sa.overfit_subset \
      --parent-dataset-run "$parent" \
      --output-run "$run_dir" \
      --motions-per-package 4 \
      --seed "$seed"
  [[ -f "$run_dir/markers/cvae_overfit_subset.ok" ]] \
    || die "overfit subset marker is missing"
  update_latest cvae_overfit_subset "$run_dir"
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
      --context-mode "${CVAE_CONTEXT_MODE:-hidden}" \
      --seed "$SEED" \
      "${smoke_args[@]}"
  local marker="cvae_train.ok"
  [[ "$smoke" == "true" ]] && marker="cvae_smoke_train.ok"
  [[ -f "$run_dir/markers/$marker" ]] || die "training marker is missing: $marker"
  update_latest "$prefix" "$run_dir"
  printf '%s\n' "$run_dir"
}

overfit_model() {
  local phase="$1" dataset_run="${CVAE_DATASET_RUN:-}"
  local profile="${CVAE_OVERFIT_MODEL:-compact}" checkpoint="${CVAE_INIT_CHECKPOINT:-}"
  local config prefix marker run_dir seed="${CVAE_SEED:-20260828}" model_kind="physics_transformer"
  local overfit_smoke="${CVAE_OVERFIT_SMOKE:-false}"
  SEED="$seed"
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -f "$dataset_run/markers/cvae_overfit_subset.ok" ]] \
    || die "dedicated overfit subset marker is missing: $dataset_run"
  if [[ "$overfit_smoke" == "true" ]]; then
    [[ "$phase" == "capacity" ]] || die "overfit smoke supports capacity phase only"
    config="$SCRIPT_DIR/configs/overfit_32_smoke.json"
    profile="smoke"
  else case "$profile:$phase" in
    compact:capacity) config="$SCRIPT_DIR/configs/overfit_32_compact_capacity.json" ;;
    compact:full) config="$SCRIPT_DIR/configs/overfit_32_compact_full.json" ;;
    reference:capacity) config="$SCRIPT_DIR/configs/overfit_32_reference_capacity.json" ;;
    reference:full) config="$SCRIPT_DIR/configs/overfit_32_reference_full.json" ;;
    joint_id_only:capacity) config="$SCRIPT_DIR/configs/overfit_32_joint_id_capacity.json" ;;
    joint_id_only:full) config="$SCRIPT_DIR/configs/overfit_32_joint_id_full.json" ;;
    no_aux:full) config="$SCRIPT_DIR/configs/overfit_32_no_aux_full.json" ;;
    lean:capacity) config="$SCRIPT_DIR/configs/overfit_32_lean_capacity.json"; model_kind="physics_lean_split" ;;
    lean:full) config="$SCRIPT_DIR/configs/overfit_32_lean_full.json"; model_kind="physics_lean_split" ;;
    *) die "unsupported CVAE_OVERFIT_MODEL/phase combination: $profile/$phase" ;;
  esac; fi
  config="${CVAE_CONFIG:-$config}"
  if [[ "$phase" == "full" ]]; then
    [[ -n "$checkpoint" && -f "$checkpoint" ]] \
      || die "CVAE_INIT_CHECKPOINT is required for full overfit phase"
  elif [[ -n "$checkpoint" ]]; then
    die "capacity overfit phase must start from random initialization"
  fi
  prefix="cvae_overfit_${phase}_${profile}"
  marker="cvae_overfit_${phase}.ok"
  [[ "$overfit_smoke" == "true" ]] && marker="cvae_overfit_smoke.ok"
  run_dir="$(new_run_dir "$prefix")"
  capture_environment "$run_dir"
  local init_args=()
  [[ "$phase" == "full" ]] && init_args+=(--init-checkpoint "$checkpoint")
  [[ "$overfit_smoke" == "true" ]] && init_args+=(--smoke)
  run_logged "$run_dir" "overfit_${phase}.log" \
    "$PYTHON" -m cvae_sa.trainer \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --config "$config" \
      --model-kind "$model_kind" \
      --context-mode hidden \
      --seed "$seed" \
      --overfit-phase "$phase" \
      "${init_args[@]}"
  [[ -f "$run_dir/markers/$marker" ]] || die "overfit marker is missing: $marker"
  update_latest "overfit_${phase}_${profile}" "$run_dir"
  printf '%s\n' "$run_dir"
}

posterior_capacity() {
  local smoke="$1" dataset_run="${CVAE_DATASET_RUN:-}"
  local phase="${CVAE_POSTERIOR_PHASE:-fixed}" motions="${CVAE_POSTERIOR_MOTIONS:-1}"
  local window="${CVAE_POSTERIOR_WINDOW:-16}" checkpoint="${CVAE_INIT_CHECKPOINT:-}"
  local max_windows="${CVAE_POSTERIOR_MAX_WINDOWS:-}"
  local max_steps="${CVAE_POSTERIOR_MAX_STEPS:-}"
  local acceptance_gate="${CVAE_POSTERIOR_GATE:-exact}"
  local config="${CVAE_CONFIG:-$SCRIPT_DIR/configs/posterior_capacity_minimal.json}"
  local prefix marker run_dir latest_key seed="${CVAE_SEED:-20260830}"
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -f "$dataset_run/markers/cvae_overfit_subset.ok" ]] \
    || die "dedicated overfit subset marker is missing: $dataset_run"
  [[ "$phase" == "fixed" || "$phase" == "generalization" ]] \
    || die "CVAE_POSTERIOR_PHASE must be fixed or generalization"
  [[ -z "$max_windows" || "$max_windows" =~ ^[1-9][0-9]*$ ]] \
    || die "CVAE_POSTERIOR_MAX_WINDOWS must be a positive integer"
  [[ -z "$max_steps" || "$max_steps" =~ ^[1-9][0-9]*$ ]] \
    || die "CVAE_POSTERIOR_MAX_STEPS must be a positive integer"
  [[ "$acceptance_gate" == "exact" || "$acceptance_gate" == "progression" ]] \
    || die "CVAE_POSTERIOR_GATE must be exact or progression"
  if [[ "$phase" == "generalization" ]]; then
    [[ -n "$checkpoint" && -f "$checkpoint" ]] \
      || die "generalization requires CVAE_INIT_CHECKPOINT=.../best_<gate>.pt"
  elif [[ -n "$checkpoint" ]]; then
    die "fixed posterior capacity must start from random initialization"
  fi
  prefix="cvae_posterior_capacity_${phase}_m${motions}_t${window}"
  [[ -n "$max_windows" ]] && prefix="${prefix}_w${max_windows}"
  [[ -n "$max_steps" ]] && prefix="${prefix}_s${max_steps}"
  [[ "$acceptance_gate" == "progression" ]] && prefix="${prefix}_gprogression"
  marker="cvae_posterior_capacity.ok"
  [[ "$phase" == "generalization" ]] && marker="cvae_posterior_mask_generalization.ok"
  if [[ "$acceptance_gate" == "progression" ]]; then
    marker="cvae_posterior_capacity_progression.ok"
    [[ "$phase" == "generalization" ]] \
      && marker="cvae_posterior_mask_generalization_progression.ok"
  fi
  [[ "$smoke" == "true" ]] && marker="cvae_posterior_capacity_smoke.ok"
  run_dir="$(new_run_dir "$prefix")"
  capture_environment "$run_dir"
  local extra_args=()
  [[ "$phase" == "generalization" ]] && extra_args+=(--init-checkpoint "$checkpoint")
  [[ "$smoke" == "true" ]] && extra_args+=(--smoke)
  [[ -n "$max_windows" ]] && extra_args+=(--max-windows "$max_windows")
  [[ -n "$max_steps" ]] && extra_args+=(--max-optimizer-steps "$max_steps")
  run_logged "$run_dir" posterior_capacity.log \
    "$PYTHON" -m cvae_sa.posterior_capacity \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --config "$config" \
      --motions "$motions" \
      --window-transitions "$window" \
      --mask-phase "$phase" \
      --acceptance-gate "$acceptance_gate" \
      --seed "$seed" \
      "${extra_args[@]}"
  [[ -f "$run_dir/markers/$marker" ]] || die "posterior capacity marker is missing: $marker"
  latest_key="posterior_capacity_${phase}"
  [[ "$acceptance_gate" == "progression" ]] \
    && latest_key="posterior_capacity_${phase}_progression"
  update_latest "$latest_key" "$run_dir"
  printf '%s\n' "$run_dir"
}

overfit_single_task() {
  local dataset_run="${CVAE_DATASET_RUN:-}" task="${CVAE_OVERFIT_TASK:-}"
  local seed="${CVAE_SEED:-20260828}" profile="${CVAE_OVERFIT_MODEL:-compact}"
  local config run_dir model_kind="physics_transformer" config_prefix="overfit_32_single"
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -f "$dataset_run/markers/cvae_overfit_subset.ok" ]] \
    || die "dedicated overfit subset marker is missing: $dataset_run"
  if [[ "$profile" == "lean" ]]; then
    model_kind="physics_lean_split"
    config_prefix="overfit_32_lean_single"
  elif [[ "$profile" != "compact" ]]; then
    die "single-task CVAE_OVERFIT_MODEL must be compact or lean"
  fi
  case "$task" in
    forward_rollout) config="$SCRIPT_DIR/configs/${config_prefix}_forward_rollout.json" ;;
    inverse) config="$SCRIPT_DIR/configs/${config_prefix}_inverse.json" ;;
    history_action) config="$SCRIPT_DIR/configs/${config_prefix}_history_action.json" ;;
    arbitrary_state) config="$SCRIPT_DIR/configs/${config_prefix}_arbitrary_state.json" ;;
    arbitrary_action) config="$SCRIPT_DIR/configs/${config_prefix}_arbitrary_action.json" ;;
    *) die "CVAE_OVERFIT_TASK must be forward_rollout, inverse, history_action, arbitrary_state, or arbitrary_action" ;;
  esac
  config="${CVAE_CONFIG:-$config}"
  run_dir="$(new_run_dir "cvae_overfit_single_${profile}_${task}")"
  capture_environment "$run_dir"
  run_logged "$run_dir" "overfit_single_${task}.log" \
    "$PYTHON" -m cvae_sa.trainer \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --config "$config" \
      --model-kind "$model_kind" \
      --context-mode hidden \
      --seed "$seed" \
      --overfit-phase capacity
  [[ -f "$run_dir/markers/cvae_overfit_single_task.ok" ]] \
    || die "single-task execution marker is missing"
  update_latest "overfit_single_${profile}_${task}" "$run_dir"
  printf '%s\n' "$run_dir"
}

analyze_overfit() {
  local dataset_run="${CVAE_DATASET_RUN:-}" checkpoint="${CVAE_ANALYSIS_CHECKPOINT:-}"
  local run_dir seed="${CVAE_SEED:-20260828}"
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -f "$dataset_run/markers/cvae_overfit_subset.ok" ]] \
    || die "dedicated overfit subset marker is missing: $dataset_run"
  local checkpoint_args=()
  if [[ -n "$checkpoint" ]]; then
    [[ -f "$checkpoint" ]] || die "CVAE_ANALYSIS_CHECKPOINT is missing: $checkpoint"
    checkpoint_args+=(--checkpoint "$checkpoint")
  fi
  run_dir="$(new_run_dir cvae_overfit_analysis)"
  capture_environment "$run_dir"
  run_logged "$run_dir" overfit_analysis.log \
    "$PYTHON" -m cvae_sa.overfit_analysis \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --history-steps "${CVAE_ANALYSIS_HISTORY_STEPS:-1,4,10,32}" \
      --max-samples "${CVAE_ANALYSIS_MAX_SAMPLES:-4096}" \
      --max-sensitivity-batches "${CVAE_ANALYSIS_MAX_BATCHES:-16}" \
      --seed "$seed" \
      "${checkpoint_args[@]}"
  [[ -f "$run_dir/markers/cvae_overfit_analysis.ok" ]] \
    || die "overfit analysis marker is missing"
  update_latest cvae_overfit_analysis "$run_dir"
  printf '%s\n' "$run_dir"
}

summarize_overfit() {
  local encoded="${CVAE_OVERFIT_RUNS:-}" run_dir
  [[ -n "$encoded" ]] || die "CVAE_OVERFIT_RUNS is required (colon-separated run directories)"
  local runs=()
  IFS=':' read -r -a runs <<< "$encoded"
  local run_args=() item
  for item in "${runs[@]}"; do
    [[ -f "$item/manifests/training_summary.json" ]] \
      || die "overfit training summary is missing: $item"
    run_args+=(--run "$item")
  done
  run_dir="$(new_run_dir cvae_overfit_suite)"
  capture_environment "$run_dir"
  run_logged "$run_dir" summarize_overfit.log \
    "$PYTHON" -m cvae_sa.overfit_report \
      "${run_args[@]}" --output-run "$run_dir"
  [[ -f "$run_dir/markers/cvae_overfit_suite.ok" ]] \
    || die "overfit suite marker is missing"
  update_latest cvae_overfit_suite "$run_dir"
  printf '%s\n' "$run_dir"
}

summarize_single_tasks() {
  local encoded="${CVAE_OVERFIT_RUNS:-}" run_dir
  [[ -n "$encoded" ]] || die "CVAE_OVERFIT_RUNS is required (colon-separated run directories)"
  local runs=()
  IFS=':' read -r -a runs <<< "$encoded"
  local run_args=() item
  for item in "${runs[@]}"; do
    [[ -f "$item/manifests/training_summary.json" ]] \
      || die "single-task training summary is missing: $item"
    run_args+=(--run "$item")
  done
  run_dir="$(new_run_dir cvae_overfit_single_suite)"
  capture_environment "$run_dir"
  run_logged "$run_dir" summarize_single_tasks.log \
    "$PYTHON" -m cvae_sa.overfit_single_report \
      "${run_args[@]}" --output-run "$run_dir"
  [[ -f "$run_dir/markers/cvae_overfit_single_suite.ok" ]] \
    || die "single-task suite did not pass; diagnostics are preserved in $run_dir"
  update_latest cvae_overfit_single_suite "$run_dir"
  printf '%s\n' "$run_dir"
}

diagnose_overfit_fixture() {
  local dataset_run="${CVAE_DATASET_RUN:-}" encoded="${CVAE_OVERFIT_RUNS:-}"
  local checkpoint_kinds="${CVAE_FIXTURE_CHECKPOINTS:-best,last}" run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -f "$dataset_run/markers/cvae_overfit_subset.ok" ]] \
    || die "dedicated overfit subset marker is missing: $dataset_run"
  [[ -n "$encoded" ]] \
    || die "CVAE_OVERFIT_RUNS is required (five colon-separated single-task runs)"
  local runs=()
  IFS=':' read -r -a runs <<< "$encoded"
  [[ "${#runs[@]}" -eq 5 ]] \
    || die "diagnose-overfit-fixture requires exactly five single-task runs"
  local run_args=() item
  for item in "${runs[@]}"; do
    [[ -f "$item/manifests/training_summary.json" ]] \
      || die "single-task training summary is missing: $item"
    run_args+=(--training-run "$item")
  done
  run_dir="$(new_run_dir cvae_overfit_fixture_diagnostic)"
  capture_environment "$run_dir"
  run_logged "$run_dir" overfit_fixture_diagnostic.log \
    "$PYTHON" -m cvae_sa.overfit_fixture_eval \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --checkpoint-kinds "$checkpoint_kinds" \
      "${run_args[@]}"
  [[ -f "$run_dir/markers/cvae_overfit_fixture_diagnostic.ok" ]] \
    || die "exact fixture diagnostic execution marker is missing"
  update_latest cvae_overfit_fixture_diagnostic "$run_dir"
  printf '%s\n' "$run_dir"
}

action_finetune_model() {
  local smoke="$1" dataset_run="${CVAE_DATASET_RUN:-}"
  local checkpoint="${CVAE_INIT_CHECKPOINT:-}" prefix config marker run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -n "$checkpoint" ]] || die "CVAE_INIT_CHECKPOINT is required"
  [[ -f "$checkpoint" ]] || die "initial checkpoint is missing: $checkpoint"
  if [[ "$smoke" == "true" ]]; then
    prefix="cvae_action_finetune_smoke"
    config="${CVAE_CONFIG:-$SCRIPT_DIR/configs/physics_v3_action_finetune_smoke.json}"
    marker="cvae_action_finetune_smoke.ok"
  else
    prefix="cvae_action_finetune"
    config="${CVAE_CONFIG:-$SCRIPT_DIR/configs/physics_v3_action_finetune.json}"
    marker="cvae_action_finetune.ok"
  fi
  run_dir="$(new_run_dir "$prefix")"
  capture_environment "$run_dir"
  local smoke_args=()
  [[ "$smoke" == "true" ]] && smoke_args+=(--smoke)
  run_logged "$run_dir" action_finetune.log \
    "$PYTHON" -m cvae_sa.trainer \
      --dataset-run "$dataset_run" \
      --output-run "$run_dir" \
      --config "$config" \
      --model-kind "${CVAE_MODEL_KIND:-physics_transformer}" \
      --context-mode "${CVAE_CONTEXT_MODE:-hidden}" \
      --seed "$SEED" \
      --init-checkpoint "$checkpoint" \
      --fine-tune-mode action \
      "${smoke_args[@]}"
  [[ -f "$run_dir/markers/$marker" ]] \
    || die "Action fine-tune marker is missing: $marker"
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
  [[ "$latent_mode" == "prior_mean" || "$latent_mode" == "oracle_best_of_n" ]] \
    || die "CVAE_REPLAY_LATENT_MODE must be prior_mean or oracle_best_of_n"
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

validate_state_mask_video() {
  local dataset_run="${CVAE_DATASET_RUN:-}" checkpoint="${CVAE_CHECKPOINT:-}"
  local split="${CVAE_STATE_SPLIT:-validation}" package="${CVAE_STATE_PACKAGE:-Locomotion}"
  local motion_key="${CVAE_STATE_MOTION_KEY:-auto}" variant="${CVAE_STATE_VARIANT:-auto}"
  local preset="${CVAE_STATE_MASK_PRESET:-state_prediction_v1}"
  local latent_mode="${CVAE_STATE_LATENT_MODE:-prior_mean}"
  local latent_samples="${CVAE_STATE_LATENT_SAMPLES:-8}"
  local render_mode="${CVAE_STATE_RENDER:-representatives}"
  local root_mode="${CVAE_STATE_ROOT_MODE:-integrate_predicted}"
  local state_seed="${CVAE_STATE_SEED:-$SEED}" run_dir
  [[ -n "$dataset_run" ]] || die "CVAE_DATASET_RUN is required"
  [[ -n "$checkpoint" ]] || die "CVAE_CHECKPOINT is required"
  [[ -d "$SONIC_KIT_DIR" && -f "$SONIC_KIT_DIR/sonic_repro.sh" ]] \
    || die "SONIC reproduction kit is unavailable: $SONIC_KIT_DIR"
  [[ "$split" == "validation" ]] \
    || die "State-mask video validation is restricted to CVAE_STATE_SPLIT=validation"
  [[ "$latent_mode" == "prior_mean" ]] \
    || die "CVAE_STATE_LATENT_MODE must be prior_mean"
  [[ "$latent_samples" =~ ^[1-9][0-9]*$ ]] \
    || die "CVAE_STATE_LATENT_SAMPLES must be a positive integer"
  [[ "$render_mode" == "representatives" || "$render_mode" == "all" \
      || "$render_mode" == "none" ]] \
    || die "CVAE_STATE_RENDER must be representatives, all, or none"
  [[ "$root_mode" == "integrate_predicted" ]] \
    || die "CVAE_STATE_ROOT_MODE must be integrate_predicted"
  [[ "$state_seed" =~ ^[0-9]+$ ]] \
    || die "CVAE_STATE_SEED must be a non-negative integer"
  run_dir="$(new_run_dir cvae_state_mask_eval)"
  capture_environment "$run_dir"
  run_logged "$run_dir" state_mask_evaluate.log \
    "$PYTHON" -m cvae_sa.state_mask_eval evaluate \
      --dataset-run "$dataset_run" \
      --checkpoint "$checkpoint" \
      --output-run "$run_dir" \
      --split "$split" \
      --package "$package" \
      --motion-key "$motion_key" \
      --variant "$variant" \
      --preset "$preset" \
      --latent-mode "$latent_mode" \
      --latent-samples "$latent_samples" \
      --render-mode "$render_mode" \
      --root-mode "$root_mode" \
      --seed "$state_seed"
  if [[ "$render_mode" != "none" ]]; then
    run_logged "$run_dir" state_mask_render_orchestration.log \
      env STATE_MASK_RUN_DIR="$run_dir" \
        bash "$SONIC_KIT_DIR/sonic_repro.sh" render-state-mask
  fi
  run_logged "$run_dir" state_mask_finalize.log \
    "$PYTHON" -m cvae_sa.state_mask_eval finalize \
      --output-run "$run_dir" \
      --render-mode "$render_mode"
  [[ -f "$run_dir/markers/cvae_state_mask_video.ok" ]] \
    || die "State-mask video marker is missing"
  update_latest cvae_state_mask_eval "$run_dir"
  printf '%s\n' "$run_dir"
}

ORIGINAL_ARGS=("$@")
[[ -x "$PYTHON" ]] || die "Python environment is unavailable: $PYTHON"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  build-index) build_index ;;
  build-physics-index) build_physics_index ;;
  build-overfit-subset) build_overfit_subset ;;
  smoke-train) train_model true ;;
  train) train_model false ;;
  overfit-capacity) overfit_model capacity ;;
  overfit-full) overfit_model full ;;
  overfit-single-task) overfit_single_task ;;
  posterior-capacity-smoke) posterior_capacity true ;;
  posterior-capacity) posterior_capacity false ;;
  analyze-overfit) analyze_overfit ;;
  diagnose-overfit-fixture) diagnose_overfit_fixture ;;
  summarize-overfit) summarize_overfit ;;
  summarize-single-tasks) summarize_single_tasks ;;
  smoke-action-finetune) action_finetune_model true ;;
  action-finetune) action_finetune_model false ;;
  evaluate) evaluate_model ;;
  sample) sample_model ;;
  validate-action-mask-replay) validate_action_mask_replay ;;
  validate-state-mask-video) validate_state_mask_video ;;
  *) die "usage: bash ./cvae_repro.sh {build-index|build-physics-index|build-overfit-subset|smoke-train|train|overfit-capacity|overfit-full|overfit-single-task|posterior-capacity-smoke|posterior-capacity|analyze-overfit|diagnose-overfit-fixture|summarize-overfit|summarize-single-tasks|smoke-action-finetune|action-finetune|evaluate|sample|validate-action-mask-replay|validate-state-mask-video}" ;;
esac
