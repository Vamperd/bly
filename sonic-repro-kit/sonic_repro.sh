#!/usr/bin/env bash

set -Eeuo pipefail

# Reproducible SONIC minimal-reproduction runner for Ubuntu 24.04 + RTX 4090.
# It never deletes an existing environment, repository, dataset, or run directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${SONIC_WORK_ROOT:-$HOME/bly/sonic-repro}"
VENV_DIR="${SONIC_VENV_DIR:-$WORK_ROOT/.venv-sonic}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$WORK_ROOT/IsaacLab}"
SONIC_DIR="${SONIC_DIR:-$WORK_ROOT/GR00T-WholeBodyControl}"
STATE_DIR="${SONIC_STATE_DIR:-$WORK_ROOT/state}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK_ROOT/.uv-cache}"
RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.3.2}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-37ddf626871758333d6ed89cf64ad702aef127d0}"
# Official main HEAD observed on 2026-08-18. Override explicitly to test another revision.
SONIC_COMMIT="${SONIC_COMMIT:-c374bae5b9039cd0ee71377e654d11ce1bc69e1d}"

MIN_INSTALL_FREE_GIB="${MIN_INSTALL_FREE_GIB:-50}"
MAX_RUN_GPU_USED_MIB="${MAX_RUN_GPU_USED_MIB:-3000}"
EVAL_ENVS="${EVAL_ENVS:-32}"
RENDER_ENVS="${RENDER_ENVS:-8}"
SMOKE_ENVS="${SMOKE_ENVS:-16}"
COLLECT_MOTION_COUNT="${COLLECT_MOTION_COUNT:-all}"
COLLECT_BATCH_MOTIONS="${COLLECT_BATCH_MOTIONS:-8}"
COLLECT_RANDOMIZATION_PROFILE="${COLLECT_RANDOMIZATION_PROFILE:-startup}"
if [[ -n "${COLLECT_VARIANTS_PER_MOTION+x}" ]]; then
  COLLECT_VARIANTS_PER_MOTION="$COLLECT_VARIANTS_PER_MOTION"
elif [[ "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]]; then
  COLLECT_VARIANTS_PER_MOTION=2
else
  COLLECT_VARIANTS_PER_MOTION=4
fi
COLLECT_ENVS="${COLLECT_ENVS:-}"
COLLECT_SEED="${COLLECT_SEED:-20260823}"
COLLECT_VARIANT_OFFSET="${COLLECT_VARIANT_OFFSET:-}"
COLLECT_MOTION_FILE="${COLLECT_MOTION_FILE:-sample_data/robot_filtered}"
COLLECT_SMPL_MOTION_FILE="${COLLECT_SMPL_MOTION_FILE:-zeros}"
COLLECT_MOTION_MANIFEST="${COLLECT_MOTION_MANIFEST:-}"
COLLECT_BASELINE_SUMMARY="${COLLECT_BASELINE_SUMMARY:-}"
COLLECT_DATASET_NAME="${COLLECT_DATASET_NAME:-sonic_minimal_sa}"
COLLECT_RUN_DIR="${COLLECT_RUN_DIR:-}"
BONES_MIN_DOWNLOAD_FREE_GIB="${BONES_MIN_DOWNLOAD_FREE_GIB:-70}"
BONES_MIN_STAGE_FREE_GIB="${BONES_MIN_STAGE_FREE_GIB:-40}"
OFFLINE_RENDER_ENVS="${OFFLINE_RENDER_ENVS:-1}"
OFFLINE_FRAME_SKIP="${OFFLINE_FRAME_SKIP:-2}"
OFFLINE_WIDTH="${OFFLINE_WIDTH:-960}"
OFFLINE_HEIGHT="${OFFLINE_HEIGHT:-540}"
OFFLINE_GL="${OFFLINE_GL:-egl}"
OFFLINE_CAMERA_DISTANCE="${OFFLINE_CAMERA_DISTANCE:-2.0}"
ACTION_MASK_RUN_DIR="${ACTION_MASK_RUN_DIR:-}"
ACTION_MASK_GL="${ACTION_MASK_GL:-$OFFLINE_GL}"
ACTION_MASK_WIDTH="${ACTION_MASK_WIDTH:-$OFFLINE_WIDTH}"
ACTION_MASK_HEIGHT="${ACTION_MASK_HEIGHT:-$OFFLINE_HEIGHT}"
ACTION_MASK_CAMERA_DISTANCE="${ACTION_MASK_CAMERA_DISTANCE:-$OFFLINE_CAMERA_DISTANCE}"

printf -v SCRIPT_INVOCATION '%q ' "$0" "$@"

export UV_CACHE_DIR

timestamp() {
  date +%Y%m%d_%H%M%S
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

resolved_runs_root() {
  need_command realpath
  local allowed_root resolved_root
  allowed_root="$(realpath -m -- "$HOME/bly/runs")"
  resolved_root="$(realpath -m -- "$RUNS_ROOT")"
  [[ "$resolved_root" == "$allowed_root" || "$resolved_root" == "$allowed_root/"* ]] \
    || die "RUNS_ROOT must be $allowed_root or one of its subdirectories; found $resolved_root"
  printf '%s\n' "$resolved_root"
}

validated_run_dir() {
  local candidate="$1"
  local root resolved
  root="$(resolved_runs_root)"
  resolved="$(realpath -m -- "$candidate")"
  [[ "$resolved" == "$root/"* ]] \
    || die "Run directory escapes RUNS_ROOT: $resolved"
  printf '%s\n' "$resolved"
}

ensure_run_layout() {
  local run_dir
  run_dir="$(validated_run_dir "$1")"
  mkdir -p \
    "$run_dir/logs" \
    "$run_dir/videos" \
    "$run_dir/data" \
    "$run_dir/checkpoints" \
    "$run_dir/manifests" \
    "$run_dir/markers"
}

prepare_dirs() {
  local runs_root
  runs_root="$(resolved_runs_root)"
  mkdir -p "$WORK_ROOT" "$STATE_DIR" "$runs_root"
}

activate_env() {
  [[ -f "$VENV_DIR/bin/activate" ]] || die "uv environment not found: $VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

free_gib_for_path() {
  df -Pk "$1" | awk 'NR==2 {printf "%d\n", $4/1024/1024}'
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN {sum=0} {gsub(/ /, "", $1); sum += $1} END {print sum+0}'
}

require_run_gpu_capacity() {
  need_command nvidia-smi
  local used
  used="$(gpu_used_mib)"
  if (( used > MAX_RUN_GPU_USED_MIB )); then
    nvidia-smi
    nvidia-smi pmon -c 1 || true
    die "GPU memory in use is ${used} MiB; stop or coordinate existing compute jobs before SONIC. Required <= ${MAX_RUN_GPU_USED_MIB} MiB."
  fi
  log "GPU gate passed: ${used} MiB currently used."
}

phase_preflight() {
  prepare_dirs
  local report="$STATE_DIR/preflight_$(timestamp).log"
  {
    echo "generated_at=$(date -Is)"
    echo "work_root=$WORK_ROOT"
    echo "host=$(hostname)"
    echo "kernel=$(uname -srmo)"
    echo
    cat /etc/os-release
    echo
    ldd --version | sed -n "1p"
    echo
    free -h
    echo
    df -h "$HOME" "$WORK_ROOT"
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi
      echo
      nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
        --format=csv,noheader
    else
      echo "nvidia-smi: missing"
    fi
    echo
    command -v uv >/dev/null 2>&1 && uv --version || echo "uv: missing"
    command -v git >/dev/null 2>&1 && git --version || echo "git: missing"
    command -v git-lfs >/dev/null 2>&1 && git lfs version || echo "git-lfs: missing"
  } 2>&1 | tee "$report"

  local free_gib
  free_gib="$(free_gib_for_path "$WORK_ROOT")"
  if (( free_gib < MIN_INSTALL_FREE_GIB )); then
    die "Only ${free_gib} GiB free at $WORK_ROOT; minimal installation requires at least ${MIN_INSTALL_FREE_GIB} GiB free."
  fi
  log "Preflight report: $report"
}

phase_install_system() {
  need_command sudo
  sudo apt-get update
  sudo apt-get install -y git git-lfs cmake build-essential curl
  git lfs install
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    need_command curl
    log "uv is missing; invoking Astral's official installer."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  need_command uv
  uv --version
}

phase_install_env() {
  prepare_dirs
  ensure_uv
  need_command git

  local free_gib
  free_gib="$(free_gib_for_path "$WORK_ROOT")"
  (( free_gib >= MIN_INSTALL_FREE_GIB )) \
    || die "Need at least ${MIN_INSTALL_FREE_GIB} GiB free; found ${free_gib} GiB."

  uv python install "$PYTHON_VERSION"
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
  else
    log "Keeping existing uv environment: $VENV_DIR"
  fi
  activate_env

  uv pip install --upgrade pip
  uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com
  uv pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

  if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
    git clone --branch "$ISAACLAB_REF" --depth 1 \
      https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  fi
  local isaaclab_actual
  isaaclab_actual="$(git -C "$ISAACLAB_DIR" rev-parse HEAD)"
  [[ "$isaaclab_actual" == "$ISAACLAB_COMMIT" ]] \
    || die "Isaac Lab checkout mismatch: expected $ISAACLAB_COMMIT, found $isaaclab_actual"

  (
    cd "$ISAACLAB_DIR"
    ./isaaclab.sh --install none
  )

  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
import isaaclab
print("Isaac Lab:", getattr(isaaclab, "__version__", "unknown"))
PY

  uv pip freeze > "$STATE_DIR/environment_freeze_$(timestamp).txt"
  log "Environment installed. Run verify-isaac next; first launch may request NVIDIA EULA acceptance."
}

phase_verify_isaac() {
  activate_env
  [[ -d "$ISAACLAB_DIR" ]] || die "Isaac Lab checkout not found: $ISAACLAB_DIR"
  require_run_gpu_capacity
  (
    cd "$ISAACLAB_DIR"
    python scripts/tutorials/00_sim/create_empty.py --headless
  )
}

phase_clone_sonic() {
  prepare_dirs
  activate_env
  need_command git
  need_command git-lfs

  if [[ ! -d "$SONIC_DIR/.git" ]]; then
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$SONIC_DIR"
  else
    local sonic_origin sonic_dirty
    sonic_origin="$(git -C "$SONIC_DIR" remote get-url origin)"
    [[ "$sonic_origin" == "https://github.com/NVlabs/GR00T-WholeBodyControl.git" \
      || "$sonic_origin" == "git@github.com:NVlabs/GR00T-WholeBodyControl.git" ]] \
      || die "Existing SONIC directory has an unexpected origin: $sonic_origin"
    sonic_dirty="$(git -C "$SONIC_DIR" status --porcelain)"
    [[ -z "$sonic_dirty" ]] \
      || die "Existing SONIC checkout has local changes. Preserve or commit them before switching revisions."
  fi
  git -C "$SONIC_DIR" cat-file -e "${SONIC_COMMIT}^{commit}" 2>/dev/null \
    || git -C "$SONIC_DIR" fetch origin "$SONIC_COMMIT"
  git -C "$SONIC_DIR" checkout --detach "$SONIC_COMMIT"
  git -C "$SONIC_DIR" lfs pull

  uv pip install -e "$SONIC_DIR/gear_sonic[training]"
  uv pip install huggingface_hub

  git -C "$SONIC_DIR" rev-parse HEAD > "$STATE_DIR/sonic_commit.txt"
  git -C "$ISAACLAB_DIR" rev-parse HEAD > "$STATE_DIR/isaaclab_commit.txt"
  uv pip freeze > "$STATE_DIR/environment_freeze.txt"
}

phase_download_sample() {
  activate_env
  [[ -d "$SONIC_DIR/.git" ]] || die "SONIC repository not found: $SONIC_DIR"
  (
    cd "$SONIC_DIR"
    python download_from_hf.py --training --no-smpl
    python download_from_hf.py --sample
    test -f sonic_release/last.pt
    test -f sonic_release/config.yaml
    test -d sample_data/robot_filtered
    test -d sample_data/smpl_filtered
    du -sh sonic_release sample_data
    python check_environment.py --training
  ) 2>&1 | tee "$STATE_DIR/sample_download_and_check_$(timestamp).log"
}

new_run_dir() {
  local task_name="$1"
  local root dir
  root="$(resolved_runs_root)"
  dir="$(validated_run_dir "$root/${task_name}_$(timestamp)")"
  [[ ! -e "$dir" ]] || die "Run directory already exists: $dir"
  ensure_run_layout "$dir"
  printf '%s\n' "$dir"
}

latest_run_dir_or_new() {
  local task_name="$1"
  local candidate raw_candidate
  if [[ -f "$STATE_DIR/latest_run_dir.txt" ]]; then
    raw_candidate="$(<"$STATE_DIR/latest_run_dir.txt")"
    candidate="$raw_candidate"
    if candidate="$(validated_run_dir "$candidate")" && [[ -d "$candidate" ]]; then
      ensure_run_layout "$candidate"
      printf '%s\n' "$candidate"
      return
    fi
    log "Ignoring stale or invalid latest run pointer: $raw_candidate" >&2
  fi
  new_run_dir "$task_name"
}

resolve_offline_run_dir() {
  local candidate="${OFFLINE_RUN_DIR:-}"
  if [[ -z "$candidate" ]]; then
    [[ -f "$STATE_DIR/latest_run_dir.txt" ]] \
      || die "No latest run pointer. Run dump-trajectory first or set OFFLINE_RUN_DIR."
    candidate="$(<"$STATE_DIR/latest_run_dir.txt")"
  fi
  candidate="$(validated_run_dir "$candidate")"
  [[ -d "$candidate" ]] || die "Run directory does not exist: $candidate"
  ensure_run_layout "$candidate"
  printf '%s\n' "$candidate"
}

update_latest_run_pointer() {
  local run_dir tmp
  run_dir="$(validated_run_dir "$1")"
  tmp="$STATE_DIR/latest_run_dir.txt.tmp.$$"
  printf '%s\n' "$run_dir" > "$tmp"
  mv -f -- "$tmp" "$STATE_DIR/latest_run_dir.txt"
}

mark_stage() {
  local run_dir="$1"
  local marker_name="$2"
  local marker tmp
  marker="$run_dir/markers/$marker_name"
  tmp="$marker.tmp.$$"
  printf 'completed_at=%s\n' "$(date -Is)" > "$tmp"
  mv -f -- "$tmp" "$marker"
}

record_exit_code() {
  local run_dir="$1"
  local stage_name="$2"
  local exit_code="$3"
  printf '%s\n' "$exit_code" > "$run_dir/manifests/${stage_name}_exit_code.txt"
}

prepare_eval_checkpoint() {
  local run_dir="$1"
  local input_dir="$run_dir/checkpoints/eval_input"
  local source_checkpoint="$SONIC_DIR/sonic_release/last.pt"
  local source_config="$SONIC_DIR/sonic_release/config.yaml"
  [[ -f "$source_checkpoint" ]] || die "Checkpoint missing: $source_checkpoint"
  [[ -f "$source_config" ]] || die "Checkpoint config missing: $source_config"
  mkdir -p "$input_dir"
  if [[ ! -e "$input_dir/last.pt" && ! -L "$input_dir/last.pt" ]]; then
    ln -s -- "$source_checkpoint" "$input_dir/last.pt"
  fi
  cp -- "$source_config" "$input_dir/config.yaml"
  printf '%s\n' "$input_dir/last.pt"
}

write_run_manifest() {
  local run_dir="$1"
  ensure_run_layout "$run_dir"
  nvidia-smi > "$run_dir/manifests/nvidia-smi.txt"
  git -C "$SONIC_DIR" status --short --branch > "$run_dir/manifests/sonic_status.txt"
  git -C "$SONIC_DIR" rev-parse HEAD > "$run_dir/manifests/sonic_commit.txt"
  git -C "$ISAACLAB_DIR" status --short --branch > "$run_dir/manifests/isaaclab_status.txt"
  git -C "$ISAACLAB_DIR" rev-parse HEAD > "$run_dir/manifests/isaaclab_commit.txt"
  uv pip freeze > "$run_dir/manifests/environment_freeze.txt"
  printf '%s\n' "$SCRIPT_INVOCATION" > "$run_dir/manifests/invocation.txt"
  {
    echo "generated_at=$(date -Is)"
    echo "work_root=$WORK_ROOT"
    echo "runs_root=$(resolved_runs_root)"
    echo "eval_envs=$EVAL_ENVS"
    echo "render_envs=$RENDER_ENVS"
    echo "smoke_envs=$SMOKE_ENVS"
    echo "collect_envs=$COLLECT_ENVS"
    echo "collect_motion_count=$COLLECT_MOTION_COUNT"
    echo "collect_batch_motions=$COLLECT_BATCH_MOTIONS"
    echo "collect_variants_per_motion=$COLLECT_VARIANTS_PER_MOTION"
    echo "collect_variant_offset=$COLLECT_VARIANT_OFFSET"
    echo "collect_randomization_profile=$COLLECT_RANDOMIZATION_PROFILE"
    echo "collect_seed=$COLLECT_SEED"
    echo "collect_motion_file=$COLLECT_MOTION_FILE"
    echo "collect_smpl_motion_file=$COLLECT_SMPL_MOTION_FILE"
    echo "collect_motion_manifest=$COLLECT_MOTION_MANIFEST"
    echo "collect_dataset_name=$COLLECT_DATASET_NAME"
    echo "offline_render_envs=$OFFLINE_RENDER_ENVS"
    echo "offline_frame_skip=$OFFLINE_FRAME_SKIP"
    echo "offline_width=$OFFLINE_WIDTH"
    echo "offline_height=$OFFLINE_HEIGHT"
    echo "offline_gl=$OFFLINE_GL"
    echo "seed=${RUN_SEED:-0}"
    echo "python=$(command -v python)"
  } > "$run_dir/manifests/run_config.txt"
}

phase_eval() {
  activate_env
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] || die "Checkpoint missing; run download-sample first."
  local run_dir checkpoint_path
  run_dir="$(new_run_dir eval)"
  write_run_manifest "$run_dir"
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  update_latest_run_pointer "$run_dir"
  log "Run directory: $run_dir"

  if (
    cd "$SONIC_DIR"
    python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_eval" \
      ++eval_callbacks=im_eval \
      ++run_eval_loop=False \
      "++num_envs=$EVAL_ENVS" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/logs/metrics.log"; then
    record_exit_code "$run_dir" eval 0
    mark_stage "$run_dir" eval.ok
  else
    local rc=$?
    record_exit_code "$run_dir" eval "$rc"
    return "$rc"
  fi
}

phase_render() {
  activate_env
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] || die "Checkpoint missing; run download-sample first."
  [[ -d "$SONIC_DIR/sample_data/robot_filtered" ]] || die "Sample motions missing; run download-sample first."
  local run_dir checkpoint_path
  run_dir="$(latest_run_dir_or_new render_isaac)"
  if [[ ! -f "$run_dir/manifests/run_config.txt" ]]; then
    write_run_manifest "$run_dir"
  fi
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  update_latest_run_pointer "$run_dir"

  if (
    cd "$SONIC_DIR"
    python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_render_isaac" \
      ++eval_callbacks=im_eval \
      ++run_eval_loop=False \
      "++num_envs=$RENDER_ENVS" \
      ++manager_env.config.render_results=True \
      "++manager_env.config.save_rendering_dir=$run_dir/videos" \
      ++manager_env.config.env_spacing=10.0 \
      "~manager_env/recorders=empty" "+manager_env/recorders=render" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/logs/render_isaac.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" render_isaac "$rc"
    return "$rc"
  fi

  find "$run_dir/videos" -maxdepth 1 -type f -name '*.mp4' -size +0c -print \
    | tee "$run_dir/manifests/rendered_videos.txt"
  if [[ ! -s "$run_dir/manifests/rendered_videos.txt" ]]; then
    record_exit_code "$run_dir" render_isaac 1
    die "Render finished without an MP4 artifact."
  fi
  record_exit_code "$run_dir" render_isaac 0
  mark_stage "$run_dir" render.ok
}

phase_collect_state_action() {
  activate_env
  prepare_dirs
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] \
    || die "Checkpoint missing; run download-sample first."
  [[ -f "$SONIC_DIR/gear_sonic/config/manager_env/recorders/minimal_state_action.yaml" ]] \
    || die "SONIC collector patch is not applied; follow sonic-repro-kit/README.md."
  [[ "$COLLECT_BATCH_MOTIONS" =~ ^[1-9][0-9]*$ ]] \
    || die "COLLECT_BATCH_MOTIONS must be a positive integer; found $COLLECT_BATCH_MOTIONS"
  [[ "$COLLECT_VARIANTS_PER_MOTION" =~ ^[1-9][0-9]*$ ]] \
    || die "COLLECT_VARIANTS_PER_MOTION must be a positive integer; found $COLLECT_VARIANTS_PER_MOTION"
  [[ "$COLLECT_SEED" =~ ^[0-9]+$ ]] \
    || die "COLLECT_SEED must be a non-negative integer; found $COLLECT_SEED"
  [[ "$COLLECT_RANDOMIZATION_PROFILE" == "startup" \
      || "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]] \
    || die "Unsupported COLLECT_RANDOMIZATION_PROFILE: $COLLECT_RANDOMIZATION_PROFILE"
  [[ "$COLLECT_DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "COLLECT_DATASET_NAME contains unsupported characters: $COLLECT_DATASET_NAME"
  [[ "$COLLECT_DATASET_NAME" != *.hdf5 ]] \
    || die "COLLECT_DATASET_NAME must not include the .hdf5 extension"

  local motion_path available_motion_count selected_motion_count batch_motion_count
  if [[ "$COLLECT_MOTION_FILE" = /* ]]; then
    motion_path="$COLLECT_MOTION_FILE"
  else
    motion_path="$SONIC_DIR/$COLLECT_MOTION_FILE"
  fi
  [[ -d "$motion_path" ]] || die "Motion directory missing: $motion_path"
  available_motion_count="$({
    find "$motion_path" -type f -name '*.pkl' ! -name 'metadata.pkl' -print
  } | wc -l)"
  (( available_motion_count > 0 )) || die "No motion PKLs found in $motion_path"
  if [[ "$COLLECT_MOTION_COUNT" == "all" ]]; then
    selected_motion_count="$available_motion_count"
  else
    [[ "$COLLECT_MOTION_COUNT" =~ ^[1-9][0-9]*$ ]] \
      || die "COLLECT_MOTION_COUNT must be 'all' or a positive integer"
    (( COLLECT_MOTION_COUNT <= available_motion_count )) \
      || die "Requested $COLLECT_MOTION_COUNT motions, only $available_motion_count are available"
    selected_motion_count="$COLLECT_MOTION_COUNT"
  fi
  batch_motion_count="$COLLECT_BATCH_MOTIONS"
  if (( batch_motion_count > selected_motion_count )); then
    batch_motion_count="$selected_motion_count"
  fi
  (( selected_motion_count % batch_motion_count == 0 )) \
    || die "Motion count $selected_motion_count must be divisible by batch size $batch_motion_count"

  local derived_envs
  derived_envs=$(( batch_motion_count * COLLECT_VARIANTS_PER_MOTION ))
  if [[ -n "$COLLECT_ENVS" && "$COLLECT_ENVS" != "$derived_envs" ]]; then
    die "COLLECT_ENVS=$COLLECT_ENVS must equal $batch_motion_count x $COLLECT_VARIANTS_PER_MOTION = $derived_envs"
  fi
  COLLECT_ENVS="$derived_envs"
  if [[ -z "$COLLECT_VARIANT_OFFSET" ]]; then
    if [[ "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]]; then
      COLLECT_VARIANT_OFFSET=4
    else
      COLLECT_VARIANT_OFFSET=0
    fi
  fi
  [[ "$COLLECT_VARIANT_OFFSET" =~ ^[0-9]+$ ]] \
    || die "COLLECT_VARIANT_OFFSET must be a non-negative integer"

  local free_gib
  free_gib="$(free_gib_for_path "$RUNS_ROOT")"
  (( free_gib >= BONES_MIN_STAGE_FREE_GIB )) \
    || die "Collection requires at least ${BONES_MIN_STAGE_FREE_GIB} GiB free; found ${free_gib} GiB"
  if [[ "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]]; then
    [[ "$COLLECT_VARIANTS_PER_MOTION" == "2" ]] \
      || die "initial_state_mild requires exactly 2 variants per motion"
    [[ "$COLLECT_VARIANT_OFFSET" == "4" ]] \
      || die "initial_state_mild requires COLLECT_VARIANT_OFFSET=4"
    [[ -f "$COLLECT_BASELINE_SUMMARY" ]] \
      || die "Set COLLECT_BASELINE_SUMMARY to the passed startup collection summary"
    [[ -f "$COLLECT_MOTION_MANIFEST" ]] \
      || die "initial_state_mild requires COLLECT_MOTION_MANIFEST"
    python "$SCRIPT_DIR/check_collection_gate.py" \
      --summary "$COLLECT_BASELINE_SUMMARY" \
      --overall-min 0.80 \
      --package-min 0.60 \
      --expected-canonical "$(( selected_motion_count * 4 ))" \
      --expected-package-count 8 \
      --expected-motion-manifest "$COLLECT_MOTION_MANIFEST"
  fi
  if [[ -n "$COLLECT_MOTION_MANIFEST" ]]; then
    [[ -f "$COLLECT_MOTION_MANIFEST" ]] \
      || die "Motion manifest missing: $COLLECT_MOTION_MANIFEST"
  fi

  local run_dir checkpoint_path runtime_manifest_path
  run_dir="$(new_run_dir collect_state_action)"
  if [[ -n "$COLLECT_MOTION_MANIFEST" ]]; then
    runtime_manifest_path="$run_dir/manifests/motion_manifest.jsonl"
    cp -- "$COLLECT_MOTION_MANIFEST" "$runtime_manifest_path"
    COLLECT_MOTION_MANIFEST="$runtime_manifest_path"
  else
    runtime_manifest_path=""
  fi
  RUN_SEED="$COLLECT_SEED" write_run_manifest "$run_dir"
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  update_latest_run_pointer "$run_dir"
  log "Collecting $selected_motion_count motions x $COLLECT_VARIANTS_PER_MOTION variants"
  log "Batch layout: $batch_motion_count motions x $COLLECT_VARIANTS_PER_MOTION variants = $COLLECT_ENVS envs"
  log "State-action collection run directory: $run_dir"

  local -a reset_randomization_args
  if [[ "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]]; then
    reset_randomization_args=(
      "++manager_env.commands.motion.randomize_eval_resets=True"
      "++manager_env.commands.motion.pose_range={x:[-0.025,0.025],y:[-0.025,0.025],z:[-0.005,0.005],roll:[-0.05,0.05],pitch:[-0.05,0.05],yaw:[-0.1,0.1]}"
      "++manager_env.commands.motion.velocity_range={x:[-0.25,0.25],y:[-0.25,0.25],z:[-0.1,0.1],roll:[-0.26,0.26],pitch:[-0.26,0.26],yaw:[-0.39,0.39]}"
      "++manager_env.commands.motion.joint_position_range=[-0.05,0.05]"
      "++manager_env.commands.motion.joint_velocity_range=[0.0,0.0]"
    )
  else
    reset_randomization_args=(
      "++manager_env.commands.motion.randomize_eval_resets=False"
    )
  fi

  if (
    cd "$SONIC_DIR"
    python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_collect_state_action" \
      "++eval_callbacks=[]" \
      ++run_eval_loop=True \
      ++run_once=True \
      ++run_all_motions_once=True \
      "++num_envs=$COLLECT_ENVS" \
      "++seed=$COLLECT_SEED" \
      ++use_encoder=g1 \
      ++manager_env.config.render_results=False \
      ++manager_env.config.enable_cameras=False \
      "~manager_env/recorders=empty" \
      "+manager_env/recorders=minimal_state_action" \
      "++manager_env.recorders.dataset_export_dir_path=$run_dir/data" \
      "++manager_env.recorders.dataset_filename=$COLLECT_DATASET_NAME" \
      "++manager_env.recorders.minimal_metadata.schema_output_path=$run_dir/manifests/state_action_schema.json" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      ++manager_env.commands.motion.use_paired_motions=False \
      ++manager_env.commands.motion.sample_unique_motions=False \
      ++manager_env.commands.motion.start_from_first_frame=True \
      ++manager_env.commands.motion.sample_from_n_initial_frames=null \
      "++manager_env.commands.motion.eval_motion_repeat=$COLLECT_VARIANTS_PER_MOTION" \
      ++manager_env.commands.motion.eval_require_full_batch=True \
      "++manager_env.commands.motion.eval_variant_offset=$COLLECT_VARIANT_OFFSET" \
      "++manager_env.commands.motion.collection_randomization_profile=$COLLECT_RANDOMIZATION_PROFILE" \
      "++manager_env.commands.motion.collection_seed=$COLLECT_SEED" \
      "++manager_env.commands.motion.collection_motion_manifest=$runtime_manifest_path" \
      "${reset_randomization_args[@]}" \
      ++manager_env.commands.motion.motion_lib_cfg.deterministic_motion_order=True \
      "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=$selected_motion_count" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$COLLECT_MOTION_FILE" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$COLLECT_SMPL_MOTION_FILE"
  ) 2>&1 | tee "$run_dir/logs/collect_state_action.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" collect_state_action "$rc"
    return "$rc"
  fi

  local -a verifier_manifest_args
  verifier_manifest_args=()
  if [[ -n "$runtime_manifest_path" ]]; then
    verifier_manifest_args=(--motion-manifest "$runtime_manifest_path")
  fi
  if python "$SCRIPT_DIR/verify_state_action.py" \
    --run-dir "$run_dir" \
    --dataset-name "$COLLECT_DATASET_NAME" \
    --expected-motion-count "$selected_motion_count" \
    --expected-variants-per-motion "$COLLECT_VARIANTS_PER_MOTION" \
    --variant-offset "$COLLECT_VARIANT_OFFSET" \
    --randomization-profile "$COLLECT_RANDOMIZATION_PROFILE" \
    "${verifier_manifest_args[@]}" \
    2>&1 | tee "$run_dir/logs/verify_state_action.log"; then
    free_gib="$(free_gib_for_path "$run_dir")"
    if (( free_gib < BONES_MIN_STAGE_FREE_GIB )); then
      record_exit_code "$run_dir" collect_state_action 1
      die "Collection finished with ${free_gib} GiB free; required >= ${BONES_MIN_STAGE_FREE_GIB} GiB"
    fi
    record_exit_code "$run_dir" collect_state_action 0
    mark_stage "$run_dir" collect_state_action.ok
  else
    local rc=$?
    record_exit_code "$run_dir" collect_state_action "$rc"
    return "$rc"
  fi
}

phase_verify_state_action() {
  activate_env
  prepare_dirs
  [[ -n "$COLLECT_RUN_DIR" ]] \
    || die "Set COLLECT_RUN_DIR to an existing collection run under $RUNS_ROOT"
  [[ "$COLLECT_MOTION_COUNT" =~ ^[1-9][0-9]*$ ]] \
    || die "Existing-run verification requires a numeric COLLECT_MOTION_COUNT"
  [[ "$COLLECT_VARIANTS_PER_MOTION" =~ ^[1-9][0-9]*$ ]] \
    || die "COLLECT_VARIANTS_PER_MOTION must be a positive integer"
  [[ "$COLLECT_RANDOMIZATION_PROFILE" == "startup" \
      || "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]] \
    || die "Unsupported COLLECT_RANDOMIZATION_PROFILE: $COLLECT_RANDOMIZATION_PROFILE"

  local run_dir variant_offset motion_manifest audit_stamp artifact log_path
  run_dir="$(validated_run_dir "$COLLECT_RUN_DIR")"
  [[ -d "$run_dir" ]] || die "Collection run does not exist: $run_dir"
  [[ -s "$run_dir/data/$COLLECT_DATASET_NAME.hdf5" ]] \
    || die "Collection dataset is missing: $run_dir/data/$COLLECT_DATASET_NAME.hdf5"
  [[ -s "$run_dir/manifests/state_action_schema.json" ]] \
    || die "Collection schema is missing: $run_dir/manifests/state_action_schema.json"
  [[ ! -e "$run_dir/markers/collect_state_action.ok" ]] \
    || die "Collection is already marked successful: $run_dir"

  variant_offset="$COLLECT_VARIANT_OFFSET"
  if [[ -z "$variant_offset" ]]; then
    if [[ "$COLLECT_RANDOMIZATION_PROFILE" == "initial_state_mild" ]]; then
      variant_offset=4
    else
      variant_offset=0
    fi
  fi
  [[ "$variant_offset" =~ ^[0-9]+$ ]] \
    || die "COLLECT_VARIANT_OFFSET must be a non-negative integer"

  motion_manifest="$COLLECT_MOTION_MANIFEST"
  if [[ -z "$motion_manifest" && -s "$run_dir/manifests/motion_manifest.jsonl" ]]; then
    motion_manifest="$run_dir/manifests/motion_manifest.jsonl"
  fi
  [[ -n "$motion_manifest" && -s "$motion_manifest" ]] \
    || die "Existing-run verification requires its motion manifest"

  audit_stamp="$(timestamp)"
  for artifact in \
    collection_summary.json \
    canonical_episode_index.json \
    additional_attempt_index.json; do
    if [[ -e "$run_dir/manifests/$artifact" ]]; then
      cp -- \
        "$run_dir/manifests/$artifact" \
        "$run_dir/manifests/$artifact.before_recheck_$audit_stamp"
    fi
  done
  log_path="$run_dir/logs/verify_state_action_recheck_$audit_stamp.log"

  if python "$SCRIPT_DIR/verify_state_action.py" \
    --run-dir "$run_dir" \
    --dataset-name "$COLLECT_DATASET_NAME" \
    --expected-motion-count "$COLLECT_MOTION_COUNT" \
    --expected-variants-per-motion "$COLLECT_VARIANTS_PER_MOTION" \
    --variant-offset "$variant_offset" \
    --randomization-profile "$COLLECT_RANDOMIZATION_PROFILE" \
    --motion-manifest "$motion_manifest" \
    2>&1 | tee "$log_path"; then
    record_exit_code "$run_dir" verify_state_action 0
    mark_stage "$run_dir" collect_state_action.ok
    log "Existing collection verified without rewriting HDF5: $run_dir"
  else
    local rc=$?
    record_exit_code "$run_dir" verify_state_action "$rc"
    return "$rc"
  fi
}

phase_bones_download_preflight() {
  prepare_dirs
  local free_gib
  free_gib="$(free_gib_for_path "$RUNS_ROOT")"
  (( free_gib >= BONES_MIN_DOWNLOAD_FREE_GIB )) \
    || die "BONES download requires at least ${BONES_MIN_DOWNLOAD_FREE_GIB} GiB free; found ${free_gib} GiB"
  log "BONES download disk gate passed: ${free_gib} GiB free."
  log "Accept the BONES-SEED license in the browser, then use hf auth login; never pass the token on the command line."
}

phase_prepare_bones_subset() {
  activate_env
  prepare_dirs
  [[ -n "${BONES_INGEST_RUN:-}" ]] \
    || die "Set BONES_INGEST_RUN to the existing ingest run under $RUNS_ROOT"
  local ingest_run free_gib
  ingest_run="$(validated_run_dir "$BONES_INGEST_RUN")"
  [[ -d "$ingest_run/data/source" ]] \
    || die "BONES source directory missing: $ingest_run/data/source"
  ensure_run_layout "$ingest_run"
  [[ ! -e "$ingest_run/manifests/bones_subset_report.json" \
      && ! -e "$ingest_run/logs/prepare_bones_subset.log" \
      && ! -e "$ingest_run/markers/prepare_bones_subset.ok" ]] \
    || die "Ingest run was already attempted; create a new BONES_INGEST_RUN"
  free_gib="$(free_gib_for_path "$ingest_run")"
  (( free_gib >= BONES_MIN_STAGE_FREE_GIB )) \
    || die "Subset preparation requires at least ${BONES_MIN_STAGE_FREE_GIB} GiB free; found ${free_gib} GiB"
  RUN_SEED="$COLLECT_SEED" write_run_manifest "$ingest_run"
  update_latest_run_pointer "$ingest_run"
  if python "$SCRIPT_DIR/prepare_bones_subset.py" \
    --ingest-run "$ingest_run" \
    --sonic-dir "$SONIC_DIR" \
    --seed "$COLLECT_SEED" \
    --candidate-per-package 40 \
    --final-per-package 32 \
    --min-free-gib "$BONES_MIN_STAGE_FREE_GIB" \
    2>&1 | tee "$ingest_run/logs/prepare_bones_subset.log"; then
    record_exit_code "$ingest_run" prepare_bones_subset 0
    mark_stage "$ingest_run" prepare_bones_subset.ok
  else
    local rc=$?
    record_exit_code "$ingest_run" prepare_bones_subset "$rc"
    return "$rc"
  fi
}

phase_dump_trajectory() {
  activate_env
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] \
    || die "Checkpoint missing; run download-sample first."
  [[ -d "$SONIC_DIR/sample_data/robot_filtered" ]] \
    || die "Sample motions missing; run download-sample first."

  local run_dir checkpoint_path
  run_dir="$(new_run_dir offline_render)"
  write_run_manifest "$run_dir"
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  update_latest_run_pointer "$run_dir"
  log "Offline trajectory run directory: $run_dir"

  if (
    cd "$SONIC_DIR"
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_dump" \
      "++eval_callbacks=[]" \
      ++run_eval_loop=True \
      ++run_once=True \
      "++num_envs=$OFFLINE_RENDER_ENVS" \
      ++manager_env.config.render_results=False \
      ++manager_env.config.enable_cameras=False \
      "++manager_env.config.render_frame_skip=$OFFLINE_FRAME_SKIP" \
      "++manager_env.recorders.trajectory._target_=offline_trajectory_recorder.BodyTrajectoryRecorderCfg" \
      "++manager_env.recorders.trajectory.save_path=$run_dir/data" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=$OFFLINE_RENDER_ENVS" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/logs/dump_trajectory.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" dump_trajectory "$rc"
    return "$rc"
  fi

  find "$run_dir/data" -maxdepth 1 -type f -name '*.trajectory.pkl' -size +0c -print \
    | tee "$run_dir/manifests/trajectory_files.txt"
  if [[ ! -s "$run_dir/manifests/trajectory_files.txt" ]]; then
    record_exit_code "$run_dir" dump_trajectory 1
    die "Trajectory dump finished without a non-empty .trajectory.pkl artifact."
  fi
  if grep -Eiq 'librtx\.scenedb|libomni\.hydra\.rtx|segmentation fault' \
    "$run_dir/logs/dump_trajectory.log"; then
    record_exit_code "$run_dir" dump_trajectory 1
    die "Trajectory dump log contains an RTX initialization failure or segmentation fault."
  fi
  record_exit_code "$run_dir" dump_trajectory 0
  mark_stage "$run_dir" trajectory_dump.ok
}

phase_render_mujoco() {
  activate_env
  local run_dir model_path
  run_dir="$(resolve_offline_run_dir)"
  model_path="$SONIC_DIR/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
  [[ -f "$model_path" ]] || die "MuJoCo G1 model missing: $model_path"

  local -a trajectories
  mapfile -d '' trajectories < <(
    find "$run_dir/data" -maxdepth 1 -type f -name '*.trajectory.pkl' -size +0c \
      -print0 | sort -z
  )
  (( ${#trajectories[@]} > 0 )) \
    || die "No non-empty trajectory files found in $run_dir/data"

  if (
    local trajectory stem output
    for trajectory in "${trajectories[@]}"; do
      stem="$(basename "$trajectory" .trajectory.pkl)"
      output="$run_dir/videos/$stem.mp4"
      python "$SCRIPT_DIR/render_mujoco_trajectory.py" \
        --trajectory "$trajectory" \
        --model "$model_path" \
        --output "$output" \
        --manifest-dir "$run_dir/manifests" \
        --width "$OFFLINE_WIDTH" \
        --height "$OFFLINE_HEIGHT" \
        --gl "$OFFLINE_GL" \
        --camera-distance "$OFFLINE_CAMERA_DISTANCE"
    done
  ) 2>&1 | tee "$run_dir/logs/render_mujoco.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" render_mujoco "$rc"
    return "$rc"
  fi

  find "$run_dir/videos" -maxdepth 1 -type f -name '*.mp4' -size +0c -print \
    | tee "$run_dir/manifests/rendered_videos.txt"
  if [[ ! -s "$run_dir/manifests/rendered_videos.txt" ]]; then
    record_exit_code "$run_dir" render_mujoco 1
    die "MuJoCo render finished without an MP4 artifact."
  fi
  record_exit_code "$run_dir" render_mujoco 0
  mark_stage "$run_dir" render.ok
}

phase_render_offline() {
  phase_dump_trajectory
  local run_dir
  run_dir="$(validated_run_dir "$(<"$STATE_DIR/latest_run_dir.txt")")"
  OFFLINE_RUN_DIR="$run_dir" phase_render_mujoco
}

json_manifest_value() {
  local manifest="$1" key="$2"
  python -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]; print(value)' \
    "$manifest" "$key"
}

prepare_action_mask_motion_dir() {
  local run_dir="$1" request="$2" motion_file expected_hash actual_hash motion_dir motion_link
  motion_file="$(json_manifest_value "$request" motion_file)"
  [[ -s "$motion_file" ]] || die "Action-mask motion PKL is missing: $motion_file"
  expected_hash="$(json_manifest_value "$request" motion_file_sha256)"
  actual_hash="$(sha256sum "$motion_file" | awk '{print $1}')"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || die "Action-mask motion PKL SHA256 no longer matches its request manifest"
  motion_dir="$run_dir/data/replay_motion"
  motion_link="$motion_dir/$(basename "$motion_file")"
  mkdir -p -- "$motion_dir"
  if [[ ! -e "$motion_link" && ! -L "$motion_link" ]]; then
    ln -s -- "$motion_file" "$motion_link"
  fi
  [[ "$(readlink -f -- "$motion_link")" == "$(readlink -f -- "$motion_file")" ]] \
    || die "Run-local motion link points at an unexpected file: $motion_link"
  printf '%s\n' "$motion_dir"
}

phase_capture_action_mask_source() {
  activate_env
  require_run_gpu_capacity
  [[ -n "$ACTION_MASK_RUN_DIR" ]] \
    || die "Set ACTION_MASK_RUN_DIR to the CVAE Action-mask evaluation run"
  local run_dir request motion_dir checkpoint_path seed output_dir
  run_dir="$(validated_run_dir "$ACTION_MASK_RUN_DIR")"
  request="$run_dir/manifests/action_mask_request.json"
  [[ -s "$request" ]] || die "Action-mask request is missing: $request"
  [[ -f "$SONIC_DIR/gear_sonic/eval_agent_trl.py" ]] \
    || die "SONIC evaluation entrypoint is missing"
  ensure_run_layout "$run_dir"
  motion_dir="$(prepare_action_mask_motion_dir "$run_dir" "$request")"
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  seed="$(json_manifest_value "$request" seed)"
  output_dir="$run_dir/data/source"
  mkdir -p -- "$output_dir"

  if (
    cd "$SONIC_DIR"
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_action_mask_source" \
      "++eval_callbacks=[]" \
      ++run_eval_loop=True \
      ++run_once=True \
      ++run_all_motions_once=False \
      ++num_envs=1 \
      "++seed=$seed" \
      ++use_encoder=g1 \
      ++manager_env.config.render_results=False \
      ++manager_env.config.enable_cameras=False \
      "~manager_env/recorders=empty" \
      ++manager_env.recorders.dataset_export_mode=0 \
      "++manager_env.recorders.trajectory._target_=action_replay_recorder.ActionReplayTrajectoryRecorderCfg" \
      "++manager_env.recorders.trajectory.save_path=$output_dir" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      ++manager_env.terminations.anchor_pos=null \
      ++manager_env.terminations.anchor_ori_full=null \
      ++manager_env.terminations.ee_body_pos=null \
      ++manager_env.terminations.foot_pos_xyz=null \
      ++manager_env.events.physics_material=null \
      ++manager_env.events.add_joint_default_pos=null \
      ++manager_env.events.base_com=null \
      ++manager_env.events.push_robot=null \
      ++manager_env.events.randomize_rigid_body_mass=null \
      ++manager_env.commands.motion.use_paired_motions=False \
      ++manager_env.commands.motion.sample_unique_motions=False \
      ++manager_env.commands.motion.start_from_first_frame=True \
      ++manager_env.commands.motion.sample_from_n_initial_frames=null \
      ++manager_env.commands.motion.eval_motion_repeat=1 \
      ++manager_env.commands.motion.eval_require_full_batch=True \
      ++manager_env.commands.motion.randomize_eval_resets=False \
      ++manager_env.commands.motion.motion_lib_cfg.deterministic_motion_order=True \
      ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1 \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$motion_dir" \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros
  ) 2>&1 | tee "$run_dir/logs/action_mask_source.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" action_mask_source "$rc"
    return "$rc"
  fi
  [[ -s "$output_dir/000000.replay.npz" \
      && -s "$output_dir/000000.trajectory.pkl" ]] \
    || die "Source capture did not produce both replay NPZ and trajectory PKL"
  record_exit_code "$run_dir" action_mask_source 0
  mark_stage "$run_dir" action_mask_source.ok
}

phase_replay_action_mask() {
  activate_env
  require_run_gpu_capacity
  [[ -n "$ACTION_MASK_RUN_DIR" ]] \
    || die "Set ACTION_MASK_RUN_DIR to the CVAE Action-mask evaluation run"
  local run_dir request source_request motion_dir checkpoint_path seed actions_file actions_hash
  local expected_actions_hash num_envs output_dir
  run_dir="$(validated_run_dir "$ACTION_MASK_RUN_DIR")"
  request="$run_dir/manifests/action_replay_request.json"
  source_request="$run_dir/manifests/action_mask_request.json"
  [[ -s "$request" && -s "$source_request" ]] \
    || die "Action replay request manifests are incomplete in $run_dir"
  motion_dir="$(prepare_action_mask_motion_dir "$run_dir" "$request")"
  checkpoint_path="$(prepare_eval_checkpoint "$run_dir")"
  seed="$(json_manifest_value "$source_request" seed)"
  actions_file="$(json_manifest_value "$request" raw_actions_file)"
  expected_actions_hash="$(json_manifest_value "$request" raw_actions_sha256)"
  num_envs="$(json_manifest_value "$request" num_envs)"
  [[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || die "Invalid Action replay environment count: $num_envs"
  [[ -s "$actions_file" ]] || die "External Action replay file is missing: $actions_file"
  actions_hash="$(sha256sum "$actions_file" | awk '{print $1}')"
  [[ "$actions_hash" == "$expected_actions_hash" ]] \
    || die "External Action replay SHA256 no longer matches its request manifest"
  output_dir="$run_dir/data/replay"
  mkdir -p -- "$output_dir"

  if (
    cd "$SONIC_DIR"
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python gear_sonic/eval_agent_trl.py \
      "+checkpoint=$checkpoint_path" \
      +headless=True \
      "hydra.run.dir=$run_dir/manifests/hydra_action_mask_replay" \
      "++eval_callbacks=[]" \
      ++run_eval_loop=True \
      ++run_once=False \
      "++external_action_replay_path=$actions_file" \
      "++num_envs=$num_envs" \
      "++seed=$seed" \
      ++use_encoder=g1 \
      ++manager_env.config.render_results=False \
      ++manager_env.config.enable_cameras=False \
      "~manager_env/recorders=empty" \
      ++manager_env.recorders.dataset_export_mode=0 \
      "++manager_env.recorders.trajectory._target_=action_replay_recorder.ActionReplayTrajectoryRecorderCfg" \
      "++manager_env.recorders.trajectory.save_path=$output_dir" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      ++manager_env.terminations.anchor_pos=null \
      ++manager_env.terminations.anchor_ori_full=null \
      ++manager_env.terminations.ee_body_pos=null \
      ++manager_env.terminations.foot_pos_xyz=null \
      ++manager_env.events.physics_material=null \
      ++manager_env.events.add_joint_default_pos=null \
      ++manager_env.events.base_com=null \
      ++manager_env.events.push_robot=null \
      ++manager_env.events.randomize_rigid_body_mass=null \
      ++manager_env.commands.motion.use_paired_motions=False \
      ++manager_env.commands.motion.sample_unique_motions=False \
      ++manager_env.commands.motion.start_from_first_frame=True \
      ++manager_env.commands.motion.sample_from_n_initial_frames=null \
      "++manager_env.commands.motion.eval_motion_repeat=$num_envs" \
      ++manager_env.commands.motion.eval_require_full_batch=True \
      ++manager_env.commands.motion.randomize_eval_resets=False \
      ++manager_env.commands.motion.motion_lib_cfg.deterministic_motion_order=True \
      ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1 \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$motion_dir" \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros
  ) 2>&1 | tee "$run_dir/logs/action_mask_replay.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" action_mask_replay "$rc"
    return "$rc"
  fi
  local replay_count trajectory_count
  replay_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.replay.npz' -size +0c | wc -l)"
  trajectory_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.trajectory.pkl' -size +0c | wc -l)"
  [[ "$replay_count" == "$num_envs" && "$trajectory_count" == "$num_envs" ]] \
    || die "Expected $num_envs replay artifacts; found NPZ=$replay_count PKL=$trajectory_count"
  record_exit_code "$run_dir" action_mask_replay 0
  mark_stage "$run_dir" action_mask_replay.ok
}

phase_render_action_mask() {
  activate_env
  [[ -n "$ACTION_MASK_RUN_DIR" ]] \
    || die "Set ACTION_MASK_RUN_DIR to the CVAE Action-mask evaluation run"
  local run_dir request model_path render_mode
  run_dir="$(validated_run_dir "$ACTION_MASK_RUN_DIR")"
  request="$run_dir/manifests/action_mask_request.json"
  [[ -s "$request" ]] || die "Action-mask request is missing: $request"
  model_path="$SONIC_DIR/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
  [[ -f "$model_path" ]] || die "MuJoCo G1 model missing: $model_path"
  render_mode="$(json_manifest_value "$request" render_mode)"
  if python "$SCRIPT_DIR/render_action_mask_comparison.py" \
    --run-dir "$run_dir" \
    --model "$model_path" \
    --render-mode "$render_mode" \
    --width "$ACTION_MASK_WIDTH" \
    --height "$ACTION_MASK_HEIGHT" \
    --gl "$ACTION_MASK_GL" \
    --camera-distance "$ACTION_MASK_CAMERA_DISTANCE" \
    2>&1 | tee "$run_dir/logs/action_mask_render.log"; then
    :
  else
    local rc=$?
    record_exit_code "$run_dir" action_mask_render "$rc"
    return "$rc"
  fi
  [[ -s "$run_dir/videos/sonic_source.mp4" \
      && -s "$run_dir/videos/original_action_replay.mp4" \
      && -s "$run_dir/videos/all_action_masks_grid.mp4" ]] \
    || die "Action-mask renderer did not produce the mandatory MP4 files"
  record_exit_code "$run_dir" action_mask_render 0
  mark_stage "$run_dir" action_mask_render.ok
}

phase_smoke_train() {
  activate_env
  require_run_gpu_capacity
  [[ -d "$SONIC_DIR/sample_data/robot_filtered" ]] || die "Sample motions missing; run download-sample first."
  local run_dir
  run_dir="$(latest_run_dir_or_new smoke_train)"
  if [[ ! -f "$run_dir/manifests/run_config.txt" ]]; then
    write_run_manifest "$run_dir"
  fi
  update_latest_run_pointer "$run_dir"

  if (
    cd "$SONIC_DIR"
    WANDB_MODE=offline python gear_sonic/train_agent_trl.py \
      +exp=manager/universal_token/all_modes/sonic_release \
      "++experiment_dir=$run_dir/checkpoints" \
      "num_envs=$SMOKE_ENVS" \
      headless=True \
      ++algo.config.num_learning_iterations=5 \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/logs/train_smoke.log"; then
    record_exit_code "$run_dir" smoke_train 0
    mark_stage "$run_dir" smoke_train.ok
  else
    local rc=$?
    record_exit_code "$run_dir" smoke_train "$rc"
    return "$rc"
  fi
}

phase_verify_minimal() {
  prepare_dirs
  need_command python3
  python3 "$SCRIPT_DIR/verify_minimal.py" \
    --work-root "$WORK_ROOT" \
    --runs-root "$RUNS_ROOT" \
    --expected-sonic-commit "$SONIC_COMMIT" \
    --expected-isaaclab-commit "$ISAACLAB_COMMIT"
}

phase_status() {
  local latest_raw latest_resolved
  phase_preflight
  echo
  echo "Pinned versions:"
  echo "  Python:    $PYTHON_VERSION"
  echo "  Isaac Sim: $ISAACSIM_VERSION"
  echo "  Isaac Lab: $ISAACLAB_REF ($ISAACLAB_COMMIT)"
  echo "  SONIC:     $SONIC_COMMIT"
  echo "  Runs root: $(resolved_runs_root)"
  echo
  if [[ -f "$STATE_DIR/latest_run_dir.txt" ]]; then
    latest_raw="$(<"$STATE_DIR/latest_run_dir.txt")"
    if latest_resolved="$(validated_run_dir "$latest_raw" 2>/dev/null)" \
      && [[ -d "$latest_resolved" ]]; then
      echo "Latest run: $latest_resolved"
    else
      echo "Latest run: INVALID OR STALE ($latest_raw)"
    fi
  else
    echo "Latest run: none"
  fi
}

usage() {
  cat <<'EOF'
Usage: ./sonic_repro.sh PHASE

Phases:
  preflight       Collect hardware, OS, disk, GPU, and tool state.
  install-system  Install Git LFS and build prerequisites with apt.
  install-env     Create uv env and install pinned Isaac Sim/Lab/PyTorch.
  verify-isaac    Start an empty headless Isaac Lab scene (Ctrl+C after success).
  clone-sonic     Clone pinned SONIC and install training dependencies.
  download-sample Download official checkpoint and sample motions; run checks.
  eval            Run sample metrics; EVAL_ENVS defaults to 32.
  render          Render through Isaac Sim RTX; RENDER_ENVS defaults to 8.
  dump-trajectory Record one headless episode without enabling cameras or RTX.
  render-mujoco   Render the latest trajectory through MuJoCo OSMesa.
  render-offline  Dump a fresh trajectory and render it through MuJoCo OSMesa.
  capture-action-mask-source
                  Capture one deterministic SONIC source trajectory for CVAE replay.
  replay-action-mask
                  Execute original and CVAE-completed raw Actions in Isaac physics.
  render-action-mask
                  Render synchronized common-camera Action-mask comparison MP4s.
  collect-state-action
                  Record and verify minimal (s_t, g_t, a_t, s_t+1) HDF5 data.
  verify-state-action
                  Revalidate an existing collection without rewriting its HDF5 data.
  bones-download-preflight
                  Require 70 GiB free before the manual gated BONES download.
  prepare-bones-subset
                  Select, extract, convert, filter, and audit the 256-motion subset.
  smoke-train     Run five offline training iterations; SMOKE_ENVS defaults to 16.
  verify-minimal  Audit commits, logs, success markers, dependencies, and MP4 output.
  setup-minimal   Install dependencies and sample data; does not run Isaac Sim.
  status          Produce a fresh status report and show pinned revisions.

Environment overrides:
  SONIC_WORK_ROOT, SONIC_RUNS_ROOT, EVAL_ENVS, RENDER_ENVS, SMOKE_ENVS,
  COLLECT_MOTION_COUNT, COLLECT_BATCH_MOTIONS, COLLECT_VARIANTS_PER_MOTION,
  COLLECT_ENVS, COLLECT_RANDOMIZATION_PROFILE, COLLECT_SEED, COLLECT_VARIANT_OFFSET,
  COLLECT_MOTION_FILE, COLLECT_SMPL_MOTION_FILE, COLLECT_MOTION_MANIFEST,
  COLLECT_BASELINE_SUMMARY, COLLECT_DATASET_NAME, COLLECT_RUN_DIR, BONES_INGEST_RUN,
  OFFLINE_RENDER_ENVS, OFFLINE_FRAME_SKIP, OFFLINE_WIDTH, OFFLINE_HEIGHT,
  OFFLINE_GL, OFFLINE_CAMERA_DISTANCE, OFFLINE_RUN_DIR,
  ACTION_MASK_RUN_DIR, ACTION_MASK_GL, ACTION_MASK_WIDTH, ACTION_MASK_HEIGHT,
  ACTION_MASK_CAMERA_DISTANCE,
  MAX_RUN_GPU_USED_MIB, SONIC_COMMIT
EOF
}

main() {
  local phase="${1:-}"
  case "$phase" in
    preflight) phase_preflight ;;
    install-system) phase_install_system ;;
    install-env) phase_install_env ;;
    verify-isaac) phase_verify_isaac ;;
    clone-sonic) phase_clone_sonic ;;
    download-sample) phase_download_sample ;;
    eval) phase_eval ;;
    render) phase_render ;;
    dump-trajectory) phase_dump_trajectory ;;
    render-mujoco) phase_render_mujoco ;;
    render-offline) phase_render_offline ;;
    capture-action-mask-source) phase_capture_action_mask_source ;;
    replay-action-mask) phase_replay_action_mask ;;
    render-action-mask) phase_render_action_mask ;;
    collect-state-action) phase_collect_state_action ;;
    verify-state-action) phase_verify_state_action ;;
    bones-download-preflight) phase_bones_download_preflight ;;
    prepare-bones-subset) phase_prepare_bones_subset ;;
    smoke-train) phase_smoke_train ;;
    verify-minimal) phase_verify_minimal ;;
    status) phase_status ;;
    setup-minimal)
      phase_preflight
      phase_install_system
      phase_install_env
      phase_clone_sonic
      phase_download_sample
      ;;
    -h|--help|help) usage ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"
