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

prepare_dirs() {
  mkdir -p "$WORK_ROOT" "$STATE_DIR"
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
  local dir="$SONIC_DIR/runs/phase1_$(timestamp)"
  mkdir -p "$dir/renders"
  printf '%s\n' "$dir"
}

write_run_manifest() {
  local run_dir="$1"
  nvidia-smi > "$run_dir/nvidia-smi.txt"
  git -C "$SONIC_DIR" status --short > "$run_dir/sonic_status.txt"
  git -C "$SONIC_DIR" rev-parse HEAD > "$run_dir/sonic_commit.txt"
  git -C "$ISAACLAB_DIR" rev-parse HEAD > "$run_dir/isaaclab_commit.txt"
  uv pip freeze > "$run_dir/environment_freeze.txt"
  {
    echo "generated_at=$(date -Is)"
    echo "work_root=$WORK_ROOT"
    echo "eval_envs=$EVAL_ENVS"
    echo "render_envs=$RENDER_ENVS"
    echo "smoke_envs=$SMOKE_ENVS"
    echo "python=$(command -v python)"
  } > "$run_dir/run_config.txt"
}

phase_eval() {
  activate_env
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] || die "Checkpoint missing; run download-sample first."
  local run_dir
  run_dir="$(new_run_dir)"
  write_run_manifest "$run_dir"
  printf '%s\n' "$run_dir" > "$STATE_DIR/latest_run_dir.txt"
  log "Run directory: $run_dir"

  (
    cd "$SONIC_DIR"
    python gear_sonic/eval_agent_trl.py \
      +checkpoint=sonic_release/last.pt \
      +headless=True \
      ++eval_callbacks=im_eval \
      ++run_eval_loop=False \
      "++num_envs=$EVAL_ENVS" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "+manager_env/terminations=tracking/eval" \
      "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/metrics.log"
  touch "$run_dir/eval.ok"
}

phase_render() {
  activate_env
  require_run_gpu_capacity
  [[ -f "$SONIC_DIR/sonic_release/last.pt" ]] || die "Checkpoint missing; run download-sample first."
  [[ -d "$SONIC_DIR/sample_data/robot_filtered" ]] || die "Sample motions missing; run download-sample first."
  local run_dir
  if [[ -f "$STATE_DIR/latest_run_dir.txt" ]]; then
    run_dir="$(<"$STATE_DIR/latest_run_dir.txt")"
  else
    run_dir="$(new_run_dir)"
    write_run_manifest "$run_dir"
    printf '%s\n' "$run_dir" > "$STATE_DIR/latest_run_dir.txt"
  fi
  mkdir -p "$run_dir/renders"

  (
    cd "$SONIC_DIR"
    python gear_sonic/eval_agent_trl.py \
      +checkpoint=sonic_release/last.pt \
      +headless=True \
      ++eval_callbacks=im_eval \
      ++run_eval_loop=False \
      "++num_envs=$RENDER_ENVS" \
      ++manager_env.config.render_results=True \
      "++manager_env.config.save_rendering_dir=$run_dir/renders" \
      ++manager_env.config.env_spacing=10.0 \
      "~manager_env/recorders=empty" "+manager_env/recorders=render" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/render.log"

  find "$run_dir/renders" -maxdepth 1 -type f -name '*.mp4' -print | tee "$run_dir/rendered_videos.txt"
  test -s "$run_dir/rendered_videos.txt" || die "Render finished without an MP4 artifact."
  touch "$run_dir/render.ok"
}

phase_smoke_train() {
  activate_env
  require_run_gpu_capacity
  [[ -d "$SONIC_DIR/sample_data/robot_filtered" ]] || die "Sample motions missing; run download-sample first."
  local run_dir
  if [[ -f "$STATE_DIR/latest_run_dir.txt" ]]; then
    run_dir="$(<"$STATE_DIR/latest_run_dir.txt")"
  else
    run_dir="$(new_run_dir)"
    write_run_manifest "$run_dir"
    printf '%s\n' "$run_dir" > "$STATE_DIR/latest_run_dir.txt"
  fi

  (
    cd "$SONIC_DIR"
    WANDB_MODE=offline python gear_sonic/train_agent_trl.py \
      +exp=manager/universal_token/all_modes/sonic_release \
      "num_envs=$SMOKE_ENVS" \
      headless=True \
      ++algo.config.num_learning_iterations=5 \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
  ) 2>&1 | tee "$run_dir/train_smoke.log"
  touch "$run_dir/smoke_train.ok"
}

phase_verify_minimal() {
  prepare_dirs
  need_command python3
  python3 "$SCRIPT_DIR/verify_minimal.py" \
    --work-root "$WORK_ROOT" \
    --expected-sonic-commit "$SONIC_COMMIT" \
    --expected-isaaclab-commit "$ISAACLAB_COMMIT"
}

phase_status() {
  phase_preflight
  echo
  echo "Pinned versions:"
  echo "  Python:    $PYTHON_VERSION"
  echo "  Isaac Sim: $ISAACSIM_VERSION"
  echo "  Isaac Lab: $ISAACLAB_REF ($ISAACLAB_COMMIT)"
  echo "  SONIC:     $SONIC_COMMIT"
  echo
  [[ -f "$STATE_DIR/latest_run_dir.txt" ]] \
    && echo "Latest run: $(<"$STATE_DIR/latest_run_dir.txt")" \
    || echo "Latest run: none"
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
  render          Render sample MP4 files; RENDER_ENVS defaults to 8.
  smoke-train     Run five offline training iterations; SMOKE_ENVS defaults to 16.
  verify-minimal  Audit commits, logs, success markers, dependencies, and MP4 output.
  setup-minimal   Install dependencies and sample data; does not run Isaac Sim.
  status          Produce a fresh status report and show pinned revisions.

Environment overrides:
  SONIC_WORK_ROOT, EVAL_ENVS, RENDER_ENVS, SMOKE_ENVS,
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
