#!/usr/bin/env bash

set -Eeuo pipefail

CHECKPOINT="${1:-}"
MOTION_KEY="${2:-}"
STEPS="${3:-${DAGGER_RENDER_STEPS:-192}}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 <mlp_round_checkpoint.pt> [motion_key] [steps]" >&2
  exit 2
fi

DAGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_REPO_ROOT="${SONIC_REPO_ROOT:-$HOME/bly/sonic-repro/GR00T-WholeBodyControl}"
SONIC_RUNS_ROOT="${SONIC_RUNS_ROOT:-$HOME/bly/runs}"
DAGGER_MOTION_FILE="${DAGGER_MOTION_FILE:-$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered}"
PYTHON="${PYTHON:-$HOME/bly/sonic-repro/.venv-sonic/bin/python}"
SONIC_TEACHER_CHECKPOINT="${SONIC_TEACHER_CHECKPOINT:-$SONIC_REPO_ROOT/sonic_release/last.pt}"

export SONIC_REPO_ROOT SONIC_RUNS_ROOT DAGGER_MOTION_FILE SONIC_TEACHER_CHECKPOINT
export DAGGER_RENDER_STEPS="$STEPS"

if [[ -z "$MOTION_KEY" ]]; then
  TRAIN_RUN="$(cd "$(dirname "$(realpath "$CHECKPOINT")")/.." && pwd)"
  TRAIN_MANIFEST="$TRAIN_RUN/manifests/dagger_mlp_manifest.json"
  [[ -f "$TRAIN_MANIFEST" ]] || TRAIN_MANIFEST="$TRAIN_RUN/manifests/dagger_mlp_manifest.partial.json"
  if [[ -f "$TRAIN_MANIFEST" ]]; then
    MOTION_KEY="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); keys=d.get("motion_keys_loaded") or d.get("motion", {}).get("loaded_keys") or []; print(keys[0] if keys else "")' "$TRAIN_MANIFEST")"
  fi
fi
if [[ -z "$MOTION_KEY" ]]; then
  echo "No motion key; pass it as argument 2." >&2
  exit 2
fi

echo "[compare] motion_key=$MOTION_KEY steps=$STEPS"

bash "$DAGGER_ROOT/scripts/render_mlp_policy.sh" "$CHECKPOINT" "$MOTION_KEY" "$STEPS"
MLP_RUN="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
MLP_VIDEO="$MLP_RUN/videos/mlp_policy.mp4"
test -s "$MLP_VIDEO" || { echo "MLP video is missing: $MLP_VIDEO" >&2; exit 1; }

export DAGGER_RENDER_REFERENCE_RUN="$MLP_RUN"
bash "$DAGGER_ROOT/scripts/render_sonic_policy.sh" "$MOTION_KEY" "$STEPS"
SONIC_RUN="$(ls -dt "$SONIC_RUNS_ROOT"/dagger_mlp_render_* 2>/dev/null | head -n 1)"
SONIC_VIDEO="$SONIC_RUN/videos/sonic_policy.mp4"
test -s "$SONIC_VIDEO" || { echo "SONIC video is missing: $SONIC_VIDEO" >&2; exit 1; }

FFMPEG="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || command -v ffmpeg || true)"
if [[ -z "$FFMPEG" ]]; then
  echo "ffmpeg is unavailable; separate videos remain: $SONIC_VIDEO $MLP_VIDEO" >&2
  exit 1
fi

COMPARE_DIR="$MLP_RUN/compare"
mkdir -p "$COMPARE_DIR" "$MLP_RUN/markers"
COMPARE_VIDEO="$COMPARE_DIR/sonic_left_mlp_right.mp4"

"$FFMPEG" -y \
  -i "$SONIC_VIDEO" \
  -i "$MLP_VIDEO" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -shortest \
  "$COMPARE_VIDEO"

test -s "$COMPARE_VIDEO" || { echo "Compare video was not produced." >&2; exit 1; }
printf 'PASS\n' > "$MLP_RUN/markers/compare.ok"

"$PYTHON" -c 'import json,sys; payload={"sonic_video": sys.argv[1], "mlp_video": sys.argv[2], "compare_video": sys.argv[3], "layout": "sonic_left_mlp_right"}; open(sys.argv[4], "w").write(json.dumps(payload, indent=2))' \
  "$SONIC_VIDEO" "$MLP_VIDEO" "$COMPARE_VIDEO" "$COMPARE_DIR/compare_manifest.json"

echo "[compare] sonic_video=$SONIC_VIDEO"
echo "[compare] mlp_video=$MLP_VIDEO"
echo "[compare] compare_video=$COMPARE_VIDEO"
