#!/usr/bin/env bash
set -Eeuo pipefail
# Pull-only receiver: run locally; no server-to-workstation SSH is required.
# Usage: receive_mlp_cvae11_128_final_ghost4.sh SSH_TARGET REMOTE_STATE_DIR LOCAL_REPLAY_DIR [--poll-seconds 60] [--max-wait-seconds 259200]
# Optional --remote-python /absolute/venv/bin/python overrides stage-recorded Python for remote frame verification when local ffprobe is absent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${GHOST_RECEIVER_PYTHON:-python3}" "$SCRIPT_DIR/mlp_cvae11_128_ghost_stage.py" receive "$@"
