#!/usr/bin/env bash
set -Eeuo pipefail

C_JOB_REF="${1:?C job ref required}"
C_RUN="${2:?C run required}"
MIG="${3:?migration root required}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$CODE/sonic-repro/.venv-sonic/bin/python}"
exec "$PYTHON" "$CODE/dagger/scripts/mlp_cvae11_128_pipeline.py" "$C_JOB_REF" "$C_RUN" "$MIG"
