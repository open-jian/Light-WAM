#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-0,1,2,3}"
export PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-4}"
export PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-32}"
export PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
export RUN_TEXT="${RUN_TEXT:-true}"
export RUN_VIDEO="${RUN_VIDEO:-false}"
exec "${SCRIPT_DIR}/precompute_robotwin.sh" "$@"
