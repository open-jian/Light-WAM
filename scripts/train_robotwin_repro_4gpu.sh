#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"
export TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-64}"
export MAX_STEPS="${MAX_STEPS:-460000}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "${SCRIPT_DIR}/train_robotwin.sh" "$@"
