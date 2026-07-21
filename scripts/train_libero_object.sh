#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DATE="${RUN_DATE:-$(TZ=America/Chicago date +%Y-%m-%d_%H-%M-%S)}"
SUITE_NAME="${SUITE_NAME:-libero_object}" \
TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}" \
RUN_TAG="${RUN_TAG:-raw-frame-as-mem}" \
WANDB_PROJECT="${WANDB_PROJECT:-LIBERO-Obj}" \
WANDB_NAME="${WANDB_NAME:-raw-frame-as-mem_${RUN_DATE}}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29578}" \
DATASET_DIR="${DATASET_DIR:-./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot}" \
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-./data/latent_cache_Wan2.1-T2V-1.3B/libero_object_2cam224}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
MAX_STEPS="${MAX_STEPS:-5000}" \
SAVE_EVERY="${SAVE_EVERY:-1000}" \
WARMUP_STEPS="${WARMUP_STEPS:-1000}" \
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-null}" \
TRAIN_VISUALIZATION_ENABLED="${TRAIN_VISUALIZATION_ENABLED:-false}" \
bash "${SCRIPT_DIR}/train_libero_core.sh" "$@"
