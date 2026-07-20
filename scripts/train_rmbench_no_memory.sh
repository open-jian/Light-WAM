#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-rmbench_no_memory_3cam_384_1e-4}"
RMBENCH_DATASET_DIR="${RMBENCH_DATASET_DIR:-${REPO_ROOT}/data/rmbench/lerobot_no_memory}"
RMBENCH_NORM_STATS="${RMBENCH_NORM_STATS:-${REPO_ROOT}/data/rmbench/dataset_stats.json}"
RMBENCH_TEXT_CACHE_DIR="${RMBENCH_TEXT_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/rmbench_no_memory}"
RMBENCH_LATENT_CACHE_DIR="${RMBENCH_LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B/rmbench_no_memory_3cam384_sharded}"
RMBENCH_USE_LATENT_CACHE="${RMBENCH_USE_LATENT_CACHE:-true}"

for override in "$@"; do
  key="${override%%=*}"
  if [[ "${key}" == *memory* || "${key}" == *history* ]]; then
    echo "[rmbench-no-memory] forbidden override: ${override}" >&2
    exit 2
  fi
done

if [[ ! -d "${RMBENCH_DATASET_DIR}" ]]; then
  echo "[rmbench-no-memory] missing converted dataset: ${RMBENCH_DATASET_DIR}" >&2
  exit 2
fi

if [[ ! -f "${RMBENCH_NORM_STATS}" ]]; then
  echo "[rmbench-no-memory] computing normalization stats: ${RMBENCH_NORM_STATS}"
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}" \
    python "${SCRIPT_DIR}/compute_lerobot_dataset_stats.py" \
      "task=${TASK_NAME}" \
      "output_dir=${REPO_ROOT}/runs/rmbench_no_memory_stats" \
      "+output_stats_path=${RMBENCH_NORM_STATS}" \
      "data.train.dataset_dirs=['${RMBENCH_DATASET_DIR}']"
fi

if [[ "${RMBENCH_USE_LATENT_CACHE}" == "true" && ! -f "${RMBENCH_LATENT_CACHE_DIR}/index.pt" ]]; then
  echo "[rmbench-no-memory] missing latent cache index: ${RMBENCH_LATENT_CACHE_DIR}/index.pt" >&2
  echo "Run scripts/precompute_rmbench_no_memory.sh before training." >&2
  exit 2
fi

export TASK_NAME
export RUN_TAG="${RUN_TAG:-lightwam_rmbench_no_memory_3cam384_1e-4}"
export ROBOTWIN_TRAIN_DIR="${RMBENCH_DATASET_DIR}"
export ROBOTWIN_VAL_DIR="${RMBENCH_DATASET_DIR}"
export ROBOTWIN_NORM_STATS="${RMBENCH_NORM_STATS}"
export TEXT_EMBED_CACHE_DIR="${RMBENCH_TEXT_CACHE_DIR}"
export LATENT_CACHE_DIR="${RMBENCH_LATENT_CACHE_DIR}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACC="${GRAD_ACC:-1}"

exec "${SCRIPT_DIR}/train_robotwin.sh" \
  "data.train.use_latent_cache=${RMBENCH_USE_LATENT_CACHE}" \
  "data.train.latent_cache_dir=${RMBENCH_LATENT_CACHE_DIR}" \
  "$@"
