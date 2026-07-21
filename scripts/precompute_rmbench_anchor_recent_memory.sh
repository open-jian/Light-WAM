#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIGHTWAM_ENV_BIN="${LIGHTWAM_ENV_BIN:-/home/jian/.local/share/mamba/envs/fastwam/bin}"
if [[ ! -x "${LIGHTWAM_ENV_BIN}/python" || ! -x "${LIGHTWAM_ENV_BIN}/torchrun" ]]; then
  echo "[rmbench-anchor-recent-memory] invalid LIGHTWAM_ENV_BIN: ${LIGHTWAM_ENV_BIN}" >&2
  exit 2
fi
export PATH="${LIGHTWAM_ENV_BIN}:${PATH}"

EXPERIMENT="rmbench/put_back_block_anchor_recent_memory_5k"
DATASET_DIR="${RMBENCH_DATASET_DIR:-${REPO_ROOT}/data/rmbench/tasks/put_back_block}"
TEXT_CACHE_DIR="${RMBENCH_TEXT_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/rmbench}"
TRAIN_CACHE_DIR="${RMBENCH_TRAIN_LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B/rmbench_tasks/put_back_block_anchor_recent_memory_episode_packed_train}"
VAL_CACHE_DIR="${RMBENCH_VAL_LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B/rmbench_tasks/put_back_block_anchor_recent_memory_episode_packed_val}"
GPU_IDS="${PRECOMPUTE_GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-4}"
BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-8}"
NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
OVERWRITE="${OVERWRITE:-false}"
RUN_TEXT="${RUN_TEXT:-true}"
RUN_VIDEO="${RUN_VIDEO:-true}"

case "${OVERWRITE}" in
  true) PRECOMPUTE_RESUME=false ;;
  false) PRECOMPUTE_RESUME=true ;;
  *)
    echo "[rmbench-anchor-recent-memory] OVERWRITE must be true or false" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[rmbench-anchor-recent-memory] missing task dataset: ${DATASET_DIR}" >&2
  exit 2
fi
if [[ "${TRAIN_CACHE_DIR}" == "${VAL_CACHE_DIR}" ]]; then
  echo "[rmbench-anchor-recent-memory] train and val cache directories must differ" >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "[rmbench-anchor-recent-memory] GPU/process mismatch: ${#GPU_ARRAY[@]} GPU ids, ${NUM_PROCESSES} processes" >&2
  exit 2
fi
for override in "$@"; do
  key="${override%%=*}"
  if [[ "${key}" == *history* || "${key}" == *memory* || "${key}" == "task" || "${key}" == "experiment" ]]; then
    echo "[rmbench-anchor-recent-memory] forbidden contract override: ${override}" >&2
    exit 2
  fi
done

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_OVERRIDES=(
  "experiment=${EXPERIMENT}"
  "data.train.dataset_dirs=['${DATASET_DIR}']"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
)

python "${SCRIPT_DIR}/validate_rmbench_anchor_recent_memory_config.py" \
  --experiment "${EXPERIMENT}" \
  "data.train.dataset_dirs=['${DATASET_DIR}']" \
  "data.train.latent_cache_dir=${TRAIN_CACHE_DIR}" \
  "data.val.latent_cache_dir=${VAL_CACHE_DIR}"

if [[ "${RUN_TEXT}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" \
    scripts/precompute_text_embeds.py \
    "${COMMON_OVERRIDES[@]}" \
    "overwrite=${OVERWRITE}" \
    "$@"
fi

if [[ "${RUN_VIDEO}" == "true" ]]; then
  for split in train val; do
    if [[ "${split}" == "train" ]]; then
      IS_TRAINING_SET=true
      CACHE_DIR="${TRAIN_CACHE_DIR}"
    else
      IS_TRAINING_SET=false
      CACHE_DIR="${VAL_CACHE_DIR}"
    fi
    echo "[rmbench-anchor-recent-memory] precompute ${split}: ${CACHE_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" \
      scripts/precompute_video_latents.py \
      "${COMMON_OVERRIDES[@]}" \
      "precompute_storage_format=episode_packed_v1" \
      "precompute_video_only=true" \
      "precompute_batch_size=${BATCH_SIZE}" \
      "precompute_num_workers=${NUM_WORKERS}" \
      "precompute_cache_dtype=model" \
      "precompute_resume=${PRECOMPUTE_RESUME}" \
      "overwrite=${OVERWRITE}" \
      "data.train.is_training_set=${IS_TRAINING_SET}" \
      "data.train.latent_cache_dir=${CACHE_DIR}" \
      "$@"
  done
fi
