#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIGHTWAM_ENV_BIN="${LIGHTWAM_ENV_BIN:-/home/jian/.local/share/mamba/envs/fastwam/bin}"
export PATH="${LIGHTWAM_ENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset NCCL_PROTO

DEFAULT_TASKS=(
  battery_try
  blocks_ranking_try
  cover_blocks
  observe_and_pickup
  place_block_mat
  press_button
  put_back_block
  rearrange_blocks
  swap_T
  swap_blocks
)
if [[ -n "${RMBENCH_TASKS:-}" ]]; then
  read -r -a TASKS <<<"${RMBENCH_TASKS}"
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

RUN_STATS="${RUN_STATS:-true}"
RUN_VIDEO="${RUN_VIDEO:-true}"
PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-0,2,3}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-3}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-16}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-4}"
PRECOMPUTE_SHARD_SIZE="${PRECOMPUTE_SHARD_SIZE:-1024}"
TASK_CONFIG="rmbench_no_memory_3cam_384_1e-4"
TEXT_CACHE_DIR="./data/text_embeds_cache/rmbench"
LATENT_ROOT="./data/latent_cache_Wan2.1-T2V-1.3B/rmbench_tasks"

IFS=',' read -r -a GPU_ARRAY <<<"${PRECOMPUTE_GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${PRECOMPUTE_NUM_PROCESSES}" ]]; then
  echo "GPU/process mismatch: ${#GPU_ARRAY[@]} GPU ids, ${PRECOMPUTE_NUM_PROCESSES} processes" >&2
  exit 2
fi
if [[ ! -d "${TEXT_CACHE_DIR}" ]]; then
  echo "Missing shared RM-Bench text cache: ${TEXT_CACHE_DIR}" >&2
  exit 2
fi

for task in "${TASKS[@]}"; do
  dataset_dir="./data/rmbench/tasks/${task}"
  stats_path="${dataset_dir}/dataset_stats_train.json"
  latent_dir="${LATENT_ROOT}/${task}_3cam384_sharded"
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Missing task dataset: ${dataset_dir}" >&2
    exit 2
  fi

  if [[ "${RUN_STATS}" == "true" ]]; then
    echo "[rmbench-assets] stats start task=${task} output=${stats_path}"
    python scripts/compute_lerobot_dataset_stats.py \
      "task=${TASK_CONFIG}" \
      "+output_stats_path=${stats_path}" \
      "data.train.dataset_dirs=['${dataset_dir}']" \
      "data.train.val_set_proportion=0.10"
    echo "[rmbench-assets] stats done task=${task}"
  fi

  if [[ "${RUN_VIDEO}" == "true" ]]; then
    echo "[rmbench-assets] video start task=${task} output=${latent_dir}"
    CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" torchrun \
      --standalone \
      --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
      scripts/precompute_video_latents.py \
      "task=${TASK_CONFIG}" \
      "overwrite=false" \
      "model.video_backbone_type=wan2_1_t2v" \
      "model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B" \
      "precompute_storage_format=sharded_v1" \
      "precompute_video_only=true" \
      "precompute_shard_size=${PRECOMPUTE_SHARD_SIZE}" \
      "precompute_batch_size=${PRECOMPUTE_BATCH_SIZE}" \
      "precompute_num_workers=${PRECOMPUTE_NUM_WORKERS}" \
      "precompute_cache_dtype=model" \
      "precompute_resume=true" \
      "precompute_timing.enabled=true" \
      "precompute_timing.sync_cuda=false" \
      "precompute_timing.log_every=10" \
      "data.train.dataset_dirs=['${dataset_dir}']" \
      "data.train.val_set_proportion=0.10" \
      "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
      "data.train.latent_cache_dir=${latent_dir}"
    echo "[rmbench-assets] video done task=${task}"
  fi
done
