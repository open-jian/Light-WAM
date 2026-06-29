#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
LIGHTWAM_ENV_BIN="${LIGHTWAM_ENV_BIN:-}"
if [[ -n "${LIGHTWAM_ENV_BIN}" ]]; then
  export PATH="${LIGHTWAM_ENV_BIN}:${PATH}"
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NCCL_PROTO_DEFAULT="${NCCL_PROTO_DEFAULT:-unset}"
if [[ -n "${NCCL_PROTO+x}" ]]; then
  export NCCL_PROTO
elif [[ "${NCCL_PROTO_DEFAULT}" == "unset" ]]; then
  unset NCCL_PROTO
elif [[ -n "${NCCL_PROTO_DEFAULT}" ]]; then
  export NCCL_PROTO="${NCCL_PROTO_DEFAULT}"
fi

TARGET="${TARGET:-${1:-libero_spatial}}"
PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-0,1,2,3}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-4}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-32}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
PRECOMPUTE_SHARD_SIZE="${PRECOMPUTE_SHARD_SIZE:-1024}"
PRECOMPUTE_LOG_EVERY="${PRECOMPUTE_LOG_EVERY:-10}"
RUN_TEXT="${RUN_TEXT:-true}"
RUN_VIDEO="${RUN_VIDEO:-true}"
OVERWRITE="${OVERWRITE:-false}"

case "${TARGET}" in
  libero_spatial)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_spatial_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_object)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_object_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_goal)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_goal_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  libero_10)
    TASK_NAME="libero_uncond_2cam224_1e-4"
    DATASET_DIR="./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224"
    TEXT_CACHE_DIR="./data/text_embeds_cache/libero"
    ;;
  robotwin)
    TASK_NAME="robotwin_uncond_3cam_384_1e-4"
    DATASET_DIR="./data/robotwin2.0/robotwin2.0"
    LATENT_CACHE_DIR="./data/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded"
    TEXT_CACHE_DIR="./data/text_embeds_cache/robotwin"
    ;;
  *)
    echo "Unsupported TARGET=${TARGET}. Expected one of: libero_spatial, libero_object, libero_goal, libero_10, robotwin" >&2
    exit 1
    ;;
esac

echo "[precompute] target=${TARGET}"
echo "[precompute] task=${TASK_NAME}"
echo "[precompute] dataset_dir=${DATASET_DIR}"
echo "[precompute] text_cache_dir=${TEXT_CACHE_DIR}"
echo "[precompute] latent_cache_dir=${LATENT_CACHE_DIR}"
echo "[precompute] gpus=${PRECOMPUTE_GPU_IDS} num_processes=${PRECOMPUTE_NUM_PROCESSES} batch_size=${PRECOMPUTE_BATCH_SIZE}"
echo "[precompute] lightwam_env_bin=${LIGHTWAM_ENV_BIN:-<none>}"
echo "[precompute] NCCL_PROTO=${NCCL_PROTO:-<unset>}"

if [[ "${RUN_TEXT}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" torchrun --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
    scripts/precompute_text_embeds.py \
    "task=${TASK_NAME}" \
    "overwrite=${OVERWRITE}" \
    "data.train.dataset_dirs=['${DATASET_DIR}']" \
    "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
fi

if [[ "${RUN_VIDEO}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU_IDS}" torchrun --standalone --nproc_per_node="${PRECOMPUTE_NUM_PROCESSES}" \
    scripts/precompute_video_latents.py \
    "task=${TASK_NAME}" \
    "overwrite=${OVERWRITE}" \
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
    "precompute_timing.log_every=${PRECOMPUTE_LOG_EVERY}" \
    "data.train.dataset_dirs=['${DATASET_DIR}']" \
    "data.train.latent_cache_dir=${LATENT_CACHE_DIR}"
fi
