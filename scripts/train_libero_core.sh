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

TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}"
SUITE_NAME="${SUITE_NAME:-libero}"
EXTRA_ARGS=("$@")

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29577}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$$}"
RUN_TAG="${RUN_TAG:-${SUITE_NAME}_lightwam}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}/${RUN_ID}_lightwam}"
WANDB_PROJECT="${WANDB_PROJECT:-light-wam}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}_${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_GROUP="${WANDB_GROUP:-null}"
RESUME="${RESUME:-null}"

DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS="${DATASET_DIRS:-}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${REPO_ROOT}/data/text_embeds_cache/libero}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_SIZE="${VIDEO_SIZE:-[224,448]}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-horizontal}"
NUM_OUTPUT_CAMERAS="${NUM_OUTPUT_CAMERAS:-2}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAX_STEPS="${MAX_STEPS:-150000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"

USE_WAM_ADAPTER="${USE_WAM_ADAPTER:-true}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
REMOVE_ORIGINAL_ACTION_EXPERT="${REMOVE_ORIGINAL_ACTION_EXPERT:-true}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-[8,16,24]}"
ADAPTER_DIM="${ADAPTER_DIM:-256}"
ADAPTER_SCALE="${ADAPTER_SCALE:-1.0}"

USE_BACKBONE_LORA="${USE_BACKBONE_LORA:-true}"
LORA_LAYER_INDICES="${LORA_LAYER_INDICES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-['self_attn.q','self_attn.k','self_attn.v','self_attn.o','cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2']}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

STATE_FUSION_PER_LAYER_DIM="${STATE_FUSION_PER_LAYER_DIM:-4608}"
STATE_FUSION_TRUNK_DIM="${STATE_FUSION_TRUNK_DIM:-6144}"
STATE_FUSION_NUM_TRUNK_BLOCKS="${STATE_FUSION_NUM_TRUNK_BLOCKS:-1}"
STATE_FUSION_STEP_POS_DIM="${STATE_FUSION_STEP_POS_DIM:-256}"
TOKEN_POOLING_NUM_QUERIES="${TOKEN_POOLING_NUM_QUERIES:-16}"

TEMPORAL_WEIGHTING_ENABLED="${TEMPORAL_WEIGHTING_ENABLED:-true}"
TEMPORAL_PREFIX_STEPS="${TEMPORAL_PREFIX_STEPS:-8}"
TEMPORAL_PREFIX_WEIGHT="${TEMPORAL_PREFIX_WEIGHT:-1.0}"
TEMPORAL_TAIL_WEIGHT="${TEMPORAL_TAIL_WEIGHT:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
PARAMETER_REPORT_ENABLED="${PARAMETER_REPORT_ENABLED:-true}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-False}"
DISTRIBUTED_DEBUG_SYNC_TRAIN_STEP="${DISTRIBUTED_DEBUG_SYNC_TRAIN_STEP:-false}"
DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED="${DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED:-false}"
DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES="${DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES:-65536}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# NCCL 2.26.2 on this Blackwell host hit cudaErrorIllegalAddress on >=128KB
# collectives. The environment has been upgraded to NCCL 2.27.3, so the normal
# default is to let NCCL choose protocols. Set NCCL_PROTO_DEFAULT=LL only as a
# fallback for older/broken NCCL stacks.
NCCL_PROTO_DEFAULT="${NCCL_PROTO_DEFAULT:-unset}"
if [[ -n "${NCCL_PROTO+x}" ]]; then
  export NCCL_PROTO
elif [[ "${NCCL_PROTO_DEFAULT}" == "unset" ]]; then
  unset NCCL_PROTO
elif [[ -n "${NCCL_PROTO_DEFAULT}" ]]; then
  export NCCL_PROTO="${NCCL_PROTO_DEFAULT}"
fi

DATASET_OVERRIDE_ARGS=()
if [[ -n "${DATASET_DIRS}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=${DATASET_DIRS}")
elif [[ -n "${DATASET_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=['${DATASET_DIR}']")
fi
if [[ -n "${LATENT_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.latent_cache_dir=${LATENT_CACHE_DIR}")
fi
if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'")
fi

echo "[launch] task=${TASK_NAME}"
echo "[launch] mode=lightwam_libero gpus=${GPU_IDS} num_processes=${NUM_PROCESSES} port=${MAIN_PROCESS_PORT}"
echo "[launch] suite=${SUITE_NAME}"
echo "[launch] run_tag=${RUN_TAG}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] resume=${RESUME}"
echo "[launch] dataset_dir=${DATASET_DIR:-<config default>}"
echo "[launch] dataset_dirs=${DATASET_DIRS:-<config default>}"
echo "[launch] latent_cache_dir=${LATENT_CACHE_DIR:-<config default>}"
echo "[launch] text_embed_cache_dir=${TEXT_EMBED_CACHE_DIR}"
echo "[launch] wandb.project=${WANDB_PROJECT} wandb.name=${WANDB_NAME} wandb.group=${WANDB_GROUP} wandb.mode=${WANDB_MODE}"
echo "[launch] batch_size=${BATCH_SIZE} grad_acc=${GRAD_ACC} num_workers=${NUM_WORKERS} eval_every=${EVAL_EVERY}"
echo "[launch] max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} checkpoint.max_to_keep=${CHECKPOINT_MAX_TO_KEEP} warmup_steps=${WARMUP_STEPS} num_epochs=${NUM_EPOCHS}"
echo "[launch] learning_rate=${LEARNING_RATE} lr_scheduler_type=${LR_SCHEDULER_TYPE}"
echo "[launch] model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}"
echo "[launch] model.state_fusion_action_expert_config.token_pooling_num_queries=${TOKEN_POOLING_NUM_QUERIES}"
echo "[launch] distributed.chunked_collectives.enabled=${DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED} max_bytes=${DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES}"
echo "[launch] accelerate_config=${ACCELERATE_CONFIG_FILE}"
echo "[launch] lightwam_env_bin=${LIGHTWAM_ENV_BIN:-<none>}"
echo "[launch] NCCL_PROTO=${NCCL_PROTO:-<unset>}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "task=${TASK_NAME}" \
  "output_dir=${OUTPUT_DIR}" \
  "resume=${RESUME}" \
  "wandb.enabled=true" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_NAME}" \
  "wandb.group=${WANDB_GROUP}" \
  "wandb.mode=${WANDB_MODE}" \
  "batch_size=${BATCH_SIZE}" \
  "gradient_accumulation_steps=${GRAD_ACC}" \
  "num_workers=${NUM_WORKERS}" \
  "eval_every=${EVAL_EVERY}" \
  "learning_rate=${LEARNING_RATE}" \
  "lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
  "max_steps=${MAX_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  "checkpoint.max_to_keep=${CHECKPOINT_MAX_TO_KEEP}" \
  "warmup_steps=${WARMUP_STEPS}" \
  "num_epochs=${NUM_EPOCHS}" \
  "distributed.debug_sync_train_step=${DISTRIBUTED_DEBUG_SYNC_TRAIN_STEP}" \
  "distributed.chunked_collectives.enabled=${DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED}" \
  "distributed.chunked_collectives.max_bytes=${DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES}" \
  "parameter_report.enabled=${PARAMETER_REPORT_ENABLED}" \
  "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}" \
  "model.loss.use_first_frame_residual_video_target=false" \
  "model.loss.action_temporal_weighting.enabled=${TEMPORAL_WEIGHTING_ENABLED}" \
  "model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}" \
  "model.loss.action_temporal_weighting.prefix_weight=${TEMPORAL_PREFIX_WEIGHT}" \
  "model.loss.action_temporal_weighting.tail_weight=${TEMPORAL_TAIL_WEIGHT}" \
  "model.wam_adapter.use_wam_adapter=${USE_WAM_ADAPTER}" \
  "model.wam_adapter.freeze_backbone=${FREEZE_BACKBONE}" \
  "model.wam_adapter.remove_original_action_expert=${REMOVE_ORIGINAL_ACTION_EXPERT}" \
  "model.wam_adapter.adapter_layer_indices=${ADAPTER_LAYERS}" \
  "model.wam_adapter.adapter_dim=${ADAPTER_DIM}" \
  "model.wam_adapter.adapter_scale=${ADAPTER_SCALE}" \
  "model.wam_adapter.use_backbone_lora=${USE_BACKBONE_LORA}" \
  "model.wam_adapter.lora_layer_indices=${LORA_LAYER_INDICES}" \
  "model.wam_adapter.lora_target_modules=${LORA_TARGET_MODULES}" \
  "model.wam_adapter.lora_rank=${LORA_RANK}" \
  "model.wam_adapter.lora_alpha=${LORA_ALPHA}" \
  "model.wam_adapter.lora_dropout=${LORA_DROPOUT}" \
  "model.state_fusion_action_expert_config.per_layer_dim=${STATE_FUSION_PER_LAYER_DIM}" \
  "model.state_fusion_action_expert_config.trunk_dim=${STATE_FUSION_TRUNK_DIM}" \
  "model.state_fusion_action_expert_config.num_trunk_blocks=${STATE_FUSION_NUM_TRUNK_BLOCKS}" \
  "model.state_fusion_action_expert_config.step_pos_dim=${STATE_FUSION_STEP_POS_DIM}" \
  "model.state_fusion_action_expert_config.token_pooling_type=learned_query" \
  "model.state_fusion_action_expert_config.token_pooling_num_queries=${TOKEN_POOLING_NUM_QUERIES}" \
  "model.state_fusion_action_expert_config.token_pooling_num_heads=8" \
  "model.state_fusion_action_expert_config.feature_sources=[adapted]" \
  "model.video_backbone_type=wan2_1_t2v" \
  "model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B" \
  "model.video_latent_spatial_downsample_factor=2" \
  "model.apply_video_latent_downsample_to_action_branch=false" \
  "timing_breakdown.enabled=false" \
  "timing_breakdown.sync_cuda=false" \
  "train_visualization.enabled=true" \
  "train_visualization.every=500" \
  "train_visualization.fps=8" \
  "train_visualization.tiled=false" \
  "data.train.use_latent_cache=true" \
  "data.train.num_frames=${NUM_FRAMES}" \
  "data.train.action_video_freq_ratio=${ACTION_VIDEO_FREQ_RATIO}" \
  "data.train.video_size=${VIDEO_SIZE}" \
  "data.train.concat_multi_camera=${CONCAT_MULTI_CAMERA}" \
  "data.train.processor.num_output_cameras=${NUM_OUTPUT_CAMERAS}" \
  "${DATASET_OVERRIDE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
