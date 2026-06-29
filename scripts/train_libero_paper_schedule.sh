#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-lightwam_libero_paper_repro}"
RUN_PREFIX="${RUN_PREFIX:-paper_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-light-wam-libero-paper-repro}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_PREFIX}}"

# The paper-repro launch used 4 GPUs x 16 samples/GPU = global batch 64.
# Keep that global batch by default when changing GPU counts, so optimizer
# steps, sample count, and LR schedule remain comparable.
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
GRAD_ACC="${GRAD_ACC:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-}"
ALLOW_GLOBAL_BATCH_MISMATCH="${ALLOW_GLOBAL_BATCH_MISMATCH:-false}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29577}"
DRY_RUN="${DRY_RUN:-false}"

# Comma or space separated. Examples:
#   LIBERO_SUITE_ORDER="goal libero_10"
#   LIBERO_SUITE_ORDER="object,spatial,goal,libero_10"
LIBERO_SUITE_ORDER="${LIBERO_SUITE_ORDER:-${LIBERO_SUITES:-object spatial goal libero_10}}"

PRECOMPUTE_MISSING_LATENTS="${PRECOMPUTE_MISSING_LATENTS:-true}"
PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS:-${GPU_IDS}}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-32}"
PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS:-8}"
PRECOMPUTE_SHARD_SIZE="${PRECOMPUTE_SHARD_SIZE:-1024}"
PRECOMPUTE_LOG_EVERY="${PRECOMPUTE_LOG_EVERY:-10}"
PRECOMPUTE_RUN_TEXT="${PRECOMPUTE_RUN_TEXT:-false}"

mkdir -p "${BASE_OUTPUT_DIR}"
printf '%s\n' "${BASE_OUTPUT_DIR}" > "${BASE_OUTPUT_DIR}/LATEST_BASE_OUTPUT_DIR.txt"

die() {
  echo "[config-error] $*" >&2
  exit 2
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

parse_gpu_ids() {
  [[ -n "${GPU_IDS}" ]] || die "GPU_IDS is empty"
  IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
  [[ "${#GPU_ID_LIST[@]}" -gt 0 ]] || die "GPU_IDS did not contain any GPU ids"

  local gpu_id
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    [[ -n "${gpu_id}" ]] || die "GPU_IDS contains an empty entry: ${GPU_IDS}"
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "GPU_IDS must be numeric comma-separated ids, got: ${GPU_IDS}"
  done

  GPU_COUNT="${#GPU_ID_LIST[@]}"
  NUM_PROCESSES="${NUM_PROCESSES:-${GPU_COUNT}}"
  is_positive_int "${NUM_PROCESSES}" || die "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}"
  if [[ "${NUM_PROCESSES}" -ne "${GPU_COUNT}" ]]; then
    die "NUM_PROCESSES (${NUM_PROCESSES}) must match the number of GPU_IDS (${GPU_COUNT}); set GPU_IDS to exactly the GPUs you want to use"
  fi
}

compute_batch_size() {
  is_positive_int "${GRAD_ACC}" || die "GRAD_ACC must be a positive integer, got: ${GRAD_ACC}"
  is_positive_int "${TARGET_GLOBAL_BATCH_SIZE}" || die "TARGET_GLOBAL_BATCH_SIZE must be a positive integer, got: ${TARGET_GLOBAL_BATCH_SIZE}"

  if [[ -z "${BATCH_SIZE}" ]]; then
    local denom="$((NUM_PROCESSES * GRAD_ACC))"
    if [[ "$((TARGET_GLOBAL_BATCH_SIZE % denom))" -ne 0 ]]; then
      die "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE} is not divisible by NUM_PROCESSES*GRAD_ACC=${denom}; set BATCH_SIZE explicitly or choose a divisible target global batch"
    fi
    BATCH_SIZE="$((TARGET_GLOBAL_BATCH_SIZE / denom))"
  fi

  is_positive_int "${BATCH_SIZE}" || die "BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}"
  EFFECTIVE_GLOBAL_BATCH="$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACC))"
  if [[ "${EFFECTIVE_GLOBAL_BATCH}" -ne "${TARGET_GLOBAL_BATCH_SIZE}" && "${ALLOW_GLOBAL_BATCH_MISMATCH}" != "true" ]]; then
    die "effective global batch is ${EFFECTIVE_GLOBAL_BATCH}, but TARGET_GLOBAL_BATCH_SIZE is ${TARGET_GLOBAL_BATCH_SIZE}; unset BATCH_SIZE, set TARGET_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH}, or set ALLOW_GLOBAL_BATCH_MISMATCH=true"
  fi
}

parse_suite_order() {
  local suite_order="${LIBERO_SUITE_ORDER//,/ }"
  read -r -a SCHEDULE_SUITES <<< "${suite_order}"
  [[ "${#SCHEDULE_SUITES[@]}" -gt 0 ]] || die "LIBERO_SUITE_ORDER did not contain any suites"

  local suite
  for suite in "${SCHEDULE_SUITES[@]}"; do
    suite_script "${suite}" >/dev/null
    suite_max_steps "${suite}" >/dev/null
    suite_save_every "${suite}" >/dev/null
  done
}

cache_dir_for_suite() {
  case "$1" in
    object) echo "./data/latent_cache_Wan2.1-T2V-1.3B/libero_object_2cam224" ;;
    spatial) echo "./data/latent_cache_Wan2.1-T2V-1.3B/libero_spatial_2cam224" ;;
    goal) echo "./data/latent_cache_Wan2.1-T2V-1.3B/libero_goal_2cam224" ;;
    libero_10) echo "./data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224" ;;
    *) die "unknown suite: $1" ;;
  esac
}

precompute_target_for_suite() {
  case "$1" in
    object) echo "libero_object" ;;
    spatial) echo "libero_spatial" ;;
    goal) echo "libero_goal" ;;
    libero_10) echo "libero_10" ;;
    *) die "unknown suite: $1" ;;
  esac
}

suite_script() {
  case "$1" in
    object) echo "train_libero_object.sh" ;;
    spatial) echo "train_libero_spatial.sh" ;;
    goal) echo "train_libero_goal.sh" ;;
    libero_10) echo "train_libero_10.sh" ;;
    *) die "unknown suite: $1" ;;
  esac
}

suite_max_steps() {
  case "$1" in
    object) echo "${OBJECT_MAX_STEPS:-12500}" ;;
    spatial) echo "${SPATIAL_MAX_STEPS:-60000}" ;;
    goal) echo "${GOAL_MAX_STEPS:-60000}" ;;
    libero_10) echo "${LIBERO_10_MAX_STEPS:-80000}" ;;
    *) die "unknown suite: $1" ;;
  esac
}

suite_save_every() {
  case "$1" in
    object) echo "${OBJECT_SAVE_EVERY:-2500}" ;;
    spatial) echo "${SPATIAL_SAVE_EVERY:-5000}" ;;
    goal) echo "${GOAL_SAVE_EVERY:-5000}" ;;
    libero_10) echo "${LIBERO_10_SAVE_EVERY:-5000}" ;;
    *) die "unknown suite: $1" ;;
  esac
}

is_latent_cache_ready() {
  local suite="$1"
  local cache_dir
  cache_dir="$(cache_dir_for_suite "${suite}")"
  [[ -f "${cache_dir}/index.pt" ]] && return 0
  [[ -n "$(find "${cache_dir}" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null || true)" ]]
}

run_precompute_suite() {
  local suite="$1"
  local target
  target="$(precompute_target_for_suite "${suite}")"

  local output_dir="${BASE_OUTPUT_DIR}/${RUN_PREFIX}_precompute_${suite}"
  local log_file="${output_dir}/precompute.log"
  mkdir -p "${output_dir}"

  {
    echo "[precompute-start] $(date) gpus=${PRECOMPUTE_GPU_IDS} processes=${PRECOMPUTE_NUM_PROCESSES} suite=${suite} target=${target}"
    echo "[precompute-start] output_dir=${output_dir}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[dry-run] would run TARGET=${target} bash ${SCRIPT_DIR}/precompute.sh"
    else
      TARGET="${target}" \
      PRECOMPUTE_GPU_IDS="${PRECOMPUTE_GPU_IDS}" \
      PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES}" \
      PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE}" \
      PRECOMPUTE_NUM_WORKERS="${PRECOMPUTE_NUM_WORKERS}" \
      PRECOMPUTE_SHARD_SIZE="${PRECOMPUTE_SHARD_SIZE}" \
      PRECOMPUTE_LOG_EVERY="${PRECOMPUTE_LOG_EVERY}" \
      RUN_TEXT="${PRECOMPUTE_RUN_TEXT}" \
      RUN_VIDEO=true \
      OVERWRITE=false \
      bash "${SCRIPT_DIR}/precompute.sh"
    fi
    echo "[precompute-done] $(date) suite=${suite}"
  } 2>&1 | tee "${log_file}"
}

ensure_latent_caches() {
  local missing=()
  local suite
  for suite in "${SCHEDULE_SUITES[@]}"; do
    if is_latent_cache_ready "${suite}"; then
      echo "[cache-ready] suite=${suite} dir=$(cache_dir_for_suite "${suite}")"
    else
      echo "[cache-missing] suite=${suite} dir=$(cache_dir_for_suite "${suite}")"
      missing+=("${suite}")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[cache] all required latent caches are present"
    return 0
  fi

  if [[ "${PRECOMPUTE_MISSING_LATENTS}" != "true" ]]; then
    echo "[cache-error] missing latent caches: ${missing[*]}"
    echo "[cache-error] set PRECOMPUTE_MISSING_LATENTS=true or run scripts/precompute_libero.sh first"
    exit 2
  fi

  echo "[cache] precomputing missing latent caches sequentially: ${missing[*]}"
  echo "[cache] precompute_run_text=${PRECOMPUTE_RUN_TEXT} batch_size=${PRECOMPUTE_BATCH_SIZE} shard_size=${PRECOMPUTE_SHARD_SIZE}"
  for suite in "${missing[@]}"; do
    run_precompute_suite "${suite}"
  done

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[cache] dry-run mode: skipping post-precompute cache validation"
    return 0
  fi

  for suite in "${SCHEDULE_SUITES[@]}"; do
    if ! is_latent_cache_ready "${suite}"; then
      echo "[cache-error] cache still missing after precompute: suite=${suite} dir=$(cache_dir_for_suite "${suite}")"
      exit 2
    fi
  done
  echo "[cache] all required latent caches are ready after precompute"
}

run_suite() {
  local suite="$1"
  local index="$2"
  local total="$3"
  local script
  local max_steps
  local save_every
  script="$(suite_script "${suite}")"
  max_steps="$(suite_max_steps "${suite}")"
  save_every="$(suite_save_every "${suite}")"

  local run_id="${RUN_PREFIX}_${suite}_${max_steps}steps"
  local output_dir="${BASE_OUTPUT_DIR}/${run_id}"
  local log_file="${output_dir}/train.log"
  mkdir -p "${output_dir}"

  {
    echo "[suite-start] $(date) ${index}/${total} gpus=${GPU_IDS} processes=${NUM_PROCESSES} suite=${suite} max_steps=${max_steps} save_every=${save_every}"
    echo "[suite-start] output_dir=${output_dir}"
    echo "[suite-start] wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_GROUP} wandb_name=${run_id}"
    echo "[suite-start] batch_size_per_gpu=${BATCH_SIZE} grad_acc=${GRAD_ACC} global_batch=${EFFECTIVE_GLOBAL_BATCH}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[dry-run] would run bash ${SCRIPT_DIR}/${script}"
    else
      RUN_ID="${run_id}" \
      RUN_TAG="${RUN_TAG}" \
      OUTPUT_DIR="${output_dir}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      WANDB_GROUP="${WANDB_GROUP}" \
      WANDB_NAME="${run_id}" \
      WANDB_MODE="${WANDB_MODE}" \
      GPU_IDS="${GPU_IDS}" \
      NUM_PROCESSES="${NUM_PROCESSES}" \
      MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
      MAX_STEPS="${max_steps}" \
      SAVE_EVERY="${save_every}" \
      CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP}" \
      EVAL_EVERY="${EVAL_EVERY}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      GRAD_ACC="${GRAD_ACC}" \
      NUM_WORKERS="${NUM_WORKERS}" \
      bash "${SCRIPT_DIR}/${script}" \
        "log_every=${LOG_EVERY}" \
        "parameter_report.enabled=false" \
        "train_visualization.enabled=false"
    fi
    echo "[suite-done] $(date) suite=${suite}"
  } 2>&1 | tee "${log_file}"
}

parse_gpu_ids
compute_batch_size
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-${NUM_PROCESSES}}"
is_positive_int "${PRECOMPUTE_NUM_PROCESSES}" || die "PRECOMPUTE_NUM_PROCESSES must be a positive integer, got: ${PRECOMPUTE_NUM_PROCESSES}"
parse_suite_order

echo "[schedule-start] $(date)"
echo "[schedule] run_prefix=${RUN_PREFIX}"
echo "[schedule] base_output_dir=${BASE_OUTPUT_DIR}"
echo "[schedule] suite_order=${SCHEDULE_SUITES[*]}"
echo "[schedule] gpus=${GPU_IDS} num_processes=${NUM_PROCESSES}"
echo "[schedule] target_global_batch=${TARGET_GLOBAL_BATCH_SIZE} batch_size_per_gpu=${BATCH_SIZE} grad_acc=${GRAD_ACC}"
echo "[schedule] effective_global_batch=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACC))"
echo "[schedule] allow_global_batch_mismatch=${ALLOW_GLOBAL_BATCH_MISMATCH}"
echo "[schedule] checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP}"
echo "[schedule] paper targets: object=${OBJECT_MAX_STEPS:-12500} spatial=${SPATIAL_MAX_STEPS:-60000} goal=${GOAL_MAX_STEPS:-60000} libero_10=${LIBERO_10_MAX_STEPS:-80000}"
echo "[schedule] precompute_missing_latents=${PRECOMPUTE_MISSING_LATENTS} precompute_gpus=${PRECOMPUTE_GPU_IDS} precompute_processes=${PRECOMPUTE_NUM_PROCESSES}"
echo "[schedule] dry_run=${DRY_RUN}"

ensure_latent_caches

total_suites="${#SCHEDULE_SUITES[@]}"
suite_index=1
for suite in "${SCHEDULE_SUITES[@]}"; do
  run_suite "${suite}" "${suite_index}" "${total_suites}"
  suite_index="$((suite_index + 1))"
done

echo "[schedule-done] $(date)"
