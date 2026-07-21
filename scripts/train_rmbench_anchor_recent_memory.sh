#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIGHTWAM_ENV_BIN="${LIGHTWAM_ENV_BIN:-/home/jian/.local/share/mamba/envs/fastwam/bin}"
if [[ ! -x "${LIGHTWAM_ENV_BIN}/python" || ! -x "${LIGHTWAM_ENV_BIN}/accelerate" ]]; then
  echo "[rmbench-anchor-recent-memory] invalid LIGHTWAM_ENV_BIN: ${LIGHTWAM_ENV_BIN}" >&2
  exit 2
fi
export PATH="${LIGHTWAM_ENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

EXPERIMENT="rmbench/put_back_block_anchor_recent_memory_5k"
DEFAULT_INIT_CKPT="${REPO_ROOT}/runs/rmbench_no_memory/put_back_block/r460k_5k/2026-07-21_12-25-24/checkpoints/weights/step_005000.pt"
INIT_CKPT="${INIT_CKPT:-${DEFAULT_INIT_CKPT}}"
TRAIN_CACHE_DIR="${RMBENCH_TRAIN_LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B/rmbench_tasks/put_back_block_anchor1_recent1to4_episode_packed_train}"
VAL_CACHE_DIR="${RMBENCH_VAL_LATENT_CACHE_DIR:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B/rmbench_tasks/put_back_block_anchor1_recent1to4_episode_packed_val}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_INPUT_PREFLIGHT="${SKIP_INPUT_PREFLIGHT:-false}"

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "[rmbench-anchor-recent-memory] missing initialization checkpoint: ${INIT_CKPT}" >&2
  exit 2
fi
if [[ "${TRAIN_CACHE_DIR}" == "${VAL_CACHE_DIR}" ]]; then
  echo "[rmbench-anchor-recent-memory] train and val cache directories must differ" >&2
  exit 2
fi
for override in "$@"; do
  key="${override%%=*}"
  if [[ "${key}" == *history* || "${key}" == *memory* || "${key}" == "task" || "${key}" == "experiment" || "${key}" == "resume" ]]; then
    echo "[rmbench-anchor-recent-memory] forbidden contract override: ${override}" >&2
    echo "Use INIT_CKPT to select the initialization checkpoint." >&2
    exit 2
  fi
done

OVERRIDES=(
  "resume=${INIT_CKPT}"
  "data.train.latent_cache_dir=${TRAIN_CACHE_DIR}"
  "data.val.latent_cache_dir=${VAL_CACHE_DIR}"
  "$@"
)

python "${SCRIPT_DIR}/validate_rmbench_anchor_recent_memory_config.py" \
  --experiment "${EXPERIMENT}" \
  "${OVERRIDES[@]}"

LAUNCH_FLAGS=()
if [[ "${DRY_RUN}" == "true" ]]; then
  LAUNCH_FLAGS+=(--dry-run)
fi
if [[ "${SKIP_INPUT_PREFLIGHT}" == "true" ]]; then
  LAUNCH_FLAGS+=(--skip-input-preflight)
fi

exec python "${SCRIPT_DIR}/launch_rmbench_experiment.py" \
  --experiment "${EXPERIMENT}" \
  "${LAUNCH_FLAGS[@]}" \
  "${OVERRIDES[@]}"
