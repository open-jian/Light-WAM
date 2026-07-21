#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIGHTWAM_ENV_BIN="${LIGHTWAM_ENV_BIN:-/home/jian/.local/share/mamba/envs/fastwam/bin}"
if [[ ! -x "${LIGHTWAM_ENV_BIN}/python" ]]; then
  echo "[rmbench-anchor-recent-memory] invalid LIGHTWAM_ENV_BIN: ${LIGHTWAM_ENV_BIN}" >&2
  exit 2
fi
export PATH="${LIGHTWAM_ENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

CKPT="${CKPT:-}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to/memory/checkpoint $0 [Hydra overrides...]" >&2
  exit 2
fi
for override in "$@"; do
  key="${override%%=*}"
  if [[ "${override}" == --* || "${key}" == *history* || "${key}" == *memory* || "${key}" == "task" || "${key}" == "experiment" || "${key}" == "ckpt" || "${key}" == "EVALUATION.use_training_run_config" || "${key}" == "EVALUATION.training_config_path" || "${key}" == "EVALUATION.action_horizon" || "${key}" == "EVALUATION.replan_steps" || "${key}" == "EVALUATION.skip_get_obs_within_replan" ]]; then
    echo "[rmbench-anchor-recent-memory] forbidden contract override: ${override}" >&2
    exit 2
  fi
done

VALIDATE_ARGS=(--checkpoint "${CKPT}")
EVAL_CONFIG_ARGS=()
if [[ -n "${TRAINING_CONFIG_PATH}" ]]; then
  VALIDATE_ARGS+=(--training-config "${TRAINING_CONFIG_PATH}")
  EVAL_CONFIG_ARGS+=("EVALUATION.training_config_path=${TRAINING_CONFIG_PATH}")
fi
python "${SCRIPT_DIR}/validate_rmbench_anchor_recent_memory_config.py" "${VALIDATE_ARGS[@]}"

exec python experiments/rmbench/eval_rmbench_single.py \
  --config-name sim_rmbench_put_back_block_anchor_recent_memory_5k \
  "ckpt=${CKPT}" \
  "EVALUATION.task_name=put_back_block" \
  "EVALUATION.use_training_run_config=true" \
  "${EVAL_CONFIG_ARGS[@]}" \
  "$@"
