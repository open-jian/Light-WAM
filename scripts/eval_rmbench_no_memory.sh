#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

CKPT="${CKPT:-}"
TASK_NAME="${TASK_NAME:-}"
if [[ -z "${CKPT}" || -z "${TASK_NAME}" ]]; then
  echo "Usage: CKPT=/path/to/checkpoint TASK_NAME=press_button $0 [Hydra overrides...]" >&2
  exit 2
fi

exec python experiments/rmbench/eval_rmbench_single.py \
  "ckpt=${CKPT}" \
  "EVALUATION.task_name=${TASK_NAME}" \
  "$@"
