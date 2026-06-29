#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-lightwam_libero_paper_repro}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}}"

if [[ $# -gt 0 ]]; then
  LAUNCH_DIR="$1"
else
  LAUNCH_DIR="$(cat "${BASE_OUTPUT_DIR}/LATEST_LAUNCH_DIR.txt")"
fi

LOG_FILE="${LAUNCH_DIR}/schedule.log"
PID_FILE="${LAUNCH_DIR}/schedule.pid"

echo "[launch_dir] ${LAUNCH_DIR}"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "[pid] ${PID}"
  if ps -p "${PID}" -o pid,ppid,pgid,sid,stat,etime,cmd; then
    :
  else
    echo "[state] schedule process exited"
  fi
else
  echo "[pid] missing"
fi

echo "[gpu]"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true

echo "[runs]"
find "${BASE_OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort | tail -n 12

if [[ -f "${LOG_FILE}" ]]; then
  echo "[recent progress]"
  rg -n "\[schedule|\[cache|\[precompute|\[pair|\[suite|step=[0-9]+/|loss=|wandb|Traceback|Error|CUDA|NCCL|\\[ckpt|\\[done|ckpt-prune" "${LOG_FILE}" | tail -n 60 || true
  echo "[tail]"
  tail -n 40 "${LOG_FILE}"
else
  echo "[log missing] ${LOG_FILE}"
fi
