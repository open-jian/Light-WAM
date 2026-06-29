#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-lightwam_libero_paper_repro}"
RUN_PREFIX="${RUN_PREFIX:-paper_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/runs/${RUN_TAG}}"
LAUNCH_DIR="${BASE_OUTPUT_DIR}/${RUN_PREFIX}_schedule"
LOG_FILE="${LAUNCH_DIR}/schedule.log"

mkdir -p "${LAUNCH_DIR}"
printf '%s\n' "${LAUNCH_DIR}" > "${BASE_OUTPUT_DIR}/LATEST_LAUNCH_DIR.txt"
printf '%s\n' "${RUN_PREFIX}" > "${BASE_OUTPUT_DIR}/LATEST_RUN_PREFIX.txt"

RUN_TAG="${RUN_TAG}" \
RUN_PREFIX="${RUN_PREFIX}" \
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
setsid bash "${SCRIPT_DIR}/train_libero_paper_schedule.sh" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
PID="$!"

printf '%s\n' "${PID}" > "${LAUNCH_DIR}/schedule.pid"
cat > "${LAUNCH_DIR}/launch_summary.txt" <<EOF
run_tag=${RUN_TAG}
run_prefix=${RUN_PREFIX}
base_output_dir=${BASE_OUTPUT_DIR}
launch_dir=${LAUNCH_DIR}
log_file=${LOG_FILE}
pid=${PID}
gpu_ids=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
libero_suite_order=${LIBERO_SUITE_ORDER:-${LIBERO_SUITES:-object spatial goal libero_10}}
target_global_batch_size=${TARGET_GLOBAL_BATCH_SIZE:-64}
batch_size=${BATCH_SIZE:-auto}
grad_acc=${GRAD_ACC:-1}
allow_global_batch_mismatch=${ALLOW_GLOBAL_BATCH_MISMATCH:-false}
started_at=$(date)
EOF

echo "[launched] run_prefix=${RUN_PREFIX}"
echo "[launched] launch_dir=${LAUNCH_DIR}"
echo "[launched] log=${LOG_FILE}"
echo "[launched] pid=${PID}"
echo "[launched] gpu_ids=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
echo "[launched] suite_order=${LIBERO_SUITE_ORDER:-${LIBERO_SUITES:-object spatial goal libero_10}}"
echo "[monitor] bash ${SCRIPT_DIR}/monitor_libero_paper_schedule.sh"
echo "[stop] kill -- -${PID}"
