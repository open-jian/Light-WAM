#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jian/.local/share/mamba/envs/fastwam/bin/python}"
RAW_REPO="${RAW_REPO:-yuanty/robotwin2.0-fastwam}"
LATENT_REPO="${LATENT_REPO:-l1ziang/lightwam-offline-cache}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/robotwin2.0}"
LATENT_ROOT="${LATENT_ROOT:-${REPO_ROOT}/data/latent_cache_Wan2.1-T2V-1.3B}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-true}"
DOWNLOAD_LATENTS="${DOWNLOAD_LATENTS:-true}"
EXTRACT_RAW="${EXTRACT_RAW:-true}"
EXTRACT_LATENTS="${EXTRACT_LATENTS:-true}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

hf_download() {
  local repo_id="$1"
  shift
  "${PYTHON_BIN}" -m huggingface_hub.commands.huggingface_cli download \
    "${repo_id}" \
    --repo-type dataset \
    --max-workers "${HF_MAX_WORKERS}" \
    "$@"
}

if [[ "${DOWNLOAD_RAW}" == "true" && ! -f "${RAW_ROOT}/robotwin2.0/meta/tasks.jsonl" ]]; then
  mkdir -p "${RAW_ROOT}"
  hf_download "${RAW_REPO}" --local-dir "${RAW_ROOT}"
  if [[ "${EXTRACT_RAW}" == "true" ]]; then
    (cd "${RAW_ROOT}" && cat robotwin2.0.tar.gz.part-* | tar -xzf -)
  fi
fi

if [[ "${DOWNLOAD_LATENTS}" == "true" && ! -d "${LATENT_ROOT}/robotwin_3cam384_sharded" ]]; then
  mkdir -p "${LATENT_ROOT}"
  hf_download "${LATENT_REPO}" \
    --include "latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded.tar.part-*" \
    --local-dir "${REPO_ROOT}/data"
  if [[ "${EXTRACT_LATENTS}" == "true" ]]; then
    (cd "${LATENT_ROOT}" && cat robotwin_3cam384_sharded.tar.part-* | tar -xf -)
  fi
fi

echo "[assets] raw=${RAW_ROOT}/robotwin2.0"
echo "[assets] latents=${LATENT_ROOT}/robotwin_3cam384_sharded"
echo "[assets] archive parts are retained for integrity/resume."
