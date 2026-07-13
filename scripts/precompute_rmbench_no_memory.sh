#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=rmbench_no_memory exec "${SCRIPT_DIR}/precompute.sh" "$@"
