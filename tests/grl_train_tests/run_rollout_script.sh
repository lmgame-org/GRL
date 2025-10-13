#!/usr/bin/env bash

set -Eeuo pipefail

# Resolve repository root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CACHE_DIR="${REPO_ROOT}/cache"

mkdir -p "${CACHE_DIR}"

# Timestamped log filename
TS="$(date +%Y%m%d_%H%M%S)"
STREAM_LOG="${CACHE_DIR}/rollout_stream_${TS}.log"
LATEST_LOG="${CACHE_DIR}/rollout_stream_latest.log"

# Choose python interpreter
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python interpreter not found." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "[run_rollout_script] repo_root=${REPO_ROOT}" | tee "${STREAM_LOG}"
echo "[run_rollout_script] cache_dir=${CACHE_DIR}" | tee -a "${STREAM_LOG}"
echo "[run_rollout_script] python=$(${PY} -V 2>&1)" | tee -a "${STREAM_LOG}"

CMD="${PY} ${REPO_ROOT}/tests/grl_train_tests/qwen_rollout_test.py"
echo "[run_rollout_script] executing: ${CMD}" | tee -a "${STREAM_LOG}"

set +e
${CMD} 2>&1 | tee -a "${STREAM_LOG}"
STATUS=$?
set -e

# Update latest symlink (or copy if symlink unsupported)
ln -sf "${STREAM_LOG}" "${LATEST_LOG}" 2>/dev/null || cp -f "${STREAM_LOG}" "${LATEST_LOG}"

echo "[run_rollout_script] exit_status=${STATUS}" | tee -a "${STREAM_LOG}"
echo "[run_rollout_script] stream_log=${STREAM_LOG}" | tee -a "${STREAM_LOG}"
echo "[run_rollout_script] latest_log=${LATEST_LOG}" | tee -a "${STREAM_LOG}"

exit ${STATUS}


