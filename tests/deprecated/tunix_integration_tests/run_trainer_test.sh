#!/usr/bin/env bash

set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure project root is on PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Log directory and file
LOG_DIR="${SCRIPT_DIR}/test_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/trainer_test_${TS}.log"

echo "=========================================="
echo "Trainer Test Runner"
echo "Started at: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Run the test and stream output to both console and log
python "${SCRIPT_DIR}/trainer_test.py" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "✅ Tests completed successfully"
else
  echo "❌ Tests failed with exit code ${EXIT_CODE}"
fi
echo "Saved log to: ${LOG_FILE}"
echo "Completed at: $(date)"
echo "=========================================="

exit ${EXIT_CODE}


