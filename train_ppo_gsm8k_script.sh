#!/bin/bash

# Train script for PPO on GSM8K
# Runs grl/train_ppo_gsm8k_example.py and logs output to cache/ like quick_train_qwen_halfb.sh

set -euo pipefail

# Setup project root (script is in project root)
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Setup logging
mkdir -p cache
LOG_FILE="cache/train_ppo_gsm8k_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"


PY_SCRIPT="$PROJECT_ROOT/grl/train_ppo_gsm8k_example.py"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "Error: Python script not found: $PY_SCRIPT" | tee -a "$LOG_FILE"
  exit 1
fi

python "$PY_SCRIPT" 2>&1 | tee "$LOG_FILE"

echo "Training completed. Log: $LOG_FILE"

