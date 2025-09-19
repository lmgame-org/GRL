#!/bin/bash

# Train script for multi-turn PPO (experimental)
# Runs grl/tunix_ppo_train.py and logs output to cache/

set -euo pipefail

# Setup project root (script is in project root)
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Setup logging
mkdir -p cache
LOG_FILE="cache/train_ppo_multi_turn_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"


PY_SCRIPT="$PROJECT_ROOT/grl/tunix_ppo_train.py"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "Error: Tunix PPO train script not found: $PY_SCRIPT" | tee -a "$LOG_FILE"
  exit 1
fi

python "$PY_SCRIPT" 2>&1 | tee "$LOG_FILE"

echo "Training completed. Log: $LOG_FILE"

