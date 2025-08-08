#!/bin/bash

# Sokoban PPO Training with Qwen 7B
# Configurable script with key training parameters

# ------ Configurable Parameters ------
CUDA_VISIBLE_DEVICES=${1:-"0,1,2,3"}
AGENT_GROUP_NUM=${2:-"8"}
AGENT_GROUP_SIZE=${3:-"16"}
VALIDATION_AGENT_GROUP_NUM=${4:-"256,256,256,256,256,256,256,256,256,256,256,256"}
VALIDATION_AGENT_GROUP_SIZE=${5:-"1,1,1,1,1,1,1,1,1,1,1,1"}
TRAINING_TASKS=${6:-"simpleSokobanAgent"}
VALIDATION_TASKS=${7:-"simpleSokobanAgent,largeSokobanAgent,gsm8kAgent_single_turn,gsm8kAgent_5_turn,blocksworldAgent_text,blocksworldAgent_1d,blocksworldAgent_2d_sparse,tetrisAgent_type_1_dim_4,tetrisAgent_type_2_dim_3,tetrisAgent_type_2_dim_4,webshopAgent,birdAgent"}
N_GPUS_PER_NODE=${8:-4}
PROJECT_NAME=${9:-"lmgame_train"}
EXPERIMENT_NAME=${10:-"sokoban_qwen_7b_$(date +"%Y%m%d_%H%M%S")"}
MODEL_PATH=${11:-"Qwen/Qwen2.5-7B-Instruct"}

echo "=== Sokoban PPO Training with Qwen 7B ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Agent Group Num: [$AGENT_GROUP_NUM]"
echo "Agent Group Size: [$AGENT_GROUP_SIZE]"
echo "Validation Agent Group Num: [$VALIDATION_AGENT_GROUP_NUM]"
echo "Validation Agent Group Size: [$VALIDATION_AGENT_GROUP_SIZE]"
echo "Training tasks: $TRAINING_TASKS"
echo "Validation tasks: $VALIDATION_TASKS"
echo "N_GPUs per node: $N_GPUS_PER_NODE"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Model path: $MODEL_PATH"

# ------ Setup ------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Setup logging
mkdir -p cache
LOG_FILE="cache/sokoban_qwen_7b_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

# ------ Environment Setup ------
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Convert parameters to hydra arrays
IFS=',' read -ra TRAINING_ARRAY <<< "$TRAINING_TASKS"
IFS=',' read -ra VALIDATION_ARRAY <<< "$VALIDATION_TASKS"
IFS=',' read -ra VAL_GROUP_NUM_ARRAY <<< "$VALIDATION_AGENT_GROUP_NUM"
IFS=',' read -ra VAL_GROUP_SIZE_ARRAY <<< "$VALIDATION_AGENT_GROUP_SIZE"

TRAINING_OVERRIDE="rollout.training=[$(printf "\"%s\"," "${TRAINING_ARRAY[@]}" | sed 's/,$//')]"
VALIDATION_OVERRIDE="rollout.validation=[$(printf "\"%s\"," "${VALIDATION_ARRAY[@]}" | sed 's/,$//')]"
VAL_GROUP_NUM_OVERRIDE="rollout.validation_agent_group_num=[$(printf "%s," "${VAL_GROUP_NUM_ARRAY[@]}" | sed 's/,$//')]"
VAL_GROUP_SIZE_OVERRIDE="rollout.validation_agent_group_size=[$(printf "%s," "${VAL_GROUP_SIZE_ARRAY[@]}" | sed 's/,$//')]"

# ------ Run Training ------
python lmgamerl/train.py \
  --config-name "base" \
  system.CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "rollout.agent_group_num=[$AGENT_GROUP_NUM]" \
  "rollout.agent_group_size=[$AGENT_GROUP_SIZE]" \
  "$VAL_GROUP_NUM_OVERRIDE" \
  "$VAL_GROUP_SIZE_OVERRIDE" \
  "$TRAINING_OVERRIDE" \
  "$VALIDATION_OVERRIDE" \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  model_path="$MODEL_PATH" \
  2>&1 | tee "$LOG_FILE"

echo "Training completed. Log: $LOG_FILE"

# ------ Usage Examples ------
# Default: ./qwen_7b.sh
# Custom GPUs: ./qwen_7b.sh "0,1"
# Custom agent groups: ./qwen_7b.sh "0,1,2,3" "16" "32"
# Custom validation groups: ./qwen_7b.sh "0,1,2,3" "8" "16" "512,512" "1,1"
# Custom tasks: ./qwen_7b.sh "0,1,2,3" "8" "16" "256,256" "1,1" "simpleSokobanAgent" "simpleSokobanAgent,largeSokobanAgent"
# Full custom: ./qwen_7b.sh "0,1,2,3" "8" "16" "256,256,256,256" "1,1,1,1" "simpleSokobanAgent" "simpleSokobanAgent,largeSokobanAgent" 4 "my_project" "my_experiment" "Qwen/Qwen2.5-7B-Instruct"
