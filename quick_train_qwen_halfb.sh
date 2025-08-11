#!/bin/bash

# Quick Training with Qwen 0.5B - Configurable Parameters
# Uses base.yaml with parameter overrides

# ------ Configurable Parameters ------
CUDA_VISIBLE_DEVICES=${1:-"0"}
AGENT_GROUP_NUM=${2:-"8"}
AGENT_GROUP_SIZE=${3:-"16"}
VALIDATION_AGENT_GROUP_NUM=${4:-"64,64,64,64,64"}
VALIDATION_AGENT_GROUP_SIZE=${5:-"1,1,1,1,1"}
TRAINING_TASKS=${6:-"simpleSokobanAgent"}
VALIDATION_TASKS=${7:-"simpleSokobanAgent,largeSokobanAgent,gsm8kAgent_single_turn,blocksworldAgent_2d_sparse,tetrisAgent_type_1_dim_4"}
N_GPUS_PER_NODE=${8:-1}
PROJECT_NAME=${9:-"lmgame_train"}
EXPERIMENT_NAME=${10:-"quick_train_qwen_halfb_$(date +"%Y%m%d_%H%M%S")"}
MODEL_PATH=${11:-"Qwen/Qwen2.5-0.5B-Instruct"}
# Training control
TOTAL_TRAINING_STEPS=${12:-200}  # default 200 for Sokoban quick run

echo "=== Quick Training with Qwen 0.5B ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Agent Group Num: [$AGENT_GROUP_NUM]"
echo "Agent Group Size: [$AGENT_GROUP_SIZE]"
echo "Training tasks: $TRAINING_TASKS"
echo "Validation tasks: $VALIDATION_TASKS"
echo "N_GPUs per node: $N_GPUS_PER_NODE"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Model path: $MODEL_PATH"

# Setup project root (script is in project root)
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Setup logging
mkdir -p cache
LOG_FILE="cache/train_qwen_half_b_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

# Set environment and run training
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
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  model_path="$MODEL_PATH" \
  2>&1 | tee "$LOG_FILE"

echo "Training completed. Log: $LOG_FILE"

# ------ Usage Examples ------
# Default: ./quick_train_qwen_halfb.sh
# Custom GPUs: ./quick_train_qwen_halfb.sh "0,1"
# Custom agent groups: ./quick_train_qwen_halfb.sh "0" "16" "32"
# Custom tasks: ./quick_train_qwen_halfb.sh "0" "8" "16" "64,64" "1,1" "simpleSokobanAgent" "simpleSokobanAgent,largeSokobanAgent"
# Full custom: ./quick_train_qwen_halfb.sh "0" "8" "16" "64,64,64,64" "1,1,1,1" "simpleSokobanAgent" "simpleSokobanAgent,largeSokobanAgent" 1 "my_project" "my_experiment" "Qwen/Qwen2.5-0.5B-Instruct" 200