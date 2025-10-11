#!/usr/bin/env bash
set -euo pipefail
set -x

# Navigate to repo root if script is run from elsewhere
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

# Ensure required Python dependencies are present
python3 - <<'PY'
try:
    import datasets  # noqa: F401
    import pyarrow  # noqa: F401
except Exception as e:
    raise SystemExit("Missing dependencies. Please install: pip install datasets pyarrow")
PY

# Ensure local parquet data exists for GSM8K (download if missing), fail fast on error
python3 data/load_gsm8k.py --datasets gsm8k

# Paths (repo-local)
gsm8k_train_path="${REPO_ROOT}/data/gsm8k/train.parquet"
gsm8k_test_path="${REPO_ROOT}/data/gsm8k/test.parquet"

if [[ ! -f "${gsm8k_train_path}" || ! -f "${gsm8k_test_path}" ]]; then
  echo "GSM8K parquet not found after download. Please run: pip install -U datasets pyarrow && python3 data/load_gsm8k.py --overwrite" >&2
  exit 1
fi

train_files="['${gsm8k_train_path}']"
test_files="['${gsm8k_test_path}']"

# Use the Hydra entry in grl/verl_grpo_train.py which now points to configs/grpo_base.yaml
python3 -m grl.verl_grpo_train \
  algorithm.adv_estimator=grpo \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size=1024 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=verl_grpo_example_gsm8k_math \
  trainer.experiment_name=qwen2_7b_function_rm \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=15 \
  "$@"


