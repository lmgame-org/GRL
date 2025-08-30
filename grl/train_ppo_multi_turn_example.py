"""
Script version of the Jupyter notebook (tunix_ppo_multi_turn_example.ipynb),
with cells preserved in order. Jupyter magics and shell commands are
commented out for Python execution.
"""

# ======================= Imports =======================
# Model definitions and parameter loading (Tunix/Qwen2)
from tunix.models.qwen2 import params
from tunix.models.qwen2 import model
import wandb

# JAX/Flax core
import jax
import jax.numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp

# Optimizer/scheduler
import optax

# RL cluster and PPO trainer wrappers
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.ppo.ppo_learner import PpoConfig
from grl.trainer.tunix_agent_trainer import MultiTurnPpoLearner
from tunix.rl.utils import create_critic_model

# Config and metrics
from omegaconf import OmegaConf
from tunix.sft import metrics_logger

# Hugging Face IO and paths
from huggingface_hub import snapshot_download
from etils import epath
from transformers import AutoTokenizer

# Utilities and env
import gc
import os
import time
from pathlib import Path
import shutil
from jax_smi import initialise_tracking

initialise_tracking()

# ======================= Configuration =======================

# --- Model artifacts / data ---
MODEL_CP_PATH = "/home/vanitas/lmgame_projects/GRL/qwen_models"
repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA_DIR = None
TEST_DATA_DIR = None
TRAIN_FRACTION = 1.0

# --- Agent configuration (OmegaConf) ---
# Load multi-turn rollout config from configs/base.yaml and configs/agents.yaml
base_cfg = OmegaConf.load("/home/vanitas/lmgame_projects/GRL/configs/base.yaml")
agents_cfg = OmegaConf.load("/home/vanitas/lmgame_projects/GRL/configs/agents.yaml")
multi_turn_cfg = OmegaConf.merge(base_cfg, agents_cfg)
# Override rollout grouping for quicker testing
multi_turn_cfg.rollout.agent_group_num = [8]
multi_turn_cfg.rollout.agent_group_size = [16]
# Override validation rollout grouping
# Use only Sokoban for validation (exclude other environments)
multi_turn_cfg.rollout.validation = ["simpleSokobanAgent"]
multi_turn_cfg.rollout.validation_agent_group_num = [128]
multi_turn_cfg.rollout.validation_agent_group_size = [1]
# Limit turns for faster iteration
# multi_turn_cfg.simpleSokobanAgent.agent_config.max_turns = 3

# --- PPO configuration ---
# PPO hyperparameters used by Tunix PPO (aligned with YAML)
filter_ratio = 0.25
# Compute total agents = sum(agent_group_num[i] * agent_group_size[i]) and apply filter_ratio
try:
  group_nums = list(multi_turn_cfg.rollout.agent_group_num)
  group_sizes = list(multi_turn_cfg.rollout.agent_group_size)
except Exception:
  group_nums = [int(multi_turn_cfg.rollout.agent_group_num)]
  group_sizes = [int(multi_turn_cfg.rollout.agent_group_size)]
total_agents = sum(int(gn) * int(gs) for gn, gs in zip(group_nums, group_sizes))
TRAINING_BATCH_SIZE = max(1, int(total_agents * float(filter_ratio)))
NUM_PPO_EPOCHS = 1
MINI_BATCH_SIZE = 4
GAMMA = 1.0
GAE_LAMBDA = 1.0
BETA = 0.001  # base.yaml algorithm.kl_ctrl.kl_coef when use_kl_in_reward=True
EPSILON = 0.2
VF_COEF = 1.0
CLIP_RANGE_VALUE = 0.5  # ppo_trainer.yaml critic.cliprange_value

# --- Cluster / trainer / rollout configuration ---
# Sharding (fsdp, tp) — adjust to available devices
MESH = [(2, 2), ("fsdp", "tp")]
# Use integer accumulation; at least 1
GRADIENT_ACCUMULATION_STEPS = max(1, (TRAINING_BATCH_SIZE + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)

# Rollout (GRPO generation) parameters (aligned with YAML)
# Max Prompt Length: 4096
# Max Generation Steps: 400
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS =  100
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = None

# Training loop setup
NUM_BATCHES = int(200 * TRAINING_BATCH_SIZE / MINI_BATCH_SIZE)
EVAL_EVERY_N_STEPS = int(10 * TRAINING_BATCH_SIZE / MINI_BATCH_SIZE)
# Debug validation use small batch size
# NUM_BATCHES = 20
# EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * TRAIN_FRACTION * NUM_EPOCHS)
CPU_OFFLOAD = False

# Optimizer/scheduler
ACTOR_LR = 1e-6
CRITIC_LR = 1e-5
B1 = 0.9
B2 = 0.999
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0

# Checkpointing
INTERMEDIATE_CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/intermediate_ckpt/"
CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 1

# Inference presets (optional)
GENERATION_CONFIGS = {
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


# ======================= Prepare dummy reward and datasets =======================


# ----- Get dataset -----
def get_dataset(_: str | None, split: str = "train"):
  # For multi-turn rollouts, return a lightweight empty iterator of fixed length
  del _
  del split
  class _Empty:
    def __iter__(self):
      for _ in range(NUM_BATCHES):
        yield {}
    def __getitem__(self, idx):
      return {}
    def __len__(self):
      return NUM_BATCHES
  return _Empty()

dataset = get_dataset(TRAIN_DATA_DIR, "train")

# ----- Get dummy reward function -----
def _dummy_reward_fn(prompts, completions, **kwargs):
  # Return zero reward per example; length must match batch size
  batch_size = len(completions) if completions is not None else len(prompts)
  return [0.0] * batch_size



# ======================= Prepare Policy Models & Critic Models =======================

def download_model_weights(repo_id: str, local_dir: str) -> str:
  """Download model weights to local_dir and return the resolved path."""
  downloaded = snapshot_download(
      repo_id=repo_id,
      local_dir=local_dir,
      allow_patterns=["*.safetensors", "*.json"],
  )
  print("Files downloaded to:", downloaded)
  return str(downloaded)


def load_qwen2_from_safetensors(model_dir: str, model_config) -> nnx.Module:
  """Load Qwen2 from local safetensors directory."""
  if list(epath.Path(model_dir).expanduser().glob("*.safetensors")):
    return params.create_model_from_safe_tensors(model_dir, model_config)
  raise ValueError(f"No safetensors found in {model_dir}")


def save_intermediate_state(module: nnx.Module, save_dir: str) -> None:
  """Save an intermediate nnx state checkpoint once if it doesn't exist."""
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(module)
  checkpoint_path = os.path.join(Path(save_dir), "state")
  if not os.path.exists(checkpoint_path):
    checkpointer.save(checkpoint_path, state)
    # Ensure filesystem settles before continuing (matches original behavior)
    time.sleep(60)


def build_reference_model_from_ckpt(ckpt_path: str):
  """Restore reference model and return (model, mesh, model_config)."""
  mesh = jax.make_mesh(*MESH)
  model_config = model.ModelConfig.qwen2_5_0_5_b()
  abs_qwen2: nnx.Module = nnx.eval_shape(
      lambda: model.Qwen2(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_qwen2)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_qwen2)
  qwen2_ref = nnx.merge(graph_def, restored_params)
  return qwen2_ref, mesh, model_config


# 1) Download weights and load base model, then save an intermediate state
model_config = model.ModelConfig.qwen2_5_0_5_b()
model_dir = download_model_weights(repo_id, MODEL_CP_PATH)
qwen2 = load_qwen2_from_safetensors(model_dir, model_config)
save_intermediate_state(qwen2, INTERMEDIATE_CKPT_DIR)
del qwen2
gc.collect()

# 2) Build reference/policy and critic models
qwen2_ref, mesh, model_config = build_reference_model_from_ckpt(
    os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state")
)
policy_qwen2 = qwen2_ref
tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)


"""
At this point, we have:
- policy_qwen2: the policy/reference model (same weights initially)
- mesh: named device mesh for sharding
- tokenizer: loaded from local MODEL_CP_PATH
"""


def get_critic_model(_model_config, _ref_model: nnx.Module, _mesh) -> nnx.Module:
  """Build a critic; prefer replacing final head when present, else add score head.

  - If actor has a distinct `lm_head`, use `create_critic_model` to replace it with a scalar head.
  - If actor uses tied embeddings (no `lm_head`), clone backbone and append a `score` head.
  """
  head_name_to_replace = "output_proj"
  try:
    critic = create_critic_model(actor_model=_ref_model, seed=0, lm_head_to_replace=head_name_to_replace)
    print(f"[critic] Using replacement method: replaced '{head_name_to_replace}' with scalar value head.")
  except AttributeError:
    # Fallback for tied-embedding models: add a score head on top of cloned backbone
    print(
        f"[critic] Replacement head '{head_name_to_replace}' not found; "
        f"falling back to score-head on final hidden states. "
        f"tied_embedding={getattr(_model_config, 'use_tied_embedding', None)}"
    )
    class Qwen2CriticWithScoreHead(nnx.Module):
      def __init__(self, backbone: nnx.Module, rngs: nnx.Rngs):
        self.backbone = backbone
        hidden_dim = getattr(self.backbone.config, 'embed_dim')
        self.score = nnx.Linear(in_features=hidden_dim, out_features=1, use_bias=False, rngs=rngs)

      def __call__(self, input_tokens, positions, cache, attention_mask, **kwargs):
        _ = self.backbone(
            input_tokens,
            positions=positions,
            cache=cache,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = nnx.pop(self.backbone, nnx.Intermediate)['all_hidden_states'].value
        if isinstance(hidden, (list, tuple)):
          hidden = hidden[-1]
        score = self.score(hidden)
        return score

    # Clone weights into a fresh backbone (no sharing with actor/reference)
    abs_mod: nnx.Module = nnx.eval_shape(
        lambda: model.Qwen2(_model_config, rngs=nnx.Rngs(params=0))
    )
    graph_def, _ = nnx.split(abs_mod)
    ref_state = nnx.state(_ref_model)
    backbone = nnx.merge(graph_def, ref_state)
    critic = Qwen2CriticWithScoreHead(backbone, rngs=nnx.Rngs(params=0))
    print("[critic] Fallback method active: appended 'score' head over final hidden states.")

  # Shard critic state consistently with the mesh
  crit_graph_def, crit_state = nnx.split(critic)
  # Ensure critic params are bf16 to match actor/reference and reduce memory
  try:
    crit_state = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x,
        crit_state,
    )
  except Exception:
    pass
  crit_sharding = nnx.get_named_sharding(crit_state, _mesh)
  crit_state = jax.tree.map(lambda x, s: jax.device_put(x, s), crit_state, crit_sharding)
  return nnx.merge(crit_graph_def, crit_state)

critic_qwen2 = get_critic_model(model_config, qwen2_ref, mesh)


# ============================== initialize optimizer, rl_cluster, ppo_trainer =======================
qwen_tokenizer = tokenizer


# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
# Ensure a single TensorBoard log directory is cleaned per run
TB_LOG_DIR = "/home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo"
# Close any stray W&B run from previous initializations in-process
try:
  wandb.finish()
except Exception:
  pass

# Reset checkpoint and tensorboard directories to avoid auto-restore/mixing logs
def _reset_dir(path: str):
  if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True)
  os.makedirs(path, exist_ok=True)

_reset_dir(CKPT_DIR)
_reset_dir(TB_LOG_DIR)
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_LOG_DIR, flush_every_n_steps=20
)

# Compute effective optimizer steps based on rollout size, filtering and mini-batching
# (Commented out to align strictly with YAML; use MAX_STEPS and no warmup)
# try:
#   total_agents = sum(int(gn) * int(gs) for gn, gs in zip(
#       multi_turn_cfg.rollout.agent_group_num,
#       multi_turn_cfg.rollout.agent_group_size,
#   ))
#   kept_agents = max(1, int(total_agents * float(multi_turn_cfg.rollout.rollout_filter_ratio)))
# except Exception:
#   # Fallback to a reasonable default if config is missing
#   kept_agents = 32
# EFFECTIVE_STEPS = int(NUM_BATCHES * (kept_agents / max(1, MINI_BATCH_SIZE)) * NUM_PPO_EPOCHS)
# WARMUP_STEPS = 0
DECAY_STEPS = max(1, int(MAX_STEPS))

# Optimizers, learning rate schedulers, gradient clipping (split actor/critic)
actor_optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=ACTOR_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
critic_optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=CRITIC_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  actor_optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      actor_optimizer,
  )
  critic_optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      critic_optimizer,
  )


# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.CRITIC: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=CPU_OFFLOAD,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps= GRADIENT_ACCUMULATION_STEPS,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config={
        rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        ),
        rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        ),
    },
)

# todo: add per-mode rollout config for training and evaluation (especially for validation)

ppo_config = PpoConfig(
    num_ppo_epochs=NUM_PPO_EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    beta=BETA,
    epsilon=EPSILON,
    vf_coef=VF_COEF,
    clip_range_value=CLIP_RANGE_VALUE,
)
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=policy_qwen2,
    critic=critic_qwen2,
    reference=qwen2_ref,
    tokenizer=qwen_tokenizer,
    cluster_config=cluster_config,
)
# Multi-turn PPO Trainer
ppo_trainer = MultiTurnPpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=_dummy_reward_fn,
    ppo_config=ppo_config,
    multi_turn_cfg=multi_turn_cfg,
    multi_turn_processor=None,
    multi_turn_validation=False,
)

with mesh:
    ppo_trainer.train(dataset)


try:
    wandb.finish()
except Exception:
    pass


