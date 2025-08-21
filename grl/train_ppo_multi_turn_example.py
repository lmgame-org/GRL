"""
Script version of the Jupyter notebook (tunix_ppo_multi_turn_example.ipynb),
with cells preserved in order. Jupyter magics and shell commands are
commented out for Python execution.
"""

# ======================= Imports =======================
# Model definitions and parameter loading (Tunix/Qwen2)
from tunix.models.qwen2 import params
from tunix.models.qwen2 import model

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
multi_turn_cfg.rollout.agent_group_num = [4]
multi_turn_cfg.rollout.agent_group_size = [8]
# Limit turns for faster iteration
# multi_turn_cfg.simpleSokobanAgent.agent_config.max_turns = 3

# --- PPO configuration ---
# PPO hyperparameters used by Tunix PPO
NUM_PPO_EPOCHS = 1
MINI_BATCH_SIZE = 1
GAMMA = 1.0
GAE_LAMBDA = 0.95
BETA = 0.0  # Disable KL to reduce memory
EPSILON = 0.2
VF_COEF = 0.1
CLIP_RANGE_VALUE = 0.2

# --- Cluster / trainer / rollout configuration ---
# Sharding (fsdp, tp) — adjust to available devices
MESH = [(1, 2), ("fsdp", "tp")]

# Rollout (GRPO generation) parameters
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS =  256
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50

# Training loop setup
BATCH_SIZE = 1
NUM_BATCHES = 200
NUM_TEST_BATCHES = 100  # not used in this script but kept for completeness
EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * TRAIN_FRACTION * NUM_EPOCHS)

# Optimizer/scheduler
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1

# Checkpointing
INTERMEDIATE_CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/intermediate_ckpt/"
CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

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
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
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


# Critic model (required for PPO): initialize as a fresh Qwen2 and load
# the same weights as the reference to start
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

def get_critic_model(_model_config, _ref_model: nnx.Module, _mesh) -> nnx.Module:
  """Builds a critic from ref backbone and shards it to the provided mesh."""
  # Create abstract Qwen2 graph and merge ref weights for the backbone
  abs_mod: nnx.Module = nnx.eval_shape(
      lambda: model.Qwen2(_model_config, rngs=nnx.Rngs(params=0))
  )
  graph_def, _ = nnx.split(abs_mod)
  ref_state = nnx.state(_ref_model)
  backbone = nnx.merge(graph_def, ref_state)
  critic = Qwen2CriticWithScoreHead(backbone, rngs=nnx.Rngs(params=0))

  # Shard critic state consistently with the mesh
  crit_graph_def, crit_state = nnx.split(critic)
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
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo", flush_every_n_steps=20
)

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
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
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=1,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

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


