"""
Script version of the Jupyter notebook (tunix_ppo_gsm8k_example.ipynb),
with cells preserved in order. Jupyter magics and shell commands are
commented out for Python execution.
"""

# ============================== Cell 0 ==============================
from tunix.models.qwen2 import params
from tunix.models.qwen2 import model
from flax import nnx
from huggingface_hub import snapshot_download
from etils import epath

MODEL_CP_PATH = "/home/vanitas/lmgame_projects/GRL/qwen_models"
repo_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Download safetensors locally (to MODEL_CP_PATH)
downloaded_path = snapshot_download(
    repo_id=repo_id,
    local_dir=MODEL_CP_PATH,
    allow_patterns=["*.safetensors", "*.json"],
)
print("Files downloaded to:", downloaded_path)

# Prefer the actual directory where files landed
model_dir = str(downloaded_path)

config = model.ModelConfig.qwen2_5_0_5_b()
# Only attempt load if .safetensors exist
if list(epath.Path(model_dir).expanduser().glob("*.safetensors")):
  qwen2 = params.create_model_from_safe_tensors(model_dir, config)
else:
  raise ValueError(f"No safetensors found in {model_dir}")
# nnx.display(qwen2)


# ============================== Cell 1 ==============================
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)


# ============================== Cell 2 ==============================
# The following were Jupyter shell/magic commands; commented for script use.
# !pip install -q tensorflow
# !pip install -q tensorboardX
# !pip install -q grain
# !pip install -q git+https://github.com/google/tunix
# !pip install -q git+https://github.com/google/qwix

# !pip uninstall -q -y flax
# !pip install -q git+https://github.com/google/flax.git

# !pip install -q datasets
# !pip install -q tensorflow_datasets


# ============================== Cell 3 ==============================
import functools
import gc
import os
from pprint import pprint
import time

from flax import nnx as _nnx  # avoid shadowing, though nnx already imported
import humanize
import jax
import jax.numpy as jnp
# import kagglehub
import optax
from orbax import checkpoint as ocp
from tqdm.auto import tqdm
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.ppo.ppo_learner import PpoConfig
from grl.trainer.tunix_agent_trainer import MultiTurnPpoLearner
from omegaconf import OmegaConf
from tunix.sft import metrics_logger
from pathlib import Path
from jax_smi import initialise_tracking
initialise_tracking()


# ============================== Cell 4 ==============================
"""Multi-turn training uses rollout-generated data only; no external dataset."""
TRAIN_DATA_DIR = None
TEST_DATA_DIR = None
TRAIN_FRACTION = 1.0

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
# Use single-device mesh to avoid device mismatch during bring-up/testing.
MESH = [(1, 2), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 4096
TOTAL_GENERATION_STEPS = 400
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50

# ====== PPO ======
# PPO hyperparameters (defaults from tunix.tunix.rl.ppo.ppo_learner.PpoConfig)
NUM_PPO_EPOCHS = 4
MINI_BATCH_SIZE = 1
GAMMA = 1.0
GAE_LAMBDA = 0.95
BETA = 0.04
EPSILON = 0.2
VF_COEF = 0.1
CLIP_RANGE_VALUE = 0.2
# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 200
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 100

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES  * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/intermediate_ckpt/"
CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


# ============================== Cell 5 ==============================
# ----- Utility functions -----
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

"""No text templates or parsing needed for multi-turn Sokoban rollout."""


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


# ============================== Cell 6 ==============================
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(qwen2)
checkpoint_path = os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state")
if not os.path.exists(checkpoint_path):
  checkpointer.save(os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state"), state)
time.sleep(60)
del qwen2
del state
gc.collect()


# ============================== Cell 7 ==============================
# ----- load models -----
def get_ref_model(ckpt_path):
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



# Reference model
qwen2_ref, mesh, model_config = get_ref_model(
    ckpt_path=os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state")
)
# nnx.display(qwen2_ref)



# Policy model (use original, no LoRA)
policy_qwen2 = qwen2_ref
# nnx.display(policy_qwen2)


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


# ============================== Cell 8 ==============================
"""No external reward functions: rollout provides rewards.

However, Tunix's PPO requires either a reward model or at least one reward
function to be provided at initialization. We provide a dummy reward function
that returns zeros just to satisfy the initialization constraint. The actual
rewards come from the multi-turn rollout in `MultiTurnPpoLearner`.
"""

def _dummy_reward_fn(prompts, completions, **kwargs):
  # Return zero reward per example; length must match batch size
  batch_size = len(completions) if completions is not None else len(prompts)
  return [0.0] * batch_size


# ============================== Cell 9 ==============================
"""No evaluation for multi-turn rollout-only training."""


# Use HF tokenizer already loaded earlier (AutoTokenizer.from_pretrained)
# and the original Qwen2 policy model (no LoRA)
qwen_tokenizer = tokenizer
# rollout_sampler = sampler_lib.Sampler(
#     transformer=policy_qwen2,
#     tokenizer=qwen_tokenizer,
#     cache_config=sampler_lib.CacheConfig(
#         cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
#         num_layers=model_config.num_layers,
#         num_kv_heads=model_config.num_kv_heads,
#         head_dim=model_config.head_dim,
#     ),
# )


# (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
#     test_dataset,
#     rollout_sampler,
#     **GENERATION_CONFIGS["greedy"],
# )
# print(
#     f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
#     f" {format_accuracy=}%"
# )


# ============================== Cell 10 ==============================
# ----- Training -----

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo", flush_every_n_steps=20
)


# Logs (Jupyter tensorboard magics commented out)
# %load_ext tensorboard
# %tensorboard --logdir /home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo --port=0

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

# Load multi-turn rollout config from configs/base.yaml and configs/agents.yaml
base_cfg = OmegaConf.load("/home/vanitas/lmgame_projects/GRL/configs/base.yaml")
agents_cfg = OmegaConf.load("/home/vanitas/lmgame_projects/GRL/configs/agents.yaml")
multi_turn_cfg = OmegaConf.merge(base_cfg, agents_cfg)
# Override rollout grouping for quicker testing
multi_turn_cfg.rollout.agent_group_num = [4]
multi_turn_cfg.rollout.agent_group_size = [1]
# Limit turns for faster iteration
multi_turn_cfg.simpleSokobanAgent.agent_config.max_turns = 3

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


# ============================== Cell 11 ==============================
# No final evaluation for multi-turn rollout-only training

