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
from grl.trainer.tunix_agent_trainer_exp import PpoConfigExp, PpoLearnerExp
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

# Load Tunix base config for hyperparameters (relative paths)
BASE_DIR = Path(__file__).resolve().parents[1]
tunix_cfg = OmegaConf.load(str(BASE_DIR / "configs" / "tunix_base.yaml"))

# --- Core PPO hyperparameters (centralized) ---
ENTROPY_COEFF = float(tunix_cfg.ppo.entropy_coeff)
ENTROPY_AGGS_MODE = str(tunix_cfg.ppo.aggs_mode)

# --- Model artifacts / data ---
MODEL_CP_PATH = str(BASE_DIR / "qwen_models")
repo_id = str(tunix_cfg.model.repo_id)
TRAIN_DATA_DIR = None
TEST_DATA_DIR = None
TRAIN_FRACTION = 1.0

# --- Agent configuration (OmegaConf) ---
# Use Tunix base YAML only as the single source of truth; compose defaults if present
def _compose_with_defaults(cfg):
  out = cfg
  try:
    df = list(cfg.get("defaults", []))
  except Exception:
    df = []
  for item in df:
    if item == "agents":
      agents_path = BASE_DIR / "configs" / "agents.yaml"
      try:
        _agents = OmegaConf.load(str(agents_path))
        out = OmegaConf.merge(out, _agents)
      except Exception:
        pass
  return out

multi_turn_cfg = _compose_with_defaults(tunix_cfg)

# --- PPO configuration ---
# PPO hyperparameters used by Tunix PPO (from YAML)
filter_ratio = float(multi_turn_cfg.rollout.rollout_filter_ratio)
# Compute total agents = sum(agent_group_num[i] * agent_group_size[i]) and apply filter_ratio
try:
  group_nums = list(multi_turn_cfg.rollout.agent_group_num)
  group_sizes = list(multi_turn_cfg.rollout.agent_group_size)
except Exception:
  group_nums = [int(multi_turn_cfg.rollout.agent_group_num)]
  group_sizes = [int(multi_turn_cfg.rollout.agent_group_size)]
total_agents = sum(int(gn) * int(gs) for gn, gs in zip(group_nums, group_sizes))
TRAINING_BATCH_SIZE = max(1, int(total_agents * float(filter_ratio)))
NUM_PPO_EPOCHS = int(tunix_cfg.ppo.num_ppo_epochs)
MINI_BATCH_SIZE = int(tunix_cfg.ppo.mini_batch_size)
GAMMA = float(tunix_cfg.ppo.gamma)
GAE_LAMBDA = float(tunix_cfg.ppo.gae_lambda)
BETA = float(tunix_cfg.ppo.beta)
EPSILON = float(tunix_cfg.ppo.epsilon)
VF_COEF = float(tunix_cfg.ppo.vf_coef)
CLIP_RANGE_VALUE = float(tunix_cfg.ppo.clip_range_value)

# ===== Adjustable PPO clipping hyperparameters (moved to top) =====
CLIP_RATIO_LOW = float(tunix_cfg.ppo.clip_ratio_low)
CLIP_RATIO_HIGH = float(tunix_cfg.ppo.clip_ratio_high)
CLIP_RATIO_C = float(tunix_cfg.ppo.clip_ratio_c)
kl_penalty_method = str(tunix_cfg.ppo.kl_penalty_method)

# --- Cluster / trainer / rollout configuration ---
# Sharding (fsdp, tp) — adjust to available devices
MESH = [(2, 2), ("fsdp", "tp")]
# Use integer accumulation; at least 1
# GRADIENT_ACCUMULATION_STEPS = max(1, (TRAINING_BATCH_SIZE + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
GRADIENT_ACCUMULATION_STEPS = int(tunix_cfg.training.gradient_accumulation_steps)

# Rollout (GRPO generation) parameters (aligned with YAML)
# Max Prompt Length: 4096
# Max Generation Steps: 400
MAX_PROMPT_LENGTH = int(tunix_cfg.rollout_runtime.max_prompt_length)
TOTAL_GENERATION_STEPS = int(tunix_cfg.rollout_runtime.total_generation_steps)
TEMPERATURE = float(tunix_cfg.rollout_runtime.temperature_train)
EVAL_TEMPERATURE = float(tunix_cfg.rollout_runtime.temperature_eval)
TOP_P = float(tunix_cfg.rollout_runtime.top_p)
TOP_K = None if tunix_cfg.rollout_runtime.top_k is None else int(tunix_cfg.rollout_runtime.top_k)

# Training loop setup
NUM_BATCHES = int(200 * TRAINING_BATCH_SIZE / MINI_BATCH_SIZE)
# Use YAML value if > 0, else fallback
_ee = int(tunix_cfg.training.eval_every_n_steps)
EVAL_EVERY_N_STEPS = _ee if _ee and _ee > 0 else int(10 * TRAINING_BATCH_SIZE / MINI_BATCH_SIZE)
# Debug validation use small batch size
# NUM_BATCHES = 20
# EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 1
_max_steps_cfg = int(tunix_cfg.training.max_steps)
MAX_STEPS = _max_steps_cfg if _max_steps_cfg and _max_steps_cfg > 0 else int(NUM_BATCHES * TRAIN_FRACTION * NUM_EPOCHS)
CPU_OFFLOAD = False

# Optimizer/scheduler
ACTOR_LR = float(tunix_cfg.training.actor_lr)
CRITIC_LR = float(tunix_cfg.training.critic_lr)
B1 = float(tunix_cfg.training.b1)
B2 = float(tunix_cfg.training.b2)
WEIGHT_DECAY = float(tunix_cfg.training.weight_decay)
MAX_GRAD_NORM = float(tunix_cfg.training.max_grad_norm)

# Checkpointing (compose absolute paths from repo root; allow env overrides)
RUN_ROOT = (BASE_DIR / "content").resolve()
_default_intermediate = (RUN_ROOT / "intermediate_ckpt").resolve()
_default_ckpts = (RUN_ROOT / "ckpts").resolve()
INTERMEDIATE_CKPT_DIR = os.environ.get("GRL_INTERMEDIATE_CKPT_DIR", str(_default_intermediate))
CKPT_DIR = os.environ.get("GRL_CKPT_DIR", str(_default_ckpts))
SAVE_INTERVAL_STEPS = int(tunix_cfg.training.save_interval_steps)
MAX_TO_KEEP = int(tunix_cfg.training.max_to_keep)
print("Checkpoint dirs:", {"intermediate": INTERMEDIATE_CKPT_DIR, "ckpts": CKPT_DIR})

# Inference presets removed for brevity


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

# ======================= Print Config Summary =======================
def _print_config_summary():
  def _to_dict(cfg):
    try:
      return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
      return cfg
  print("===== Tunix Config (merged) =====")
  print(_to_dict(tunix_cfg))
  print("===== Agents Config =====")
  print(_to_dict(agents_cfg))
  print("===== Multi-turn (merged) rollout keys =====")
  print(_to_dict(multi_turn_cfg.rollout))
  print("===== PPO =====")
  print(_to_dict(tunix_cfg.ppo))
  print("===== Training =====")
  print(_to_dict(tunix_cfg.training))
  print("===== Rollout Runtime =====")
  print(_to_dict(tunix_cfg.rollout_runtime))

_print_config_summary()



# ======================= Prepare Policy Models & Critic Models =======================

def download_model_weights(repo_id: str, local_dir: str) -> str:
  """Download model weights and tokenizer assets; return the resolved path."""
  downloaded = snapshot_download(
      repo_id=repo_id,
      local_dir=local_dir,
      allow_patterns=[
        "*.safetensors",
        "*.json",
        "tokenizer.*",
        "*.model",
        "vocab*",
        "merges.txt",
      ],
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


def clone_module_like(src_module: nnx.Module, model_config, mesh) -> nnx.Module:
  """Create a separate nnx.Module instance with the same parameters and sharding.

  Ensures the returned module is a distinct Python object so optimizer updates on
  the actor do not affect the frozen reference.
  """
  abs_mod: nnx.Module = nnx.eval_shape(lambda: model.Qwen2(model_config, rngs=nnx.Rngs(params=0)))
  gdef, _ = nnx.split(abs_mod)
  src_state = nnx.state(src_module)
  # Best-effort: ensure arrays are placed on the provided mesh sharding
  try:
    target_sharding = nnx.get_named_sharding(src_state, mesh)
    src_state = jax.tree.map(lambda x, s: jax.device_put(x, s), src_state, target_sharding)
  except Exception:
    pass
  return nnx.merge(gdef, src_state)


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
policy_qwen2 = clone_module_like(qwen2_ref, model_config, mesh)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tokenizer.pad_token_id is None:
  tokenizer.pad_token = tokenizer.eos_token
# TODO: Maybe padding issue in trainer
print("eos_id:", tokenizer.eos_token_id, "pad_id:", tokenizer.pad_token_id)


"""
At this point, we have:
- policy_qwen2: the policy/reference model (same weights initially)
- mesh: named device mesh for sharding
- tokenizer: loaded from local MODEL_CP_PATH
"""


# """
# # ===== Helpers: NNX head materialization & replacement =====
# """
#
# def _dict_paths(d, prefix=()):
#   """Yield (path_tuple, leaf) for nested dict-likes in an NNX state."""
#   if isinstance(d, dict):
#     for k, v in d.items():
#       yield from _dict_paths(v, prefix + (k,))
#   else:
#     yield prefix, d
#
# def _find_embedding_path(nn_state):
#   """
#   Heuristic: find embedding weight (shape ~ [vocab, hidden]) under keys containing
#   'embed' and 'embedding' (e.g., state['backbone']['embed_tokens']['embedding']).
#   Returns a tuple path or None.
#   """
#   candidates = []
#   for path, leaf in _dict_paths(nn_state):
#     if not hasattr(leaf, "shape"):
#       continue
#     key_str = "/".join(map(str, path)).lower()
#     if "embed" in key_str and "embedding" in key_str and len(getattr(leaf, "shape", ())) == 2:
#       candidates.append((path, leaf.shape))
#   # prefer the largest vocab dimension
#   if not candidates:
#     return None
#   candidates.sort(key=lambda x: x[1][0], reverse=True)
#   return candidates[0][0]
#
# class Qwen2WithHead(nnx.Module):
#   """
#   Minimal wrapper that (1) exposes a real output head and (2) forwards through backbone
#   with output_hidden_states=True so we can attach heads to the last hidden layer.
#   """
#   def __init__(self, backbone: nnx.Module, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
#     self.backbone = backbone
#     self.output_proj = nnx.Linear(in_features=hidden_dim, out_features=out_dim, use_bias=False, rngs=rngs)
#
#   def __call__(self, input_tokens, positions, cache, attention_mask, **kwargs):
#     _ = self.backbone(
#       input_tokens,
#       positions=positions,
#       cache=cache,
#       attention_mask=attention_mask,
#       output_hidden_states=True,
#       **kwargs,
#     )
#     h_all = nnx.pop(self.backbone, nnx.Intermediate)['all_hidden_states'].value
#     if isinstance(h_all, (list, tuple)):
#       h = h_all[-1]
#     else:
#       h = h_all
#     return self.output_proj(h)
#
# def _materialize_head_from_tied_embedding(backbone_module: nnx.Module, mesh, hidden_dim: int, vocab_size: int):
#   """
#   Create Qwen2WithHead(backbone, H→V), initialize kernel with embed^T (de-tie),
#   and shard to mesh.
#   """
#   # Build wrapper
#   wrapper = Qwen2WithHead(backbone_module, hidden_dim, vocab_size, rngs=nnx.Rngs(params=0))
#   gdef, state = nnx.split(wrapper)
#
#   # Locate embedding in the backbone state
#   embed_path = _find_embedding_path(state)
#   if embed_path is None:
#     print("[critic] WARNING: embedding not found; initializing output head to zeros.")
#     kernel = jnp.zeros((hidden_dim, vocab_size), dtype=state['output_proj']['kernel'].dtype)
#   else:
#     embed = state
#     for key in embed_path:
#       embed = embed[key]
#     # embed is [V, H] -> kernel needs [H, V]
#     kernel = jnp.asarray(embed.T, dtype=state['output_proj']['kernel'].dtype)
#
#   state['output_proj']['kernel'] = kernel
#
#   # shard and merge
#   sharding = nnx.get_named_sharding(state, mesh)
#   state = jax.tree.map(lambda x, s: jax.device_put(x, s), state, sharding)
#   return nnx.merge(gdef, state)
#
# def _replace_output_head_with_value(critic_with_head: Qwen2WithHead, mesh, hidden_dim: int):
#   """
#   True replacement: keep the attribute name `output_proj`, but swap the module shape/type
#   to Linear(H→1). This mirrors Torch's `model.lm_head = nn.Linear(H,1)`.
#   """
#   # Split current module
#   gdef, st = nnx.split(critic_with_head)
#
#   # Replace module attribute at Python level
#   critic_with_head.output_proj = nnx.Linear(in_features=hidden_dim, out_features=1, use_bias=False, rngs=nnx.Rngs(params=0))
#
#   # Split again to get the new state tree (with shape [H,1])
#   gdef2, st2 = nnx.split(critic_with_head)
#
#   # Optional: initialize small (or zeros) for stability
#   st2['output_proj']['kernel'] = jnp.zeros_like(st2['output_proj']['kernel'])
#
#   # Shard and merge
#   shard2 = nnx.get_named_sharding(st2, mesh)
#   st2 = jax.tree.map(lambda x, s: jax.device_put(x, s), st2, shard2)
#   return nnx.merge(gdef2, st2)
#
# def get_critic_model(_model_config, _ref_model: nnx.Module, _mesh) -> nnx.Module:
#   """
#   Build a critic that replaces the output head (no append):
#     1) Clone backbone weights from _ref_model into a fresh module.
#     2) Materialize an untied output head (H→V) and init from tied embedding (embed^T).
#     3) Replace that head with a value head (H→1).
#     4) Return the merged/sharded critic module.
#   """
#   # 1) Clone the backbone to avoid parameter sharing with policy/ref
#   abs_mod: nnx.Module = nnx.eval_shape(lambda: model.Qwen2(_model_config, rngs=nnx.Rngs(params=0)))
#   gdef_abs, _ = nnx.split(abs_mod)
#   ref_state = nnx.state(_ref_model)
#   backbone = nnx.merge(gdef_abs, ref_state)
#
#   # 2) Materialize an explicit softmax head initialized from embedding^T
#   hidden_dim = getattr(backbone.config, "embed_dim")
#   vocab_size = getattr(backbone.config, "vocab_size")
#   critic_with_head = _materialize_head_from_tied_embedding(backbone, _mesh, hidden_dim, vocab_size)
#
#   # 3) True replacement: swap H→V head with a value head H→1
#   critic_value = _replace_output_head_with_value(critic_with_head, _mesh, hidden_dim)
#
#   print("[critic] Built with true head replacement: output_proj Linear(H→1).")
#   return critic_value
#
# critic_qwen2 = get_critic_model(model_config, qwen2_ref, mesh)

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
# Zero-initialize critic value head for stable start
_g_def, _state = nnx.split(critic_qwen2)
try:
  _state['score']['kernel'] = jnp.zeros_like(_state['score']['kernel'])
except Exception:
  pass
critic_qwen2 = nnx.merge(_g_def, _state)


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
# Scheduler config removed (unused)

# Optimizers, learning rate schedulers, gradient clipping (split actor/critic)
actor_optimizer = optax.adamw(
    learning_rate=optax.constant_schedule(ACTOR_LR),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
critic_optimizer = optax.adamw(
    learning_rate=optax.constant_schedule(CRITIC_LR),
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
            temperature=EVAL_TEMPERATURE,
            top_p=1.0,
            top_k=None,
        ),
    },
)

# todo: add per-mode rollout config for training and evaluation (especially for validation)

ppo_config = PpoConfigExp(
    num_ppo_epochs=NUM_PPO_EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    beta=BETA,
    epsilon=EPSILON,
    vf_coef=VF_COEF,
    clip_range_value=CLIP_RANGE_VALUE,
    # Entropy regularization
    entropy_coeff=ENTROPY_COEFF,
    aggs_mode=ENTROPY_AGGS_MODE,
    # ===== MODIFICATION: Asymmetric + dual-clip PPO hyperparameters =====
    clip_ratio_low=CLIP_RATIO_LOW,
    clip_ratio_high=CLIP_RATIO_HIGH,
    clip_ratio_c=CLIP_RATIO_C,
    kl_penalty_method=kl_penalty_method,
)
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=policy_qwen2,
    critic=critic_qwen2,
    reference=qwen2_ref,
    tokenizer=qwen_tokenizer,
    cluster_config=cluster_config,
)
# Multi-turn PPO Trainer (experimental)
ppo_trainer = PpoLearnerExp(
    rl_cluster=rl_cluster,
    ppo_config=ppo_config,
    reward_fns=_dummy_reward_fn,
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


