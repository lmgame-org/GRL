"""Tunix PPO training entrypoint.

Clear functional layout:
  1) Compose config (OmegaConf) and derive hyperparameters
  2) Initialize models (reference, actor, critic)
  3) Initialize RLCluster, logging, and trainer
  4) Orchestrate training in main()
"""

# ======================= Imports =======================
# Standard library
import gc
import os
import shutil
import time
from pathlib import Path

# Third-party
import wandb
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from etils import epath
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp
from transformers import AutoTokenizer
from hydra import main as hydra_main

# Local application
from tunix.models.qwen2 import model, params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from grl.trainer.tunix_agent_trainer import PpoConfigExp, PpoLearnerExp

from jax_smi import initialise_tracking

initialise_tracking()

# ======================= Globals (Config) =======================
BASE_DIR = Path(__file__).resolve().parents[1]


def derive_hparams(cfg):
  """Derive scalar hyperparameters and convenience values from config."""
  # Core PPO (trainer.ppo)
  entropy_coef = float(cfg.trainer.ppo.entropy_coef)

  # Rollout agent sizing
  filter_ratio = float(cfg.rollout.rollout_filter_ratio)
  try:
    group_nums = list(cfg.rollout.agent_group_num)
    group_sizes = list(cfg.rollout.agent_group_size)
  except Exception:
    group_nums = [int(cfg.rollout.agent_group_num)]
    group_sizes = [int(cfg.rollout.agent_group_size)]
  total_agents = sum(
      int(gn) * int(gs) for gn, gs in zip(group_nums, group_sizes)
  )
  training_batch_size = max(1, int(total_agents * filter_ratio))

  # PPO schedulers
  num_ppo_epochs = int(cfg.trainer.ppo.num_ppo_epochs)
  # Mini-batch size is part of cluster.training_config in new config
  mini_batch_size = int(cfg.cluster.training_config.mini_batch_size)
  gamma = float(cfg.trainer.ppo.gamma)
  gae_lambda = float(cfg.trainer.ppo.gae_lambda)
  beta = float(cfg.trainer.ppo.beta)
  epsilon = float(cfg.trainer.ppo.epsilon)
  clip_range_value = float(cfg.trainer.ppo.clip_range_value)
  epsilon_low = float(cfg.trainer.ppo.epsilon_low)
  epsilon_high = float(cfg.trainer.ppo.epsilon_high)
  epsilon_c = float(cfg.trainer.ppo.epsilon_c)

  # Mesh (cluster.mesh)
  try:
    mesh_shape = tuple(int(x) for x in cfg.cluster.mesh.shape)
    mesh_axes = tuple(str(x) for x in cfg.cluster.mesh.axes)
    mesh = [mesh_shape, mesh_axes]
  except Exception:
    mesh = [(2, 2), ("fsdp", "tp")]

  # Rollout runtime (cluster.rollout_config)
  train_rc = cfg.cluster.rollout_config.train
  eval_rc = cfg.cluster.rollout_config.eval
  max_prompt_length = int(train_rc.max_prompt_length)
  total_generation_steps = int(train_rc.total_generation_steps)
  temperature_train = float(train_rc.temperature)
  temperature_eval = float(eval_rc.temperature)
  top_p = float(train_rc.top_p)
  top_k = None if train_rc.top_k is None else int(train_rc.top_k)

  # Optimizer (trainer.optim)
  actor_lr = float(cfg.trainer.optim.actor_lr)
  critic_lr = float(cfg.trainer.optim.critic_lr)
  b1 = float(cfg.trainer.optim.b1)
  b2 = float(cfg.trainer.optim.b2)
  weight_decay = float(cfg.trainer.optim.weight_decay)
  max_grad_norm = float(cfg.trainer.optim.max_grad_norm)
  grad_accum = int(cfg.trainer.optim.gradient_accumulation_steps)
  try:
    optim_type = str(cfg.trainer.optim.type)
  except Exception:
    optim_type = "constant"

  # Training loop setup (cluster.training_config)
  num_batches = int(200 * training_batch_size / max(1, mini_batch_size))
  ee = int(cfg.cluster.training_config.eval_every_n_steps)
  eval_every_n_steps = ee if ee and ee > 0 else int(10 * grad_accum)
  num_epochs = 1
  max_steps_cfg = int(cfg.cluster.training_config.max_steps)
  train_fraction = 1.0
  max_steps = (
      max_steps_cfg
      if max_steps_cfg and max_steps_cfg > 0
      else int(num_batches * train_fraction * num_epochs)
  )
  cpu_offload = bool(cfg.cluster.offload_to_cpu)
  rollout_engine = str(cfg.cluster.rollout_engine)

  # Paths
  run_root = (BASE_DIR / "content").resolve()
  default_intermediate = (run_root / "intermediate_ckpt").resolve()
  default_ckpts = (run_root / "ckpts").resolve()
  intermediate_ckpt_dir = os.environ.get(
      "GRL_INTERMEDIATE_CKPT_DIR", str(default_intermediate)
  )
  ckpt_dir = os.environ.get("GRL_CKPT_DIR", str(default_ckpts))
  save_interval_steps = int(
      cfg.cluster.training_config.checkpoint.save_interval_steps
  )
  max_to_keep = int(cfg.cluster.training_config.checkpoint.max_to_keep)

  return {
      "entropy_coef": entropy_coef,
      "filter_ratio": filter_ratio,
      "group_nums": group_nums,
      "group_sizes": group_sizes,
      "total_agents": total_agents,
      "training_batch_size": training_batch_size,
      "num_ppo_epochs": num_ppo_epochs,
      "mini_batch_size": mini_batch_size,
      "gamma": gamma,
      "gae_lambda": gae_lambda,
      "beta": beta,
      "epsilon": epsilon,
      "clip_range_value": clip_range_value,
      "epsilon_low": epsilon_low,
      "epsilon_high": epsilon_high,
      "epsilon_c": epsilon_c,
      "mesh": mesh,
      "max_prompt_length": max_prompt_length,
      "total_generation_steps": total_generation_steps,
      "temperature_train": temperature_train,
      "temperature_eval": temperature_eval,
      "top_p": top_p,
      "top_k": top_k,
      "num_batches": num_batches,
      "eval_every_n_steps": eval_every_n_steps,
      "num_epochs": num_epochs,
      "max_steps": max_steps,
      "cpu_offload": cpu_offload,
      "rollout_engine": rollout_engine,
      "actor_lr": actor_lr,
      "critic_lr": critic_lr,
      "b1": b1,
      "b2": b2,
      "weight_decay": weight_decay,
      "max_grad_norm": max_grad_norm,
      "grad_accum": grad_accum,
      "optim_type": optim_type,
      "run_root": str(run_root),
      "intermediate_ckpt_dir": intermediate_ckpt_dir,
      "ckpt_dir": ckpt_dir,
      "save_interval_steps": save_interval_steps,
      "max_to_keep": max_to_keep,
  }


# Inference presets removed for brevity


# ======================= Dataset helpers =======================


def get_dataset(num_batches: int, batch_size: int, split: str = "train"):
  # For multi-turn rollouts, return a lightweight empty iterator of fixed length
  del split

  class _Empty:

    def __iter__(self):
      for _ in range(num_batches):
        # Minimal placeholder to satisfy trainer interfaces; actual data comes from rollout
        yield {"prompts": [""] * batch_size}

    def __getitem__(self, idx):
      return {"prompts": [""] * batch_size}

    def __len__(self):
      return num_batches

  return _Empty()


# ----- Get dummy reward function -----
def _dummy_reward_fn(prompts, completions, **kwargs):
  # Return zero reward per example; length must match batch size
  batch_size = len(completions) if completions is not None else len(prompts)
  return [0.0] * batch_size


# ======================= Print Config Summary =======================
def _print_config_summary(cfg, derived):
  try:
    merged = OmegaConf.to_container(cfg, resolve=True)
  except Exception:
    merged = cfg
  print("===== Hydra Config (resolved) =====")
  print(merged)
  print("===== Derived PPO/Training/Cluster/Rollout =====")
  print(
      {
          "trainer.ppo": {
              "num_ppo_epochs": derived["num_ppo_epochs"],
              "gamma": derived["gamma"],
              "gae_lambda": derived["gae_lambda"],
              "beta": derived["beta"],
              "epsilon": derived["epsilon"],
              "clip_range_value": derived["clip_range_value"],
              "entropy_coef": derived["entropy_coef"],
              "epsilon_low": derived["epsilon_low"],
              "epsilon_high": derived["epsilon_high"],
              "epsilon_c": derived["epsilon_c"],
          },
          "trainer.optim": {
              "gradient_accumulation_steps": derived["grad_accum"],
              "actor_lr": derived["actor_lr"],
              "critic_lr": derived["critic_lr"],
              "b1": derived["b1"],
              "b2": derived["b2"],
              "weight_decay": derived["weight_decay"],
              "max_grad_norm": derived["max_grad_norm"],
          },
          "cluster.training_config": {
              "max_steps": derived["max_steps"],
              "eval_every_n_steps": derived["eval_every_n_steps"],
              "save_interval_steps": derived["save_interval_steps"],
              "max_to_keep": derived["max_to_keep"],
              "rollout_engine": derived["rollout_engine"],
              "offload_to_cpu": derived["cpu_offload"],
          },
          "cluster.rollout_config": {
              "train": {
                  "max_prompt_length": derived["max_prompt_length"],
                  "total_generation_steps": derived["total_generation_steps"],
                  "temperature": derived["temperature_train"],
                  "top_p": derived["top_p"],
                  "top_k": derived["top_k"],
              },
              "eval": {
                  "max_prompt_length": derived["max_prompt_length"],
                  "total_generation_steps": derived["total_generation_steps"],
                  "temperature": derived["temperature_eval"],
                  "top_p": 1.0,
                  "top_k": None,
              },
          },
          "derived": {
              "total_agents": derived["total_agents"],
              "training_batch_size": derived["training_batch_size"],
              "num_batches": derived["num_batches"],
          },
      }
  )


# ======================= Model building helpers =======================
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


def build_reference_model_from_ckpt(ckpt_path: str, mesh):
  """Restore reference model and return (model, mesh, model_config)."""
  model_config = model.ModelConfig.qwen2_5_0_5b()
  with mesh:
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
  with mesh:
    abs_mod: nnx.Module = nnx.eval_shape(
        lambda: model.Qwen2(model_config, rngs=nnx.Rngs(params=0))
    )
  gdef, _ = nnx.split(abs_mod)
  src_state = nnx.state(src_module)
  # Best-effort: ensure arrays are placed on the provided mesh sharding
  try:
    target_sharding = nnx.get_named_sharding(src_state, mesh)
    src_state = jax.tree.map(
        lambda x, s: jax.device_put(x, s), src_state, target_sharding
    )
  except Exception:
    pass
  return nnx.merge(gdef, src_state)


def build_models_and_tokenizer(cfg, derived):
  """Download/load models, build reference/actor/critic, tokenizer and mesh."""
  model_cp_path = str(BASE_DIR / "qwen_models")
  repo_id = str(cfg.model.repo_id)
  model_dir = download_model_weights(repo_id, model_cp_path)
  mesh = jax.make_mesh(*derived["mesh"])  # [shape, axes]
  model_config = model.ModelConfig.qwen2_5_0_5b()
  with mesh:
    qwen2 = load_qwen2_from_safetensors(model_dir, model_config)
  save_intermediate_state(qwen2, derived["intermediate_ckpt_dir"])
  del qwen2
  gc.collect()

  qwen2_ref, mesh, model_config = build_reference_model_from_ckpt(
      os.path.join(Path(derived["intermediate_ckpt_dir"]), "state"), mesh
  )
  policy_qwen2 = clone_module_like(qwen2_ref, model_config, mesh)
  tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  print("eos_id:", tokenizer.eos_token_id, "pad_id:", tokenizer.pad_token_id)
  critic_qwen2 = get_critic_model(model_config, qwen2_ref, mesh)
  return policy_qwen2, critic_qwen2, qwen2_ref, tokenizer, mesh, model_config


"""
At this point, we have:
- policy_qwen2: the policy/reference model (same weights initially)
- mesh: named device mesh for sharding
- tokenizer: loaded from local MODEL_CP_PATH
"""


# --- Critic: simple Linear(H->1) head (no bias) ---
class Qwen2CriticTokenClass(nnx.Module):

  def __init__(self, backbone: nnx.Module, rngs: nnx.Rngs):
    self.backbone = backbone
    hidden_dim = getattr(
        self.backbone.config,
        "hidden_size",
        getattr(self.backbone.config, "embed_dim"),
    )
    self.classifier = nnx.Linear(
        in_features=hidden_dim, out_features=1, use_bias=False, rngs=rngs
    )

  def __call__(self, input_tokens, positions, cache, attention_mask, **kwargs):
    _ = self.backbone(
        input_tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        output_hidden_states=True,
        **kwargs,
    )
    h = nnx.pop(self.backbone, nnx.Intermediate)["all_hidden_states"].value
    if isinstance(h, (list, tuple)):
      h = h[-1]
    return self.classifier(h)


def get_critic_model(
    _model_config, _ref_model: nnx.Module, _mesh
) -> nnx.Module:
  """Builds a critic from ref backbone and shards it to the provided mesh."""
  with _mesh:
    abs_mod: nnx.Module = nnx.eval_shape(
        lambda: model.Qwen2(_model_config, rngs=nnx.Rngs(params=0))
    )
  graph_def, _ = nnx.split(abs_mod)
  ref_state = nnx.state(_ref_model)
  backbone = nnx.merge(graph_def, ref_state)

  critic = Qwen2CriticTokenClass(backbone, rngs=nnx.Rngs(params=0))
  crit_graph_def, crit_state = nnx.split(critic)

  # Initialization: small normal for weights (seeded)
  seed = 123
  key = jax.random.PRNGKey(seed)
  kshape = crit_state["classifier"]["kernel"].shape
  kdtype = crit_state["classifier"]["kernel"].dtype
  crit_state["classifier"]["kernel"] = (
      jax.random.normal(key, kshape, dtype=kdtype) * 0.02
  )

  # Shard once at the end
  crit_sharding = nnx.get_named_sharding(crit_state, _mesh)
  crit_state = jax.tree.map(
      lambda x, s: jax.device_put(x, s), crit_state, crit_sharding
  )
  return nnx.merge(crit_graph_def, crit_state)


def reset_dir(path: str):
  if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True)
  os.makedirs(path, exist_ok=True)


def build_optimizers(derived):
  if str(derived.get("optim_type", "constant")).lower() == "constant":
    actor_lr_schedule = optax.constant_schedule(derived["actor_lr"])
    critic_lr_schedule = optax.constant_schedule(derived["critic_lr"])
  else:
    # Fallback: use fixed float if an unknown type is provided
    actor_lr_schedule = derived["actor_lr"]
    critic_lr_schedule = derived["critic_lr"]

  actor_opt = optax.adamw(
      learning_rate=actor_lr_schedule,
      b1=derived["b1"],
      b2=derived["b2"],
      weight_decay=derived["weight_decay"],
  )
  critic_opt = optax.adamw(
      learning_rate=critic_lr_schedule,
      b1=derived["b1"],
      b2=derived["b2"],
      weight_decay=derived["weight_decay"],
  )
  if derived["max_grad_norm"] is not None:
    actor_opt = optax.chain(
        optax.clip_by_global_norm(max_norm=derived["max_grad_norm"]), actor_opt
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(max_norm=derived["max_grad_norm"]), critic_opt
    )
  return actor_opt, critic_opt


def build_cluster_config(mesh, tokenizer, derived, cfg):
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=derived["save_interval_steps"],
      max_to_keep=derived["max_to_keep"],
  )
  tb_log_dir = str(BASE_DIR / "content" / "tmp" / "tensorboard" / "ppo")
  reset_dir(derived["ckpt_dir"])
  reset_dir(tb_log_dir)
  metrics_opts = metrics_logger.MetricsLoggerOptions(
      log_dir=tb_log_dir, flush_every_n_steps=20
  )

  actor_opt, critic_opt = build_optimizers(derived)

  # Build role_to_mesh from cfg.cluster.role_to_mesh
  def _mesh_from_spec(spec):
    if spec is None or spec == "same":
      return mesh
    try:
      shape = tuple(int(x) for x in spec["shape"])  # type: ignore[index]
      axes = tuple(str(x) for x in spec["axes"])  # type: ignore[index]
      return jax.make_mesh(shape, axes)
    except Exception:
      return mesh

  r2m_cfg = getattr(cfg.cluster, "role_to_mesh", "same")
  if isinstance(r2m_cfg, (str, type(None))) and (
      r2m_cfg is None or r2m_cfg == "same"
  ):
    role_to_mesh = {
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.CRITIC: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    }
  else:
    # Expect mapping per role
    role_to_mesh = {
        rl_cluster_lib.Role.ACTOR: _mesh_from_spec(r2m_cfg.get("actor")),
        rl_cluster_lib.Role.CRITIC: _mesh_from_spec(r2m_cfg.get("critic")),
        rl_cluster_lib.Role.REFERENCE: _mesh_from_spec(
            r2m_cfg.get("reference")
        ),
        rl_cluster_lib.Role.ROLLOUT: _mesh_from_spec(r2m_cfg.get("rollout")),
    }

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh=role_to_mesh,
      rollout_engine=derived["rollout_engine"],
      offload_to_cpu=derived["cpu_offload"],
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=actor_opt,
          critic_optimizer=critic_opt,
          mini_batch_size=(
              int(cfg.cluster.training_config.mini_batch_size)
              if getattr(cfg.cluster.training_config, "mini_batch_size", None)
              is not None
              else None
          ),
          training_micro_batch_size=(
              int(cfg.cluster.training_config.training_micro_batch_size)
              if getattr(
                  cfg.cluster.training_config, "training_micro_batch_size", None
              )
              is not None
              else None
          ),
          eval_every_n_steps=derived["eval_every_n_steps"],
          max_steps=derived["max_steps"],
          gradient_accumulation_steps=derived["grad_accum"],
          metrics_logging_options=metrics_opts,
          checkpoint_root_directory=derived["ckpt_dir"],
          checkpointing_options=checkpointing_options,
      ),
      rollout_config={
          rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
              max_tokens_to_generate=derived["total_generation_steps"],
              max_prompt_length=derived["max_prompt_length"],
              kv_cache_size=derived["max_prompt_length"]
              + derived["total_generation_steps"]
              + 256,
              temperature=derived["temperature_train"],
              top_p=derived["top_p"],
              top_k=derived["top_k"],
          ),
          rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
              max_tokens_to_generate=derived["total_generation_steps"],
              max_prompt_length=derived["max_prompt_length"],
              kv_cache_size=derived["max_prompt_length"]
              + derived["total_generation_steps"]
              + 256,
              temperature=derived["temperature_eval"],
              top_p=1.0,
              top_k=None,
          ),
      },
  )
  return cluster_config


def build_trainer(rl_cluster, cfg, derived, _multi_turn_cfg):
  ppo_cfg = cfg.trainer.ppo
  ppo_config = PpoConfigExp(
      num_ppo_epochs=int(ppo_cfg.num_ppo_epochs),
      gamma=float(ppo_cfg.gamma),
      gae_lambda=float(ppo_cfg.gae_lambda),
      beta=float(ppo_cfg.beta),
      epsilon=float(ppo_cfg.epsilon),
      clip_range_value=float(ppo_cfg.clip_range_value),
      entropy_coef=float(ppo_cfg.entropy_coef),
      epsilon_low=float(ppo_cfg.epsilon_low),
      epsilon_high=float(ppo_cfg.epsilon_high),
      epsilon_c=float(ppo_cfg.epsilon_c),
      kl_method=str(ppo_cfg.kl_method),
  )
  trainer = PpoLearnerExp(
      rl_cluster=rl_cluster,
      ppo_config=ppo_config,
      reward_fns=_dummy_reward_fn,
      multi_turn_cfg=_multi_turn_cfg,
      multi_turn_processor=None,
      multi_turn_validation=False,
  )
  return trainer


@hydra_main(
    config_path="../configs", config_name="tunix_base", version_base=None
)
def main(cfg: DictConfig):
  # Use Hydra cfg directly
  derived = derive_hparams(cfg)
  _print_config_summary(cfg, derived)

  # Build dataset (use derived num_batches and mini_batch_size for placeholder sizing)
  dataset = get_dataset(derived["num_batches"], derived["mini_batch_size"], "train")

  # Init models
  policy_qwen2, critic_qwen2, qwen2_ref, tokenizer, mesh, model_config = (
      build_models_and_tokenizer(cfg, derived)
  )

  # RL cluster and trainer
  cluster_config = build_cluster_config(mesh, tokenizer, derived, cfg)
  with mesh:
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_qwen2,
        critic=critic_qwen2,
        reference=qwen2_ref,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    trainer = build_trainer(rl_cluster, cfg, derived, cfg)
    trainer.train(dataset)

  try:
    wandb.finish()
  except Exception:
    pass


if __name__ == "__main__":
  main()
