"""
General training entrypoint for Tunix PPO/GRPO, modeled after train.py and
the GRPO demo notebook. Uses configs from configs/tunix_base.yaml and
configs/tunix_ppo_trainer.yaml to assemble the training pipeline.

This is an outline that wires together:
- Config loading
- Mesh and optimizer construction
- Cluster/rollout configs (tunix.rl.rl_cluster + tunix.rl.rollout)
- RL cluster and learner (GRPO or PPO)
- Optional dataset construction
- Trainer.train()
"""

from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import OmegaConf


def _build_mesh(cfg) -> Any:
    """Builds the JAX mesh from config.mesh.dims."""
    import jax

    dims = cfg.mesh.dims
    # Expecting dims like: [[1, 4], ["fsdp", "tp"]]
    if not (isinstance(dims, list) and len(dims) == 2):
        raise ValueError("mesh.dims must be [[mesh_shape...], [axis_names...]]")
    mesh_shape, axis_names = dims[0], dims[1]
    return jax.make_mesh(mesh_shape, axis_names)


def _build_optimizer(cfg) -> Any:
    """Creates an AdamW optimizer per tunix demo parameters."""
    import optax

    tr = cfg.training
    lr = tr.optimizer.learning_rate
    b1 = tr.optimizer.b1
    b2 = tr.optimizer.b2
    wd = tr.optimizer.weight_decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=tr.optimizer.warmup_steps,
        decay_steps=tr.max_steps,
        end_value=0.0,
    )
    adamw = optax.adamw(learning_rate=schedule, b1=b1, b2=b2, weight_decay=wd)
    if tr.optimizer.max_grad_norm is not None:
        return optax.chain(optax.clip_by_global_norm(tr.optimizer.max_grad_norm), adamw)
    return adamw


def _build_rollout_config(cfg) -> Any:
    """Maps cfg.rollout to base_rollout.RolloutConfig."""
    from tunix.rl.rollout import base_rollout

    return base_rollout.RolloutConfig(
        max_tokens_to_generate=cfg.rollout.max_tokens_to_generate,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
        top_k=cfg.rollout.top_k,
        max_prompt_length=cfg.rollout.max_prompt_length,
        kv_cache_size=cfg.rollout.kv_cache_size,
    )


def _build_cluster_config(cfg, mesh, optimizer) -> Any:
    """Constructs rl_cluster.ClusterConfig and RLTrainingConfig from cfg."""
    from tunix.rl import rl_cluster as rl_cluster_lib

    # Map mesh alias to actual JAX Mesh
    role_to_mesh = {
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    }

    rl_training_cfg = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=cfg.training.eval_every_n_steps,
        max_steps=cfg.training.max_steps,
        gradient_accumulation_steps=1,
        metrics_logging_options=cfg.training.metrics_logging_options,
        checkpoint_root_directory=cfg.training.checkpoint.ckpt_dir,
        checkpointing_options=cfg.training.checkpoint,
    )

    rollout_cfg = _build_rollout_config(cfg)

    return rl_cluster_lib.ClusterConfig(
        role_to_mesh=role_to_mesh,
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_training_cfg,
        rollout_config=rollout_cfg,
    )


def _build_models_and_tokenizer(cfg, mesh):
    """Loads/creates actor + reference models and tokenizer.

    Replace this with real model loading (e.g., Gemma2 + LoRA) following the demo.
    """
    # Example (see tunix/examples/grpo_demo.ipynb): load base model, apply LoRA,
    # shard on `mesh`, and create tokenizer. Here we raise to highlight where to plug in.
    raise NotImplementedError("Implement model/tokenizer loading per your use case")


def _maybe_build_dataset(cfg):
    """Optionally builds a dataset from cfg.data.* (like the demo's grain pipeline)."""
    return None  # Outline only; plug in your dataset when ready


def _build_learner(cfg, rl_cluster):
    """Builds GRPO or PPO learner based on available config sections."""
    # Prefer GRPO if `grpo` section is present, else PPO
    if hasattr(cfg, "grpo"):
        from tunix.rl.grpo.grpo_learner import GrpoLearner, GrpoConfig

        grpo_cfg = GrpoConfig(
            num_generations=cfg.grpo.num_generations,
            num_iterations=cfg.grpo.num_iterations,
            beta=cfg.grpo.beta,
            epsilon=cfg.grpo.epsilon,
        )
        return GrpoLearner(rl_cluster=rl_cluster, grpo_config=grpo_cfg, reward_fns=None)
    else:
        from tunix.rl.ppo.ppo_learner import PpoLearner, PpoConfig

        ppo_cfg = PpoConfig()
        return PpoLearner(rl_cluster=rl_cluster, ppo_config=ppo_cfg, reward_fns=None)


def _train(cfg):
    """High-level training orchestration following the GRPO demo flow."""
    # 1) Mesh
    mesh = _build_mesh(cfg)

    # 2) Optimizer
    optimizer = _build_optimizer(cfg)

    # 3) Models + tokenizer (implement per your model)
    # actor_model, reference_model, tokenizer = _build_models_and_tokenizer(cfg, mesh)
    # For outline purposes, raise a helpful error
    if os.environ.get("TUNIX_LOAD_MODELS", "0") != "1":
        raise RuntimeError("Model/tokenizer loading is not implemented in outline. Set up _build_models_and_tokenizer().")

    # 4) Cluster config
    # cluster_cfg = _build_cluster_config(cfg, mesh, optimizer)

    # 5) RLCluster
    # from tunix.rl import rl_cluster as rl_cluster_lib
    # rlc = rl_cluster_lib.RLCluster(actor=actor_model, reference=reference_model, tokenizer=tokenizer, cluster_config=cluster_cfg)

    # 6) Learner
    # learner = _build_learner(cfg, rlc)

    # 7) Dataset (optional)
    # train_ds = _maybe_build_dataset(cfg)

    # 8) Train
    # with mesh:
    #     if hasattr(cfg, "grpo"):
    #         learner.train(train_ds)
    #     else:
    #         learner.train(train_ds)


@hydra.main(config_path="../configs", config_name="tunix_ppo_trainer", version_base=None)
def main(cfg):
    # Merge base and trainer configs if needed
    # You can also set defaults in configs/tunix_base.yaml to include tunix_ppo_trainer
    if os.path.exists(os.path.join(hydra.utils.get_original_cwd(), "configs", "tunix_base.yaml")):
        base = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "configs", "tunix_base.yaml"))
        cfg = OmegaConf.merge(base, cfg)

    # Print resolved config (optional)
    # from pprint import pprint
    # pprint(OmegaConf.to_container(cfg, resolve=True))

    _train(cfg)


if __name__ == "__main__":
    main()



def main():

if __name__ == "__main__":
    main()
