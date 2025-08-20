# Tunix Integration Plan — Multi‑Turn PPO Training

This document outlines how to integrate Tunix (JAX/NNX PPO/GRPO stack) with the existing multi‑turn rollout PPO training approach described in `docs/SYSTEMDESIGN.md`.

Goals:
- Replace single‑turn sampling (`rl_cluster.generate`) with a synchronous multi‑turn rollout orchestrator.
- Remove traditional dataloaders; generate training batches directly from multi‑turn rollouts.
- Keep PPO/GRPO updates and logging intact, mirroring the reference design.

References:
- Overall design: `docs/SYSTEMDESIGN.md`
- Multi‑turn rollout: `grl/rollout/sync_multi_turn_rollout.py` (reference) and `grl/rollout/tunix_sync_multi_turn_rollout.py` (Tunix port)
- VERL trainer example: `grl/trainer/agent_trainer.py`
- Tunix PPO learner and cluster: `tunix/tunix/rl/ppo/ppo_learner.py`, `tunix/tunix/rl/rl_cluster.py`, `tunix/tunix/rl/rollout/vanilla_rollout.py`

---

## Architecture Overview (Tunix integration)

Key idea: Swap `rl_cluster.generate(prompts=...)` with a multi‑turn rollout that batches prompts each turn, calls the Tunix rollout engine under the hood, and returns the tensors needed for PPO/GRPO.

---

## Components to Modify

### 1) `grl/trainer/tunix_agent_trainer.py`
Target: Replace one‑shot `rl_cluster.generate` with the multi‑turn rollout path and remove dataloaders.

- Add rollout manager wiring
  - Initialize a multi‑turn rollout orchestrator at trainer start (similar to `AgentTrainer.init_multi_turn_rollout`):
    - `self.multi_turn_rollout = TunixSyncMultiTurnRollout(rl_cluster=..., cfg=config, tokenizer=..., ...)`
    - Use the Tunix rollout model behind `rl_cluster.rollout` (VanillaRollout → `Sampler`).

- Replace sampling in PPO step
  - Wherever the current learner calls `self.rl_cluster.generate(prompts=...)`, replace with:
    - `self.multi_turn_rollout.rollout()` to run turns synchronously.
    - Collect final rollout states and build a PPO batch shape (prompt/completion ids, masks, scores).
  - Two ways to connect back to PPO:
    - a) Build `TrainExample` batches directly (ids + masks + advantages) and feed into Tunix PPO update; or
    - b) Build an intermediate batch, then compute advantages and map to `TrainExample`.

- Remove/avoid dataloaders
  - Do not create/consume traditional datasets. Batches are generated from rollouts each step.
  - Keep the optimization/update loop unchanged; only the data source is different.

Minimal function‑level changes:
- Initialize once:
  - `init_multi_turn_rollout(self)`
- Replace sampling in the data path:
  - `_generate_and_compute_advantage(...)`: swap `self.rl_cluster.generate(...)` with calls to `self.multi_turn_rollout` and map outputs to `TrainExample`.
- Optionally streamline `_prepare_data(...)` to enqueue rollout‑derived batches instead of iterating over a dataset.


### 2) `grl/rollout/tunix_sync_multi_turn_rollout.py`
Target: Port the reference `SyncMultiTurnRollout` to call Tunix engines.

- Generation backend
  - In `generate_sequences(...)`, call Tunix rollout via `rl_cluster.generate(prompts=[...])` or directly via `rl_cluster.rollout.generate(...)` with `RolloutConfig`.
  - Keep multi‑turn loop the same: batch prompts per turn → generate → env step → repeat until done.

- Batch building
  - Provide a `build_ppo_batch(...)` that returns either:
    - `TrainExample` (JAX arrays) for direct PPO update, or
    - An intermediate batch (e.g., prompt/completion ids/masks/scores) that the trainer converts to `TrainExample`.

- Validation
  - Reuse the same rollout path with deterministic settings (e.g., `do_sample=False`, `temperature=0`).


### 3) `grl/tunix_train.py`
Target: Training entrypoint modeled after the GRPO demo and `train.py`, using Hydra configs.

- Configs
  - Read `configs/tunix_ppo_trainer.yaml` (+ merge `configs/tunix_base.yaml` if present).
  - Build Mesh and Optimizer from config.

- Cluster
  - Build `RolloutConfig` from `cfg.rollout` and `ClusterConfig` from `cfg.cluster` + training options.
  - Instantiate `RLCluster(actor, reference, tokenizer, cluster_config)`.

- Trainer
  - Create `TunixAgentTrainer` (the class in `tunix_agent_trainer.py`) and call:
    - `trainer.init_workers()` (if needed for your setup)
    - `trainer.fit()` to start the main loop

- No datasets
  - Do not construct dataloaders; the trainer pulls data from multi‑turn rollouts.

---

## Training Flow (Step by Step)

1. Load configs via Hydra (`tunix_base.yaml`, `tunix_ppo_trainer.yaml`).
2. Build JAX mesh and Optax optimizer (AdamW + warmup/cosine) using demo values.
3. Load models (actor with LoRA, reference), shard to mesh, construct tokenizer.
4. Build `ClusterConfig` and `RolloutConfig`; create `RLCluster`.
5. Create `TunixAgentTrainer` and initialize its multi‑turn rollout manager.
6. For each training step:
   - Run `multi_turn_rollout.rollout()` to collect full trajectories.
   - Convert to PPO batch (`TrainExample`), compute advantages.
   - Update actor and critic; log metrics; optionally validate and checkpoint.
7. Repeat until `max_steps`.

---

## What Will Change vs. Baseline Tunix PPO

- Sampling: `self.rl_cluster.generate(...)` is replaced by `self.multi_turn_rollout.rollout()` + batch building.
- Data: No dataloaders/datasets; rollout is the sole data source.
- Trainer loop: stays PPO/GRPO‑compatible; only the sample source changes.

---

## Files and Functions to Touch (Summary)

- `grl/trainer/tunix_agent_trainer.py`
  - Add: `init_multi_turn_rollout(self)`
  - Modify: `_generate_and_compute_advantage(...)` to use rollout instead of `rl_cluster.generate`
  - Optional: `_prepare_data(...)` to remove dataset iteration and enqueue rollout batches

- `grl/rollout/tunix_sync_multi_turn_rollout.py`
  - Modify: `generate_sequences(...)` to call Tunix rollout engine
  - Add/Adapt: `build_ppo_batch(...)` to output `TrainExample` or an intermediate batch
  - Keep: `rollout()` multi‑turn loop (batch prompts → generate → env step)

- `grl/tunix_train.py`
  - Implement: Hydra entrypoint that assembles Mesh, Optimizer, RLCluster, Trainer and starts training (no dataloaders)

---

## Configuration Mapping

- `configs/tunix_ppo_trainer.yaml` (new) holds:
  - `rollout`: `max_prompt_length`, `max_tokens_to_generate`, `temperature`, `top_p`, `top_k`, `kv_cache_size`
  - `training`: `max_steps`, optimizer (AdamW), logging, checkpointing
  - `mesh`: dims for `jax.make_mesh`
  - `grpo` (optional): `num_generations`, `num_iterations`, `beta`, `epsilon`
- `configs/tunix_base.yaml` can provide general trainer defaults if desired.

---

## Next Steps

- Implement model/tokenizer load in `tunix_train.py` (Gemma + LoRA, sharded to mesh) following the demo.
- Adapt `tunix_sync_multi_turn_rollout.py` batch builder to emit `TrainExample` compatible with Tunix PPO/GRPO.
- Replace the generation call inside `tunix_agent_trainer.py` and test end‑to‑end with small configs.

========================================================================================================================
Tunix Issue Records: 
1. out of memory 
2. critic model (one more linear layer)
3. prompt id and completion ids: prompt_mask and completion_mask
