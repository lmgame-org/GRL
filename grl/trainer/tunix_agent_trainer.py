"""Thin wrapper around Tunix PPO to allow later customization.

This module exposes `TrainExample` and `PpoConfig` directly from Tunix,
and provides a subclass of `PpoLearner` for incremental overrides.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.ppo import ppo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import metrics_logger
from grl.rollout.tunix_sync_multi_turn_rollout import SyncMultiTurnRollout
import numpy as np
import jax.nn as jnn

# Re-exported wrappers inherit from Tunix PPO implementations
from tunix.rl.ppo.ppo_learner import (
    TrainExample as BaseTrainExample,  # re-export base
    PpoConfig as BasePpoConfig,       # re-export base
    PpoLearner,
)

# Extend base types with entropy configuration used only by the custom loss
@flax.struct.dataclass(frozen=True)
class TrainExample(BaseTrainExample):
  entropy_coeff: jax.Array | float = flax.struct.field(pytree_node=False, default=0.0)
  aggs_mode: str = flax.struct.field(pytree_node=False, default="token-mean")


@dataclasses.dataclass(slots=True, kw_only=True)
class PpoConfig(BasePpoConfig):
  entropy_coeff: float = 0.0
  aggs_mode: str = "token-mean"

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]

__all__ = [
    "TrainExample",
    "PpoConfig",
    "PpoLearner",
]

@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    completion_mask: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
  """Masked GAE, aligned with TRL-style implementation.

  This version respects variable-length sequences using a mask. After the
  end-of-sequence (where the mask is 0), the recursion for both the bootstrap
  value and the advantage accumulator is held constant (no updates).

  Args:
    rewards: Array of shape [B, T] containing per-token rewards.
    values: Array of shape [B, T] containing value estimates V(s_t).
    completion_mask: Boolean or 0/1 mask of shape [B, T]; positions after EOS
      must be 0.
    gamma: Discount factor.
    gae_lambda: GAE lambda parameter.

  Returns:
    A tuple (advantages, returns), each of shape [B, T].
  """
  B, T = rewards.shape
  mask = completion_mask.astype(values.dtype)

  def scan_step(carry, xs):
    next_values_carry, last_gae_carry = carry  # each [B]
    reward_t, value_t, mask_t = xs  # each [B]

    delta_t = reward_t + gamma * next_values_carry - value_t
    last_gae_unmasked = delta_t + gamma * gae_lambda * last_gae_carry

    # Apply mask gating (hold carries when mask_t == 0)
    next_values_new = value_t * mask_t + (1.0 - mask_t) * next_values_carry
    last_gae_new = last_gae_unmasked * mask_t + (1.0 - mask_t) * last_gae_carry

    return (next_values_new, last_gae_new), last_gae_new

  init_carry = (jnp.zeros((B,), dtype=values.dtype), jnp.zeros((B,), dtype=values.dtype))
  xs = (jnp.transpose(rewards), jnp.transpose(values), jnp.transpose(mask))

  (_, _), advantages_T = jax.lax.scan(
      scan_step,
      init=init_carry,
      xs=xs,
      reverse=True,
  )

  advantages = jnp.transpose(advantages_T)
  returns = advantages + values
  return advantages, returns

# (Entropy helpers removed)
@jax.jit
def entropy_from_logits(logits: jax.Array) -> jax.Array:
  """Per-token categorical entropy from logits; shape [..., V] -> [...]."""
  probs = jnn.softmax(logits, axis=-1)
  logsumexp = jax.nn.logsumexp(logits, axis=-1)
  expected_logits = jnp.sum(probs * logits, axis=-1)
  return logsumexp - expected_logits


def agg_loss(
    loss_mat: jax.Array,
    loss_mask: jax.Array,
    loss_agg_mode: str = "token-mean",
) -> jax.Array:
  """Aggregate the loss matrix into a scalar.

  Args:
    loss_mat: [B, T]
    loss_mask: [B, T]
    loss_agg_mode: aggregation mode
  Returns:
    scalar loss
  """
  loss_mask = loss_mask.astype(loss_mat.dtype)
  if loss_agg_mode == "token-mean":
    return ppo_helpers.masked_mean(loss_mat, loss_mask)
  elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = jnp.sum(loss_mat * loss_mask, axis=-1)
    return jnp.mean(seq_losses)
  elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = jnp.sum(loss_mat * loss_mask, axis=-1)
    seq_denoms = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)
    return jnp.mean(seq_losses / seq_denoms)
  elif loss_agg_mode == "seq-mean-token-sum-norm":
    seq_losses = jnp.sum(loss_mat * loss_mask, axis=-1)
    return jnp.sum(seq_losses) / loss_mask.shape[-1]
  else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon: float,
    pad_id: int,
    eos_id: int,
):
  """Computes PPO policy loss with optional entropy regularization."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # Compute per-token log-probs via model forward
  prompt_completion_ids, positions, attn_mask = common.process_ids(
      prompt_ids, completion_ids, pad_id, eos_id
  )
  logits, _ = model(
      prompt_completion_ids,
      positions=positions,
      attention_mask=attn_mask,
      cache=None,
  )

  logits_to_keep = completion_ids.shape[1]
  logits_slice = logits[:, -logits_to_keep - 1 : -1, :]
  per_token_logps = common.selective_log_softmax(logits_slice, completion_ids)
  per_token_logps = jnp.where(
      completion_mask,
      per_token_logps,
      jnp.array(1).astype(per_token_logps.dtype),
  )

  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps
  coef_1 = jnp.exp(per_token_logps - old_per_token_logps)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)
  policy_loss = jnp.maximum(
      -coef_1 * jnp.expand_dims(advantages, 1),
      -coef_2 * jnp.expand_dims(advantages, 1),
  )
  policy_loss = ppo_helpers.masked_mean(policy_loss, completion_mask)

  aux = {}
  entropy_coeff = float(train_example.entropy_coeff) if train_example.entropy_coeff is not None else 0.0
  if entropy_coeff > 0.0:
    # Per request: always use token-mean aggregation for entropy
    token_entropy = entropy_from_logits(logits_slice)
    entropy_loss = agg_loss(token_entropy, completion_mask, loss_agg_mode="token-mean")
    policy_loss = policy_loss - (entropy_coeff * entropy_loss)
    aux["entropy/loss_mean"] = entropy_loss
    aux["entropy/coeff"] = entropy_coeff

  return policy_loss, aux


class MultiTurnPpoLearner(PpoLearner):
  """Wrapper subclass of Tunix PPO PpoLearner.

  Override lifecycle hooks or training logic here progressively.
  """
  def __init__(
    self,
    rl_cluster: rl_cluster_lib.RLCluster,
    ppo_config: PpoConfig,
    reward_fns: RewardFn | List[RewardFn] | None = None,
    data_shuffle_seed: int | None = None,
    *,
    multi_turn_cfg=None,
    multi_turn_processor=None,
    multi_turn_validation: bool = False,
    ):
    super().__init__(
        rl_cluster=rl_cluster,
        ppo_config=ppo_config,
        reward_fns=reward_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
    # Override actor loss with custom PPO policy loss (with optional entropy)
    self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)

    
    # ─────────────────── Multi-turn rollout manager ───────────────────
    # Initialize when config is provided; otherwise remain None (fallback to single-turn)
    self.multi_turn_rollout = None
    self.validation_multi_turn_rollout = None
    # Keep references for potential re-init
    self._multi_turn_cfg = multi_turn_cfg
    self._multi_turn_processor = multi_turn_processor
    if multi_turn_cfg is not None:
      # Use the RLCluster tokenizer; pass through cfg/processor from caller
      self.multi_turn_rollout = SyncMultiTurnRollout(
          rl_cluster=self.rl_cluster,
          cfg=multi_turn_cfg,
          tokenizer=self.rl_cluster.tokenizer,
          processor=multi_turn_processor,
          validation=multi_turn_validation,
      )
      # Create a dedicated validation rollout with validation=True
      self.validation_multi_turn_rollout = SyncMultiTurnRollout(
          rl_cluster=self.rl_cluster,
          cfg=multi_turn_cfg,
          tokenizer=self.rl_cluster.tokenizer,
          processor=multi_turn_processor,
          validation=True,
      )
    # Storage for last built rollout batch to aid later conversion to TrainExample
    self._last_rollout_batch = None
    
  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> TrainExample:
    """Generates completions and computes advantages for PPO training.

    Args:
      training_input: A dictionary containing the training input data, including
        the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data for PPO.
    """
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    def _get_rollout_config_for_mode(_mode: metrics_logger.Mode):
      rc = self.rl_cluster.cluster_config.rollout_config
      if isinstance(rc, dict):
        key = (
          rl_cluster_lib.Mode.TRAIN
          if _mode == metrics_logger.Mode.TRAIN
          else rl_cluster_lib.Mode.EVAL
        )
        return rc[key]
      return rc

    max_prompt_length = _get_rollout_config_for_mode(mode).max_prompt_length
    
    # ─────────────────── MODIFICATION: Multi-turn rollout with deferred metrics logging ───────────────────
    # Multi-turn rollout (side-by-side). We defer logging of rollout metrics to the unified
    # metric logging section below to match PPO learner logging semantics.
    mt_metrics_dict = None
    meta_metrics_dict = None
    if self.multi_turn_rollout is not None:
      try:
        mt_batch = self.multi_turn_rollout.rollout()
        mt_batch_filtered, mt_metrics = self.multi_turn_rollout.filter_rollout_batch(mt_batch)
        self._last_rollout_batch = mt_batch_filtered
        # Defer logging of multi-turn metrics to the unified metric logging section
        mt_metrics_dict = dict(mt_metrics)
        meta_metrics_dict = dict(mt_batch_filtered.meta_info.get("metrics", {}))
      finally:
        # Always reset to release memory between iterations, even if downstream steps fail
        self.multi_turn_rollout.reset()
    # ─────────────────── END MODIFICATION ───────────────────

    # ─────────────────── MODIFICATION: Full-conversation conversion (replace single-turn rl_cluster.generate) ───────────────────
    # Convert the filtered multi-turn RolloutBatch into Tunix PPO tensors using full-conversation framing:
    #   prompt_ids = input_ids[:, :-1] (left-padded to max_prompt_length)
    #   completion_ids = input_ids[:, 1:]
    #   completion_mask = loss_mask (already aligned to completion_ids)
    #   completion_plus_one_mask constructed like Tunix for value alignment
    def _convert_rollout_batch_to_tunix_format(batch, Pmax: int):
      inp = np.array(batch.input_ids)
      loss_m = np.array(batch.loss_mask).astype(bool)  # [B, L-1]
      B, L = inp.shape

      # Full-conversation split
      fc_prompt = inp[:, :-1]     # [B, L-1]
      fc_completion = inp[:, 1:]  # [B, L-1]

      # Left-pad prompt to max_prompt_length with pad_id
      pad_id = int(self.rl_cluster.rollout.pad_id())
      if fc_prompt.shape[1] > Pmax:
        prompt_padded = fc_prompt[:, -Pmax:]
      else:
        left_pad = Pmax - fc_prompt.shape[1]
        prompt_padded = np.concatenate([
          np.full((B, left_pad), pad_id, dtype=fc_prompt.dtype),
          fc_prompt
        ], axis=1)

      prompt_ids_local = jnp.array(prompt_padded)
      completion_ids_local = jnp.array(fc_completion)
      completion_mask_local = jnp.array(loss_m)
      prompt_mask_local = (prompt_ids_local != pad_value).astype("int32")

      # EOS index and completion_plus_one_mask like Tunix
      eos_idx_local = jnp.max(common.build_positions_from_mask(completion_mask_local), axis=-1)
      is_padding_token_local = jnp.any(~completion_mask_local, axis=-1)
      completion_plus_one_mask_local = completion_mask_local.at[
          jnp.arange(B)[is_padding_token_local],
          (eos_idx_local + 1)[is_padding_token_local],
      ].set(True)

      return (
        prompt_ids_local,
        completion_ids_local,
        prompt_mask_local,
        completion_mask_local,
        eos_idx_local,
        completion_plus_one_mask_local,
      )

    if self.multi_turn_rollout is None or self._last_rollout_batch is None:
      raise RuntimeError("Multi-turn rollout is not initialized or has no batch; cannot perform full-conversation PPO.")

    (
      prompt_ids,
      completion_ids,
      prompt_mask,
      completion_mask,
      eos_idx,
      completion_plus_one_mask,
    ) = _convert_rollout_batch_to_tunix_format(
      self._last_rollout_batch,
      int(max_prompt_length),
    )

    batch_size = completion_ids.shape[0]
    # ─────────────────── END MODIFICATION ───────────────────

    # ===== Compute log probs ======
    logits_to_keep = completion_ids.shape[1]
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
      # Set log probs to 1 for padding tokens.
      ref_per_token_logps = jnp.where(
          completion_mask,
          ref_per_token_logps,
          jnp.array(0).astype(ref_per_token_logps.dtype),
      )
    else:
      ref_per_token_logps = None

    # Get log probs from the policy before model weights are updated. We use
    # the policy model here. Shape = `[B, T]`.
    # TODO(abheesht): Do we do this only when `self.num_ppo_epochs > 1`? Don't
    # see this condition here:
    # https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L1224-L1233.
    old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
    )
    # Set log probs to 1 for padding tokens.
    old_per_token_logps = jnp.where(
        completion_mask,
        old_per_token_logps,
        jnp.array(0).astype(old_per_token_logps.dtype),
    )

    # (Entropy loss computation removed)

    # ===== Value computation ======
    # Get values from the value model before model weights are updated.
    values = self.rl_cluster.get_values(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
    )

    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    # Set `values` corresponding to padding tokens to 0.
    values = jnp.where(
        completion_plus_one_mask,
        values,
        jnp.array(0).astype(values.dtype),
    )
    # Lightweight consistency check (no crash): ensure value targets align with completion length
    try:
      if int(values.shape[1]) != int(logits_to_keep):
        print(f"[warn] values length ({int(values.shape[1])}) != completion length ({int(logits_to_keep)})")
    except Exception:
      pass

    # ===== Reward computation ======
    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
      # We use the score corresponding to the last non-padding token.
      # Remember, completion is padded on the right, and the prompt is padded on
      # the left.
      last_token_scores = scores[
          jnp.arange(batch_size),
          eos_idx + max_prompt_length,
      ]
    else:
      # ─────────────────── MODIFICATION: Use rollout batch rewards (final column) instead of text-based reward_fns ───────────────────
      # Convert to JAX array and place on the same sharding as model inputs for correct device allocation.
      reward_scores_jax = jnp.asarray(self._last_rollout_batch.reward_scores)
      try:
        # Match sharding with completion_ids when available (pjit/named sharding setups)
        if hasattr(completion_ids, "sharding") and completion_ids.sharding is not None:
          reward_scores_jax = jax.device_put(reward_scores_jax, completion_ids.sharding)
      except Exception:
        # Best-effort; fallback keeps it as a regular JAX array on default device
        pass
      # Align with full-conversation rollout: reward_scores is [B, L-1], take the last column per sample.
      last_token_scores = reward_scores_jax[:, -1]
      # ─────────────────── END MODIFICATION ───────────────────

    # This is how rewards are computed. This is in accordance with TRL and
    # with verl's `NaiveRewardManager`. This is a different from GRPO, where
    # we don't consider rewards at the token level.
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. Subtract KL divergence from the reward tensor of all 0s.
    # 3. A positive reward is given only at the final timestep, so we add that
    # to the reward tensor from (2).
    # ─────────────────── MODIFICATION: Ensure rewards are float and match sharding ───────────────────
    rewards = jnp.zeros(completion_ids.shape, dtype=values.dtype)
    try:
      if hasattr(values, "sharding") and values.sharding is not None:
        rewards = jax.device_put(rewards, values.sharding)
    except Exception:
      pass
    # ─────────────────── END MODIFICATION ───────────────────
    if self.ppo_config.beta != 0.0:
      # ─────────────────── MODIFICATION: Use direct subtraction of log probs instead of KL divergence ───────────────────
      # kl = common.compute_kl_divergence(
      #     old_per_token_logps, ref_per_token_logps
      # )
      # ─────────────────── END MODIFICATION ───────────────────
      kl = old_per_token_logps - ref_per_token_logps
      rewards = rewards - self.ppo_config.beta * kl

    # Ensure indices and updates are placed on the same sharding as `rewards` for scatter-add
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
    if hasattr(rewards, "sharding") and rewards.sharding is not None:
      try:
        batch_indices = jax.device_put(batch_indices, rewards.sharding)
        eos_idx = jax.device_put(eos_idx, rewards.sharding)
        # ─────────────────── MODIFICATION: Cast rewards update to match dtype/sharding ───────────────────
        last_token_scores = jax.device_put(last_token_scores.astype(rewards.dtype), rewards.sharding)
        # ─────────────────── END MODIFICATION ───────────────────
      except Exception:
        pass
    rewards = rewards.at[batch_indices, eos_idx].add(last_token_scores.astype(rewards.dtype))

    # ===== Metric logging ======
    # TODO(abheesht): Verify metric logging. We should move these to losses,
    # because the rollout batch can be split into mini-batches.
    step = self._get_metric_logging_steps(mode)



    # Log raw scores from the reward model/fn
    self._actor_metrics_logger.log(
        "score/mean", last_token_scores.mean(), mode, step
    )
    self._actor_metrics_logger.log(
        "score/max", last_token_scores.max(), mode, step
    )
    self._actor_metrics_logger.log(
        "score/min", last_token_scores.min(), mode, step
    )

    # Log final rewards (scores + KL penalty)
    sequence_rewards = rewards.sum(-1)
    self._actor_metrics_logger.log(
        "reward/mean", sequence_rewards.mean(), mode, step
    )
    self._actor_metrics_logger.log(
        "reward/max", sequence_rewards.max(), mode, step
    )
    self._actor_metrics_logger.log(
        "reward/min", sequence_rewards.min(), mode, step
    )
    if self.ppo_config.beta != 0.0:
      # VERL: per-token mean over sequence, then mean over batch (unscaled)
      per_sequence_mean_kl = ppo_helpers.masked_mean(  # [B]
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
      )
      current_kl = per_sequence_mean_kl.mean()
      self._actor_metrics_logger.log("actor/reward_kl_penalty", float(current_kl), mode, step)
      self._actor_metrics_logger.log("actor/reward_kl_penalty_coeff", float(self.ppo_config.beta), mode, step)

    # (Entropy metrics removed)

    # ─────────────────── MODIFICATION: Log multi-turn metrics in unified logging block ───────────────────
    # Log multi-turn rollout metrics (from filter/meta) in the same block
    if mt_metrics_dict is not None:
      for name, value in mt_metrics_dict.items():
        self._actor_metrics_logger.log(name, float(value), mode, step)
    if meta_metrics_dict is not None:
      for name, value in meta_metrics_dict.items():
        self._actor_metrics_logger.log(name, float(value), mode, step)
    # ─────────────────── END MODIFICATION ───────────────────

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self._actor_metrics_logger.log(
        "completions/mean_length",
        agg_completion_mask.mean(),
        mode,
        step,
    )
    self._actor_metrics_logger.log(
        "completions/max_length",
        agg_completion_mask.max(),
        mode,
        step,
    )
    self._actor_metrics_logger.log(
        "completions/min_length",
        agg_completion_mask.min(),
        mode,
        step,
    )

    # ===== Compute advantages using Generalised Advantage Estimation ======
    # advantages, returns = ppo_helpers.compute_gae_advantages(
    #     rewards=rewards,
    #     values=values,
    #     gamma=self.ppo_config.gamma,
    #     gae_lambda=self.ppo_config.gae_lambda,
    # )
    advantages, returns = compute_gae_advantages(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )
    # Normalize advantages.
    advantages = ppo_helpers.normalize_advantages(advantages, completion_mask)
    
    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_plus_one_mask=completion_plus_one_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_per_token_logps=old_per_token_logps,
        old_values=values,
        entropy_coeff=float(self.ppo_config.entropy_coeff),
        aggs_mode=str(self.ppo_config.aggs_mode),
    )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **kargs,
  ):
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      **kargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts,]`) of scalar rewards for
      each prompt-completion pair. The rewards are computed using the provided
      reward functions (their mean).
    """
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)

    # Take the mean of rewards, because we need one score per element.
    rewards = jnp.nanmean(rewards, axis=1)
    return rewards

  def _prepare_data(
      self,
      iterator: Iterator[_TrainingInputT],
      mini_batch_size: int | None,
      proceed_num_steps: int,
      batch_repeat: int,
      data_queue: queue_lib.AbstractDataQueue[
          list[TrainExample] | common.RepeatIterable | None
      ],
      async_loading: bool = False,
      shuffle_data: bool = False,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> None:
    """Prepares the dataset for training.

    Args:
      iterator: The input iterator of the dataset.
      mini_batch_size: The mini batch size which will be used to slice the
        dataset.
      proceed_num_steps: The number of steps to proceed for the iterator if set
        to a positive number. If it's set to a non positive number, the function
        will exhaust the iterator. If the input iterator is exhausted before the
        number of steps is reached, the function will return the empty result.
      batch_repeat: The number of times to repeat the batch in the final
        dataset.
      data_queue: The data queue to use for putting the examples into.
      async_loading: Whether to load the batch asynchronously, if not async
        loading, then all the examples needed will be processed and then loaded
        into the data queue.
      shuffle_data: Whether to shuffle the data.
      mode: The mode to use for logging metrics.

    Returns:
      None. Examples are put into the data queue.
    """
    # ─────────────────── MODIFICATION: Rollout-generated batches (no dataloaders) ───────────────────
    # Future change: instead of advancing an input iterator, call
    # self.multi_turn_rollout.rollout(), then convert outputs to TrainExample
    # and queue them directly.
    # ─────────────────── END MODIFICATION ───────────────────
    example_list = []

    def _put_list_of_examples_to_data_queue():
      if shuffle_data:
        self._data_shuffle_key, _ = jax.random.split(self._data_shuffle_key)

      if not async_loading:
        data_queue.put(
            common.RepeatIterable(
                example_list,
                repeat=batch_repeat,
                mini_batch_size=mini_batch_size,
                shuffle=shuffle_data,
                key=self._data_shuffle_key if shuffle_data else None,
            )
        )
      elif batch_repeat > 1:
        # Since we have already loaded the batch in data_queue once, we only
        # need to repeat batch_repeat - 1 times.
        data_queue.put(
            common.RepeatIterable(
                example_list,
                repeat=batch_repeat - 1,
                mini_batch_size=mini_batch_size,
                shuffle=shuffle_data,
                key=self._data_shuffle_key if shuffle_data else None,
            )
        )

    while True:
      try:
        while (
            mode == metrics_logger.Mode.TRAIN
            and self._train_steps < self._last_train_step
        ):
          next(iterator)
          self._train_steps += 1
        example = next(iterator)

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._train_steps
            if mode == metrics_logger.Mode.TRAIN
            else self._eval_steps,
        ):
          advantage = self._generate_and_compute_advantage(example, mode)
        if async_loading:
          data_queue.put([advantage])

        if mode == metrics_logger.Mode.TRAIN:
          self._train_steps += 1
        else:
          self._eval_steps += 1
        example_list.append(advantage)
        if proceed_num_steps > 0 and len(example_list) == proceed_num_steps:
          _put_list_of_examples_to_data_queue()
          data_queue.put(None)
          return
      except StopIteration as e:
        if proceed_num_steps > 0:
          data_queue.put(None)
          raise e
        else:
          _put_list_of_examples_to_data_queue()
          data_queue.put(None)
          return
      except Exception as e:
        data_queue.put(None)
        raise e

  # ─────────────────── MODIFICATION: Simple validation rollout logging (EVAL only) ───────────────────
  def _validate(self, eval_ds: Iterable[_TrainingInputT]) -> None:
    """Run a single validation rollout and log ONLY rollout metrics under EVAL.

    - Ignores `eval_ds`; data comes from the environment.
    - No advantage computation or model updates.
    - Logs metrics with EVAL mode so dashboards separate train/eval.
    """
    # Ensure validation rollout exists
    if self.validation_multi_turn_rollout is None:
      if self._multi_turn_cfg is None:
        raise RuntimeError("Multi-turn validation requested but no multi_turn_cfg provided.")
      self.validation_multi_turn_rollout = SyncMultiTurnRollout(
          rl_cluster=self.rl_cluster,
          cfg=self._multi_turn_cfg,
          tokenizer=self.rl_cluster.tokenizer,
          processor=self._multi_turn_processor,
          validation=True,
      )

    # Trigger print for visibility
    try:
      eval_every = self.rl_cluster.cluster_config.training_config.eval_every_n_steps
    except Exception:
      eval_every = None
    current_train_steps = self.rl_cluster.actor_trainer.train_steps
    print(f"[VALIDATION] Start rollout | train_step={current_train_steps} eval_every_n_steps={eval_every}")

    # Run a single validation rollout (no filtering)
    mt_batch = self.validation_multi_turn_rollout.rollout()

    # Collect metrics directly from rollout batch
    metrics_dict = dict(mt_batch.meta_info.get("metrics", {}))

    # Use the actor trainer's train_steps for eval logging step to avoid out-of-order issues
    eval_log_step = current_train_steps
    for name, value in metrics_dict.items():
      try:
        self._actor_metrics_logger.log(name, float(value), metrics_logger.Mode.EVAL, eval_log_step)
      except Exception:
        pass

    # Also log completion lengths from loss_mask for quick sanity
    try:
      _cm = np.array(mt_batch.loss_mask)
      _agg = _cm.sum(axis=-1)
      self._actor_metrics_logger.log("completions/mean_length", float(_agg.mean()), metrics_logger.Mode.EVAL, eval_log_step)
      self._actor_metrics_logger.log("completions/max_length", float(_agg.max()), metrics_logger.Mode.EVAL, eval_log_step)
      self._actor_metrics_logger.log("completions/min_length", float(_agg.min()), metrics_logger.Mode.EVAL, eval_log_step)
    except Exception:
      pass

    # Reset to free memory and bump local eval counter
    self.validation_multi_turn_rollout.reset()
    self._eval_steps += 1
    print(f"[VALIDATION] Done | logged {len(metrics_dict)} metrics at step={eval_log_step}")
  # ─────────────────── END MODIFICATION ───────────────────

  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """PPO training loop."""
    # ─────────────────── MODIFICATION: Initial validation at step 0 ───────────────────
    if not hasattr(self, "_initial_eval_done") or not self._initial_eval_done:
      try:
        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation("sync_sampler_weights", step_num=0):
            self.rl_cluster.sync_weights()
      except Exception:
        pass
      print("[VALIDATION] Initial evaluation at train_step=0")
      self._validate(None)
      self._initial_eval_done = True
    # ─────────────────── END MODIFICATION ───────────────────
    train_iterator = iter(train_ds)
    while True:  # loop over M
      try:
        # reserve 1 for None and the other for repeated interable
        # if batch_repeat > 1
        train_data_queue = queue_lib.SimpleDataQueue(
            maxsize=self.grad_acc_steps + 2
        )
        # reserve 1 for None
        eval_data_queue = queue_lib.SimpleDataQueue(maxsize=2)
        initial_train_steps = self._train_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            mini_batch_size=self.ppo_config.mini_batch_size,
            proceed_num_steps=self.grad_acc_steps,
            batch_repeat=self.ppo_config.num_ppo_epochs,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=metrics_logger.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_train_steps
        ):
          while True:
            curr_train_ds = train_data_queue.get(block=True)
            if curr_train_ds is None:
              break
            if eval_ds and not curr_eval_ds:
              self._prepare_data(
                  iterator=iter(eval_ds),
                  mini_batch_size=None,
                  proceed_num_steps=-1,
                  batch_repeat=1,
                  data_queue=eval_data_queue,
                  async_loading=False,
                  mode=metrics_logger.Mode.EVAL,
              )
              curr_eval_ds = eval_data_queue.get(block=True)
            self.rl_cluster.update_actor(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
            self.rl_cluster.update_critic(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
        # call to throw stop iteration as a signal to break the loop
        future.result()
        # Sync the train steps with internal trainer, this is based on the
        # assumption that the trainer internally doesn't reset the train steps.
        self._train_steps = self.rl_cluster.actor_trainer.train_steps
        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_train_steps
          ):
            self.rl_cluster.sync_weights()
        # ─────────────────── MODIFICATION: Robust eval trigger on crossed intervals (after weight sync) ───────────────────
        # Fire validation once when we cross an eval interval boundary, AFTER syncing sampler weights
        # so evaluation uses up-to-date rollout parameters.
        try:
          eval_every = self.rl_cluster.cluster_config.training_config.eval_every_n_steps
        except Exception:
          eval_every = 0
        if not hasattr(self, "_last_eval_check_step"):
          self._last_eval_check_step = 0
        if eval_every and eval_every > 0:
          prev_q = self._last_eval_check_step // eval_every
          curr_q = self._train_steps // eval_every
          if curr_q > prev_q and self._train_steps > 0:
            print(f"[VALIDATION] Triggering (crossed interval) at train_step={self._train_steps} (eval_every_n_steps={eval_every})")
            self._validate(None)
          self._last_eval_check_step = self._train_steps
        # ─────────────────── END MODIFICATION ───────────────────
        if (
            self._train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    # Close critic before actor to ensure any final JAX monitoring callbacks
    # (which may indirectly call wandb.log) occur while the W&B run is active.
    self.rl_cluster.critic_trainer.close()
    self.rl_cluster.actor_trainer.close()
    # Ensure rollout managers are closed to release any held resources
    try:
      if self.multi_turn_rollout is not None:
        self.multi_turn_rollout.close()
    except Exception:
      pass
    try:
      if self.validation_multi_turn_rollout is not None:
        self.validation_multi_turn_rollout.close()
    except Exception:
      pass
