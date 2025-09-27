"""Experimental trainer classes extending PPO components.

This module imports `PpoLearner`, `PpoConfig`, and `TrainExample` from
`tunix.rl.ppo.ppo_learner` and provides thin subclasses for experimentation.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence
import os
import time
import logging

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.ppo import ppo_helpers
from tunix.rl.ppo.ppo_learner import PpoConfig, PpoLearner, TrainExample
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

from grl.rollout.tunix_sync_multi_turn_rollout import SyncMultiTurnRollout

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


def _assert_float32(x, name: str):
  """Assert that array-like `x` is float32. Converts to jnp.array for robust dtype access."""
  arr = jnp.asarray(x)
  assert (
      arr.dtype == jnp.float32
  ), f"{name} dtype must be float32, got {arr.dtype}"


@dataclasses.dataclass(slots=True, kw_only=True)
class PpoConfigExp(PpoConfig):
  """Experimental `PpoConfig` with fixed completion length for stable shapes."""

  max_completion_length: int | None = None


@flax.struct.dataclass(frozen=True)
class TrainExampleExp(TrainExample):
  """Placeholder subclass of `TrainExample` for future overrides."""

  pass


class PpoLearnerExp(PpoLearner):
  """Placeholder subclass of `PpoLearner`.

  Accepts and ignores extra keyword-only arguments so call sites can pass
  future options without breaking.
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      ppo_config: PpoConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
      *,
      multi_turn_cfg=None,
      multi_turn_processor=None,
      multi_turn_validation: bool = False,
  ):
    """Initializes the `PpoLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      ppo_config: An instance of `PpoConfig` containing all training-specific
        configuration options.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn):
        >>> def metric_fn(prompts, completions, rewards, advantages, **kargs):
        ...    return { ...        "prompt_min_len": (min(len(p) for p in
        prompts), np.min), ...        ... ...    }
      data_shuffle_seed: The seed for shuffling the data.
    """
    self.ppo_config = ppo_config
    super().__init__(
        rl_cluster=rl_cluster,
        ppo_config=ppo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    # ===== RlCluster should have `reward` and `critic` models =====
    if bool(reward_fns) == bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    ):
      raise ValueError(
          "PPO requires one of `reward_fns` or `rl_cluster.reward` to be set. "
          f"Received: reward_fn={reward_fns}, "
          "rl_cluster.reward="
          f"{self.rl_cluster.inference_worker._models['reward']}"
      )
    if not self.rl_cluster.inference_worker._models["critic"]:
      raise ValueError(
          "PPO requires a critic model. Please pass the correct `critic` to "
          "`RlCluster`."
      )
    # self._use_reward_model = bool(
    #     self.rl_cluster.inference_worker._models.get("reward", None)
    # )
    self._use_reward_model = False

    # ===== Configure the actor (policy) trainer =====
    # Use customized policy loss that accepts completion_mask
    self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "epsilon_low": self.ppo_config.epsilon_low,
            "epsilon_high": self.ppo_config.epsilon_high,
            "epsilon_c": self.ppo_config.epsilon_c,
            "entropy_coef": self.ppo_config.entropy_coef,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )

    # ===== Configure the critic (value) trainer =====
    # Use customized critic loss that accepts completion_mask
    self.rl_cluster.critic_trainer.with_loss_fn(ppo_value_loss_fn, has_aux=True)
    self.rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": self.ppo_config.clip_range_value,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )
    self.rl_cluster.critic_trainer.is_managed_externally = True

    # ===== Configure the metrics logger =====
    # We just log the metrics returned in `aux`. All other metrics are logged
    # by `RLCluster` itself.
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.ppo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    if (
        self.ppo_config.entropy_coef is not None
        and self.ppo_config.entropy_coef > 0.0
    ):
      actor_rl_metrics_to_log["loss/entropy"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log
    )

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log(
        {
            "vpred_mean": np.mean,
            "vf_clipfrac": np.mean,
        }
    )

    # ====== Modification: add multi-turn rollout initialization =====
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
    # ======= End Modification =====

  # ====== Debug helper =====
  def _dbg(self, message: str) -> None:
    if not hasattr(self, "_debug"):
      self._debug = os.getenv("GRL_DEBUG_TRAIN_LOOP", "1") != "0"
    if not self._debug:
      return
    # Timer store for phase durations
    if not hasattr(self, "_dbg_phase_start_ns"):
      self._dbg_phase_start_ns = {}
    # Derive step
    try:
      step = self.rl_cluster.actor_trainer.train_steps
    except Exception:
      step = "-"
    # Parse simple "phase: detail" pattern into structured fields
    phase = "misc"
    event = "log"
    details = ""
    msg = message.strip()
    if ":" in msg:
      left, right = msg.split(":", 1)
      phase = left.strip().replace(" ", "_") or "misc"
      rest = right.strip()
      if rest:
        if " " in rest:
          first, rem = rest.split(" ", 1)
          event = first.strip().replace(" ", "_") or "log"
          details = rem.strip()
        else:
          event = rest.strip().replace(" ", "_") or "log"
    else:
      details = msg

    # Track start/end to compute durations
    duration_suffix = ""
    try:
      now_ns = time.monotonic_ns()
      if event in {"start", "begin"}:
        self._dbg_phase_start_ns[phase] = now_ns
      elif event in {"done", "end", "stop"}:
        start_ns = self._dbg_phase_start_ns.pop(phase, None)
        if start_ns is not None:
          elapsed_ms = (now_ns - start_ns) / 1_000_000.0
          duration_suffix = f" duration_ms={elapsed_ms:.3f}"
    except Exception:
      pass

    logger = logging.getLogger("trainer")
    if details:
      logger.info(
          "phase=%s event=%s step=%s%s | %s",
          phase,
          event,
          step,
          duration_suffix,
          details,
      )
    else:
      logger.info(
          "phase=%s event=%s step=%s%s", phase, event, step, duration_suffix
      )

  # ======= End Debug helper =====

  # ====== Modification: add helper to convert multi-turn rollout batch =====
  def convert_multi_rollout_batch(
      self,
      batch,
      *,
      pad_value: int,
      max_prompt_length: int,
  ):
    """Convert a multi-turn rollout batch to JAX tensors for PPO.

    Splitting rule:
    - Fixed prompt length Pmax=1: prompt is the first token only
    - completion_ids are all tokens after the first token
    - completion_mask is the provided loss_mask
    - eos_idx is derived from completion_mask
    """
    inp = np.array(batch.input_ids)  # [B, L]
    loss_m = np.array(batch.loss_mask)  # [B, L-1], values in {0,1} (int32)
    B, L = inp.shape

    # Fixed Pmax = 1
    prompt_ids = jnp.array(inp[:, :1])  # [B, 1]
    prompt_mask = (prompt_ids != pad_value).astype("int32")

    # Completion is tokens after the first token; ensure at least width 1
    if L - 1 <= 0:
      completion_ids_arr = np.full((B, 1), pad_value, dtype=inp.dtype)
      completion_mask_arr = np.zeros((B, 1), dtype=np.int32)
    else:
      completion_ids_arr = inp[:, 1:]
      # loss_m is already int32 from rollout; reuse directly without casting
      completion_mask_arr = loss_m
    completion_ids = jnp.array(completion_ids_arr)
    # Keep mask as int32 once to avoid repeated casts downstream
    completion_mask = jnp.array(completion_mask_arr)

    # Right-pad/truncate to fixed max_completion_length if configured
    t_max = getattr(self.ppo_config, "max_completion_length", None)
    if t_max is not None and t_max > 0:
      t_cur = completion_ids.shape[1]
      if t_cur < t_max:
        pad_w = t_max - t_cur
        completion_ids = jnp.pad(
            completion_ids, ((0, 0), (0, pad_w)), constant_values=pad_value
        )
        completion_mask = jnp.pad(
            completion_mask, ((0, 0), (0, pad_w)), constant_values=0
        )
      elif t_cur > t_max:
        completion_ids = completion_ids[:, :t_max]
        completion_mask = completion_mask[:, :t_max]
    # Derive eos_idx from completion_mask for robustness
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    ).astype(jnp.int32)

    return prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx

  # ======= End Modification =====

  # ====== Modification: Simple validation rollout using buffered metrics (EVAL) =====
  def _validate(self, eval_ds: Iterable[TrainingInputT] | None) -> None:
    """Run one validation rollout and log metrics via buffer_metrics under EVAL.

    - Ignores `eval_ds`; data comes from the environment.
    - No advantage computation or model updates.
    - Uses RLCluster.buffer_metrics so logs are aggregated on global steps.
    """
    if self.validation_multi_turn_rollout is None:
      if self._multi_turn_cfg is None:
        raise RuntimeError(
            "Multi-turn validation requested but no multi_turn_cfg provided."
        )
      self.validation_multi_turn_rollout = SyncMultiTurnRollout(
          rl_cluster=self.rl_cluster,
          cfg=self._multi_turn_cfg,
          tokenizer=self.rl_cluster.tokenizer,
          processor=self._multi_turn_processor,
          validation=True,
      )

    # Run a single validation rollout (no filtering)
    self._dbg("rollout_val: start")
    mt_batch = self.validation_multi_turn_rollout.rollout()
    self._dbg("rollout_val: done")

    # === metrics (EVAL): rollout meta metrics ===
    metrics_dict = dict(mt_batch.meta_info.get("metrics", {}))
    try:
      for name, value in metrics_dict.items():
        self.rl_cluster.buffer_metrics(
            {name: (float(value), np.mean)}, mode=rl_cluster_lib.Mode.EVAL
        )
    except Exception:
      pass

    # === metrics (EVAL): completion lengths from loss_mask ===
    try:
      _cm = np.array(mt_batch.loss_mask)
      _agg = _cm.sum(axis=-1)
      self.rl_cluster.buffer_metrics(
          {
              "completions/mean_length": (float(_agg.mean()), np.mean),
              "completions/max_length": (float(_agg.max()), np.max),
              "completions/min_length": (float(_agg.min()), np.min),
          },
          mode=rl_cluster_lib.Mode.EVAL,
      )
    except Exception:
      pass

    self.validation_multi_turn_rollout.reset()

  # ======= End Modification =====

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
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

    # TODO(abheesht): verl allows specifying different micro batch sizes for
    # computing log probs, values, rewards, etc. We can do that here.

    # ===== Generation ======
    # ====== Modification: Enforce multi-turn rollout only =====
    if getattr(self, "multi_turn_rollout", None) is None:
      raise RuntimeError("Multi-turn rollout is required but not configured.")

    # Collect rollout metrics for buffered logging
    mt_metrics_dict = None
    meta_metrics_dict = None

    try:
      self._dbg("rollout: start")
      mt_batch = self.multi_turn_rollout.rollout()
      mt_batch_filtered, mt_metrics = (
          self.multi_turn_rollout.filter_rollout_batch(mt_batch)
      )
      self._last_rollout_batch = mt_batch_filtered
      # Prepare metrics dictionaries (filter metrics + meta metrics)
      try:
        mt_metrics_dict = dict(mt_metrics)
      except Exception:
        mt_metrics_dict = None
      try:
        meta_metrics_dict = dict(mt_batch_filtered.meta_info.get("metrics", {}))
      except Exception:
        meta_metrics_dict = None
      self._dbg("rollout: done")
    finally:
      self.multi_turn_rollout.reset()

    if getattr(self, "_last_rollout_batch", None) is None:
      raise RuntimeError(
          "Multi-turn rollout is required but missing _last_rollout_batch."
      )

    # Log basic batch shape from rollout (removed detailed shape logging)

    prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx = (
        self.convert_multi_rollout_batch(
            self._last_rollout_batch, pad_value=pad_value, max_prompt_length=0
        )
    )

    # Prepare a float mask only once where needed for arithmetic
    completion_mask_f = completion_mask.astype(jnp.float32)
    # ======= End Modification =====

    batch_size = completion_ids.shape[0]
    logits_to_keep = completion_ids.shape[1]

    # ===== Compute log probs ======
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          completion_mask=completion_mask,
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
        completion_mask=completion_mask,
    )

    # ===== Value computation ======
    # Get values from the value model before model weights are updated.
    values = self.rl_cluster.get_values(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
        completion_mask=completion_mask,
    )
    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    values = values * completion_mask_f

    # ===== Reward computation ======
    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )[:, -logits_to_keep:]
      # We use the score corresponding to the last non-padding token.
      last_token_scores = scores[jnp.arange(batch_size), eos_idx]
    else:
      # ====== Modification: use rollout reward_scores (last column) =====
      reward_scores = jnp.asarray(self._last_rollout_batch.reward_scores)
      last_token_scores = reward_scores[:, -1]
      # ======= End Modification =====

    # Reward computation is in accordance with TRL and verl's
    # `BatchRewardManager` (token-level rewards).
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. A positive reward is given only at the final timestep, so we add that
    # to the tensor of zeros.
    # 3. Subtract KL divergence from the reward tensor.
    # ====== Modification: dtype-safe reward tensor and mask-aware KL penalty =====
    rewards = jnp.zeros_like(completion_ids, dtype=jnp.float32)
    last_token_scores = jnp.asarray(last_token_scores, dtype=jnp.float32)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].set(last_token_scores)
    if self.ppo_config.beta != 0.0:
      kl = common.compute_kl_divergence(
          old_per_token_logps,
          ref_per_token_logps,
          method=self.ppo_config.kl_method,
      )
      rewards = rewards - self.ppo_config.beta * (kl * completion_mask_f)
    # ======= End Modification =====

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantages, returns = ppo_helpers.compute_gae_advantages(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask_f,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )

    # ===== Rollout metrics logging (from filter/meta) =====
    if mt_metrics_dict is not None:
      try:
        for name, value in mt_metrics_dict.items():
          self.rl_cluster.buffer_metrics(
              {name: (float(value), np.mean)}, mode=mode
          )
      except Exception:
        pass
    if meta_metrics_dict is not None:
      try:
        for name, value in meta_metrics_dict.items():
          self.rl_cluster.buffer_metrics(
              {name: (float(value), np.mean)}, mode=mode
          )
      except Exception:
        pass

    # ===== Metric logging ======
    # Log raw scores from the reward model fn
    self.rl_cluster.buffer_metrics(
        {
            "score/mean": (np.mean(last_token_scores), np.mean),
            "score/max": (np.max(last_token_scores), np.max),
            "score/min": (np.min(last_token_scores), np.min),
        },
        mode=mode,
    )

    # Log final rewards (scores + KL penalty)
    sequence_rewards = rewards.sum(-1)
    self.rl_cluster.buffer_metrics(
        {
            "reward/mean": (np.mean(sequence_rewards), np.mean),
            "reward/max": (np.max(sequence_rewards), np.max),
            "reward/min": (np.min(sequence_rewards), np.min),
        },
        mode=mode,
    )
    if self.ppo_config.beta != 0.0:
      # Average of the per-sequence mean KL
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
      )
      self.rl_cluster.buffer_metrics(
          {
              "reward_kl_penalty": (
                  per_sequence_mean_kl.mean(),
                  np.mean,
              ),
          },
          mode=mode,
      )
      # ===== MODIFICATION: Log KL penalty coefficient (beta) =====
      self.rl_cluster.buffer_metrics(
          {
              "actor/reward_kl_penalty_coeff": (
                  float(self.ppo_config.beta),
                  np.mean,
              )
          },
          mode=mode,
      )
      # ===== END MODIFICATION =====

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )

    # Log advantages.
    valid_advantages = np.ma.masked_array(
        advantages, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "advantages/mean": (valid_advantages.mean(), np.mean),
            "advantages/max": (valid_advantages.max(), np.max),
            "advantages/min": (valid_advantages.min(), np.min),
        },
        mode=mode,
    )

    # Log returns.
    valid_returns = np.ma.masked_array(
        returns, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "returns/mean": (valid_returns.mean(), np.mean),
            "returns/max": (valid_returns.max(), np.max),
            "returns/min": (valid_returns.min(), np.min),
        },
        mode=mode,
    )

    # Log values.
    valid_values = np.ma.masked_array(
        values, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "old_values/mean": (valid_values.mean(), np.mean),
            "old_values/max": (valid_values.max(), np.max),
            "old_values/min": (valid_values.min(), np.min),
        },
        mode=mode,
    )

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_per_token_logps=old_per_token_logps,
        old_values=values,
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is same as the offset of the example in the data source.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self._num_generations()
    row_offset = steps * batch_size
    row_offsets = np.arange(row_offset, row_offset + batch_size)
    return row_offsets.astype(str).tolist()

  def _num_iterations(self) -> int:
    return self.ppo_config.num_ppo_epochs

  def _num_generations(self) -> int:
    return 1

  def train(
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main entry point for the training loop."""
    # # ====== Modification: Guarded initial validation at non-decreasing step =====
    # # Ensure first validation logs at step >= 1 to avoid out-of-order logging.
    # if not hasattr(self, "_initial_eval_done") or not self._initial_eval_done:
    #   try:
    #     if self.should_sync_weights:
    #       with jax.profiler.StepTraceAnnotation(
    #           "sync_sampler_weights", step_num=0
    #       ):
    #         self.rl_cluster.sync_weights()  # increments global_steps
    #     else:
    #       # No sync -> bump global_steps to at least 1 so eval logs at step 1
    #       try:
    #         if getattr(self.rl_cluster, "global_steps", 0) < 1:
    #           self.rl_cluster.global_steps = 1
    #       except Exception:
    #         pass
    #   except Exception:
    #     pass
    #   self._dbg("initial_validation: start")
    #   self._validate(None)
    #   self._dbg("initial_validation: done")
    #   self._initial_eval_done = True
    # ======= End Modification =====

    full_batch_iterator = iter(train_ds)
    first_item = next(full_batch_iterator)
    full_batch_size = len(first_item["prompts"])
    full_batch_iterator = itertools.chain([first_item], full_batch_iterator)
    # Initialize batch sizes and gradient accumulation steps.
    (
        training_mini_batch_sizes,
        training_micro_batch_sizes,
    ) = self._initialize_batch_sizes(full_batch_size)
    # if the micro batch size is the same as the full batch size, we can use the
    # full batch iterator directly.
    if training_micro_batch_sizes == full_batch_size:
      train_iterator = full_batch_iterator
    else:
      train_iterator = self._create_micro_batch_iterator(
          full_batch_iterator, training_micro_batch_sizes
      )

    while True:  # loop over M
      try:
        # Track per-global-step training loop duration
        self._dbg("train_loop: start")
        initial_steps = self._iter_steps
        for _ in range(full_batch_size // training_mini_batch_sizes):
          # reserve 1 for None and the other for repeated interable
          # if batch_repeat > 1
          train_data_queue = queue_lib.SimpleDataQueue(
              maxsize=self.grad_acc_steps * self._num_iterations() + 1
          )
          # Use an unbounded queue for evaluation data.
          eval_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
          initial_steps = self._iter_steps
          future = self.executor.submit(
              self._prepare_data,
              iterator=train_iterator,
              proceed_num_steps=self.grad_acc_steps,
              sample_repeat=self._num_generations(),
              batch_repeat=self._num_iterations(),
              data_queue=train_data_queue,
              async_loading=self.can_enable_async_rollout,
              mode=rl_cluster_lib.Mode.TRAIN,
          )

          curr_eval_ds = None
          with jax.profiler.StepTraceAnnotation(
              "trainer", step_num=initial_steps
          ):
            while True:
              with sft_utils.time_measure(suppress_logging=True) as timer:
                curr_train_ds = train_data_queue.get(block=True)

              if curr_train_ds is None:
                break

              if self.can_enable_async_rollout:
                self.rl_cluster.buffer_metrics(
                    {
                        "actor_dequeue_time": (
                            timer(),
                            np.mean,
                        ),
                    },
                    mode=rl_cluster_lib.Mode.TRAIN,
                )

              # ===== Modification: disable inline eval data preparation in inner loop =====
              # if (
              #     eval_ds
              #     and not curr_eval_ds
              #     and self.rl_cluster.actor_trainer.train_steps
              #     % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
              #     == 0
              # ):
              #   self._eval_iter_steps = 0
              #   self._dbg("eval: prepare eval data")
              #   self._prepare_data(
              #       iterator=iter(eval_ds),
              #       proceed_num_steps=-1,
              #       sample_repeat=self._num_generations(),
              #       batch_repeat=1,
              #       data_queue=eval_data_queue,
              #       async_loading=False,
              #       mode=rl_cluster_lib.Mode.EVAL,
              #   )
              #   curr_eval_ds = eval_data_queue.get(block=True)
              #   self._dbg("eval: eval batch ready")
              # ===== End Modification =====
              self.rl_cluster.update_actor(
                  curr_train_ds,
                  curr_eval_ds,
                  skip_jit,
              )  # loop over μ
              if hasattr(self.rl_cluster, "critic_trainer"):
                self.rl_cluster.update_critic(
                    curr_train_ds,
                    curr_eval_ds,
                    skip_jit,
                )  # loop over μ

          # call to throw stop iteration as a singal to break the loop
          future.result()
          # sync the iter steps with internel trainer, this is based on the
          # assumption that the trainer internally doesn't reset the iter steps.
          # there is current a unit test to ensure this assumption.
          self._iter_steps = self.rl_cluster.actor_trainer.iter_steps

        # ====== Modification: Pre-sync validation so TRAIN and EVAL share the same logging step =====
        try:
          eval_every = (
              self.rl_cluster.cluster_config.training_config.eval_every_n_steps
          )
        except Exception:
          eval_every = 0
        if eval_every and eval_every > 0:
          grad_acc = getattr(self, "grad_acc_steps", 1) or 1
          denom = max(1, eval_every // grad_acc)
          # Evaluate just before we advance global_steps so train/eval log at the same step.
          if (self.rl_cluster.global_steps + 1) % denom == 0:
            self._validate(None)
        # ======= End Modification =====

        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_steps
          ):
            self.rl_cluster.sync_weights()
        else:
          self.rl_cluster.global_steps += (
              1  # manually increment the global steps.
          )
        # End of per-global-step training loop timing
        self._dbg("train_loop: done")
        if (
            self.rl_cluster.actor_trainer.train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    self.rl_cluster.close()


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  logits_to_keep = completion_ids.shape[1]

  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns
  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      completion_mask=completion_mask,
      stop_gradient=False,
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
  # "token mean" style of normalisation.
  vf_loss = ppo_helpers.masked_mean(clipped_vf_losses, completion_mask)
  vf_loss = 0.5 * vf_loss

  aux = {
      "vpred_mean": ppo_helpers.masked_mean(vpreds, completion_mask),
      "vf_clipfrac": ppo_helpers.masked_mean(
          (vf_losses2 > vf_losses1).astype(jnp.float32), completion_mask
      ),
  }
  return vf_loss, aux


def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon_low: float,
    epsilon_high: float,
    epsilon_c: float | None,
    entropy_coef: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the policy loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  use_dual_clip_ppo = epsilon_c is not None

  # Get log probs.
  per_token_logps, logits = common.compute_per_token_logps(
      model,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      completion_mask=completion_mask,
      stop_gradient=False,
  )

  advantages = train_example.advantages

  # Compute ratio.
  old_per_token_logps = train_example.old_per_token_logps
  ratio = jnp.exp(per_token_logps - old_per_token_logps)
  ratio_clipped = jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)

  # Vanilla PPO loss
  pg_losses_1 = -ratio * advantages
  pg_losses_2 = -ratio_clipped * advantages
  clip_pg_losses_1 = jnp.maximum(pg_losses_1, pg_losses_2)

  # Dual-clip PPO to avoid negative-advantage policy updates
  pg_losses = clip_pg_losses_1
  if use_dual_clip_ppo:
    pg_losses_3 = -epsilon_c * advantages
    clip_pg_losses_2 = jnp.minimum(pg_losses_3, clip_pg_losses_1)

    pg_losses = jnp.where(advantages < 0.0, clip_pg_losses_2, clip_pg_losses_1)

    # For logging.
    unreduced_pg_clipfrac_lower = (
        (clip_pg_losses_1 > pg_losses_3) & (advantages < 0.0)
    ).astype(jnp.float32)
    pg_clipfrac_lower = ppo_helpers.masked_mean(
        unreduced_pg_clipfrac_lower, completion_mask
    )

  # Logging
  aux = {
      "pg_clipfrac": ppo_helpers.masked_mean(
          (pg_losses_2 > pg_losses_1).astype(jnp.float32), completion_mask
      ),
  }
  if use_dual_clip_ppo:
    aux["pg_clipfrac_lower"] = pg_clipfrac_lower  # pylint: disable=undefined-variable

  # "token mean" style of normalisation
  policy_loss = ppo_helpers.masked_mean(pg_losses, completion_mask)

  # Compute entropy loss.
  if entropy_coef is not None and entropy_coef > 0.0:
    token_entropy = ppo_helpers.compute_entropy_from_logits(logits)
    # "token mean" style of normalisation.
    entropy_loss = ppo_helpers.masked_mean(token_entropy, completion_mask)
    policy_loss -= entropy_coef * entropy_loss

    # Logging
    aux["loss/entropy"] = entropy_loss

  return policy_loss, aux
