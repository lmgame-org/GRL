"""Experimental trainer classes extending PPO components.

This module imports `PpoLearner`, `PpoConfig`, and `TrainExample` from
`tunix.rl.ppo.ppo_learner` and provides thin subclasses for experimentation.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence

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
  """Subclass of `PpoConfig` for experimental overrides."""

  # KL penalty method for reward shaping: one of {"kl","k1","abs","mse","k2","low_var_kl","k3"}
  kl_penalty_method: str = "k1"
  # Entropy regularization
  entropy_coeff: float = 0.0
  aggs_mode: str = "token-mean"


@flax.struct.dataclass(frozen=True)
class TrainExampleExp(TrainExample):
  """Extended train example with entropy fields for experimental loss."""

  entropy_coeff: jax.Array | float = flax.struct.field(
      pytree_node=False, default=0.0
  )
  aggs_mode: str = flax.struct.field(pytree_node=False, default="token-mean")


class PpoLearnerExp(PpoLearner):
  """PPO (Proximal Policy Optimization) learner.

  PPO is a reinforcement learning algorithm that fine-tunes models using an
  actor-critic architecture. It optimizes a clipped surrogate objective function
  to ensure stable policy updates, preventing large, destructive changes. The
  actor (policy model) learns what actions to take, while the critic (value
  model) estimates the value of states to help calculate advantages. This
  approach balances exploration and exploitation, making it a robust choice for
  a wide range of RL tasks.

  References:
  - https://arxiv.org/abs/1707.06347
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      ppo_config: PpoConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
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
    self._use_reward_model = bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    )

    # ===== Configure the actor (policy) trainer =====
    self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "epsilon_low": self.ppo_config.epsilon_low,
            "epsilon_high": self.ppo_config.epsilon_high,
            "epsilon_c": self.ppo_config.epsilon_c,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )

    # ===== Configure the critic (value) trainer =====
    self.rl_cluster.critic_trainer.with_loss_fn(ppo_value_loss_fn, has_aux=True)
    self.rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": self.ppo_config.clip_range_value,
            "vf_coef": self.ppo_config.vf_coef,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )
    self.rl_cluster.critic_trainer.is_managed_externally = True

    # ===== Configure the metrics logger =====
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.ppo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log
    )
    # TODO(tsbao): this need to be fixed, currently it won't display in tqdm
    # since these metrics are logged in rl_cluster and aggregated at global step
    # level.
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display(
        [
            "score/mean",
            "reward/mean",
            lambda: "reward_kl_penalty"
            if self.ppo_config.beta != 0.0
            else None,
        ]
    )

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log(
        {
            "vpred_mean": np.mean,
            "vf_clipfrac": np.mean,
        }
    )

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
    # Generate. We use `model`, i.e., the policy model for generating the
    # "experiences".
    completion_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
    )
    completion_ids = completion_output.tokens
    prompt_ids = completion_output.left_padded_prompt_tokens

    batch_size = completion_ids.shape[0]
    logits_to_keep = completion_ids.shape[1]
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    )

    # ===== Compute log probs ======
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
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
    values = values * completion_mask

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
      last_token_scores = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=completion_output.text,
          mode=mode,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )

    # Reward computation is in accordance with TRL and verl's
    # `BatchRewardManager` (token-level rewards).
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. A positive reward is given only at the final timestep, so we add that
    # to the tensor of zeros.
    # 3. Subtract KL divergence from the reward tensor.
    rewards = jnp.zeros_like(completion_ids)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)
    if self.ppo_config.beta != 0.0:
      # TODO(abheesht): Add a toggle - KL can either be added directly to
      # rewards or computed in the loss function.
      kl = common.compute_kl_divergence(
          old_per_token_logps, ref_per_token_logps
      )
      kl = kl * completion_mask
      rewards = rewards - self.ppo_config.beta * kl

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantages, returns = ppo_helpers.compute_gae_advantages(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )

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
    train_iterator = iter(train_ds)
    first_item = next(train_iterator)
    input_batch_size = len(first_item["prompts"])
    train_iterator = itertools.chain([first_item], train_iterator)
    self._initialize_micro_batch_sizes(input_batch_size)

    while True:  # loop over M
      try:
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

            if (
                eval_ds
                and not curr_eval_ds
                and self.rl_cluster.actor_trainer.train_steps
                % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                == 0
            ):
              self._eval_iter_steps = 0
              self._prepare_data(
                  iterator=iter(eval_ds),
                  proceed_num_steps=-1,
                  sample_repeat=self._num_generations(),
                  batch_repeat=1,
                  data_queue=eval_data_queue,
                  async_loading=False,
                  mode=rl_cluster_lib.Mode.EVAL,
              )
              curr_eval_ds = eval_data_queue.get(block=True)
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

        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_steps
          ):
            self.rl_cluster.sync_weights()
        else:
          self.rl_cluster.global_steps += (
              1  # manually increment the global steps.
          )
        if (
            self.rl_cluster.actor_trainer.train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    self.rl_cluster.close()
