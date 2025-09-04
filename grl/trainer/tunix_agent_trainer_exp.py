"""Experimental trainer classes extending PPO components.

This module imports `PpoLearner`, `PpoConfig`, and `TrainExample` from
`tunix.rl.ppo.ppo_learner` and provides thin subclasses for experimentation.
"""

from __future__ import annotations

# Standard library
from concurrent import futures
import dataclasses
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

# Third-party
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np

# Local application
from tunix.rl.ppo.ppo_learner import (
    PpoLearner,
    PpoConfig,
    TrainExample,
)
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.ppo import ppo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import metrics_logger
from grl.rollout.tunix_sync_multi_turn_rollout import SyncMultiTurnRollout

_TrainingInputT = Dict[str, List[str] | ArrayLike]



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

    entropy_coeff: jax.Array | float = flax.struct.field(pytree_node=False, default=0.0)
    aggs_mode: str = flax.struct.field(pytree_node=False, default="token-mean")


class PpoLearnerExp(PpoLearner):
    """Subclass of `PpoLearner` for experimental overrides."""

    def __init__(
        self,
        rl_cluster,
        ppo_config,
        reward_fns=None,
        data_shuffle_seed=None,
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

        # ─────────────────── MODIFICATION: attach custom loss fns to trainers ───────────────────
        # Ensure actor uses experimental policy loss that returns entropy aux metrics
        try:
            self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)
        except Exception:
            pass
        # Ensure critic uses local value loss
        try:
            self.rl_cluster.critic_trainer.with_loss_fn(ppo_value_loss_fn, has_aux=True)
        except Exception:
            pass
        # ─────────────────── END MODIFICATION ───────────────────

        # -======== Modiifcation: add multi-turn rollout initialization =====
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
        # ======= End Modificafiont ====

    def convert_multi_rollout_batch(
        self,
        batch,
        *,
        pad_value: int,
        max_prompt_length: int,
    ):
        """Convert a multi-turn rollout batch to JAX tensors for PPO.

        Splitting rule:
        - loss_mask 0 -> prompt tokens (prefix of zeros)
        - loss_mask 1 -> completion tokens (suffix starting at first 1)
        - prompt is left-padded to max_prompt_length
        - completion_mask is all 1s for the completion length
        - eos_idx is the index of the final completion token (end of completions)
        """
        inp = np.array(batch.input_ids)  # [B, L]
        loss_m = np.array(batch.loss_mask)  # [B, L-1], values in {0,1}
        B, L = inp.shape

        # Determine split point where completions start: first 1 in loss_m
        # loss_m indexes targets for next tokens, so split index in input_ids
        # aligns to the position of that first 1.
        has_one = (loss_m == 1).any(axis=1)
        first_one = np.where(has_one, (loss_m == 1).argmax(axis=1), L)  # position in loss_m
        comp_start = first_one  # input index where completion begins
        prompt_len = comp_start
        comp_len = L - comp_start

        # Targets
        target_P = int(max_prompt_length) if int(max_prompt_length) > 0 else int(prompt_len.max()) if B > 0 else 0
        max_comp_len = int(comp_len.max()) if B > 0 else 0

        prompt_ids_arr = np.full((B, target_P), pad_value, dtype=inp.dtype)
        completion_ids_arr = np.full((B, max_comp_len if max_comp_len > 0 else 1), pad_value, dtype=inp.dtype)
        completion_mask_arr = np.zeros_like(completion_ids_arr, dtype=np.int32)
        eos_idx_arr = np.zeros((B,), dtype=np.int32)

        for i in range(B):
            p_len = int(prompt_len[i])
            c_len = int(comp_len[i])
            # Fill prompt tokens (left-padded to target_P)
            p_tokens = inp[i, :p_len]
            if p_tokens.shape[0] > target_P:
                p_tokens = p_tokens[-target_P:]
            prompt_ids_arr[i, -p_tokens.shape[0] :] = p_tokens

            # Fill completion tokens (all unmasked targets; mask all ones)
            if c_len > 0:
                c_tokens = inp[i, int(comp_start[i]) : int(comp_start[i]) + c_len]
                completion_ids_arr[i, : c_tokens.shape[0]] = c_tokens
                completion_mask_arr[i, : c_tokens.shape[0]] = 1
                eos_idx_arr[i] = c_tokens.shape[0] - 1  # end of completions
            else:
                eos_idx_arr[i] = 0

        prompt_ids = jnp.array(prompt_ids_arr)
        completion_ids = jnp.array(completion_ids_arr)
        prompt_mask = (prompt_ids != pad_value).astype("int32")
        completion_mask = jnp.array(completion_mask_arr).astype("int32")
        # Derive eos_idx from completion_mask for robustness
        eos_idx = jnp.max(
            common.build_positions_from_mask(completion_mask),
            axis=-1,
        ).astype(jnp.int32)

        return prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx

    def _get_metric_logging_steps(self, mode: metrics_logger.Mode) -> int:
        return (
            self._train_steps
            if mode == metrics_logger.Mode.TRAIN
            else self._eval_steps
        )

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

        # TODO(abheesht): verl allows specifying different micro batch sizes for
        # computing log probs, values, rewards, etc. We can do that here.

        # -======== Modiifcation: Multi-turn rollout splitting (prompt/completion) =====
        # ===== Generation / Multi-turn conversion ======
        # Prefer multi-turn rollout if available; otherwise fallback to single-turn generation.
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

        rollout_cfg = _get_rollout_config_for_mode(mode)
        Pmax = int(getattr(rollout_cfg, "max_prompt_length", 0) or 0)

        prompt_ids = None
        completion_ids = None
        completion_mask = None
        eos_idx = None

        mt_metrics_dict = None
        meta_metrics_dict = None
        if self.multi_turn_rollout is not None:
            try:
                mt_batch = self.multi_turn_rollout.rollout()
                mt_batch_filtered, mt_metrics = self.multi_turn_rollout.filter_rollout_batch(mt_batch)
                self._last_rollout_batch = mt_batch_filtered
                # Defer logging of multi-turn metrics to the unified metric logging section
                mt_metrics_dict = dict(mt_metrics)
                try:
                    meta_metrics_dict = dict(mt_batch_filtered.meta_info.get("metrics", {}))
                except Exception:
                    meta_metrics_dict = None
            finally:
                self.multi_turn_rollout.reset()

        if getattr(self, "_last_rollout_batch", None) is None:
            raise RuntimeError("Multi-turn rollout is required but missing _last_rollout_batch.")

        prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx = self.convert_multi_rollout_batch(
            self._last_rollout_batch,
            pad_value=pad_value,
            max_prompt_length=Pmax,
        )
        # ======= End Modificafiont ====
        
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
          # ======== Modiifcation: use rollout reward_scores (last column) as terminal reward =====
          reward_scores = jnp.asarray(self._last_rollout_batch.reward_scores)
          last_token_scores = reward_scores[:, -1]
          # ======= End Modificafiont ====

        # Reward computation is in accordance with TRL and verl's
        # `BatchRewardManager` (token-level rewards).
        # 1. Set all rewards (i.e., for every token) to 0s.
        # 2. A positive reward is given only at the final timestep, so we add that
        # to the tensor of zeros.
        # 3. Subtract KL divergence from the reward tensor.
        rewards = jnp.zeros_like(completion_ids)
        rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)
        if self.ppo_config.beta != 0.0:
          # ================= MODIFICATION: Configurable KL penalty method aligned with TRL =================
          # Configurable KL penalty method aligned with TRL
          _kl_method = getattr(self.ppo_config, "kl_penalty_method", "k1")
          try:
            jax.debug.print("[PPO] KL penalty method: {}", _kl_method)
          except Exception:
            pass
          kl = common.compute_kl_divergence(
            per_token_logps=old_per_token_logps,
            ref_per_token_logps=ref_per_token_logps,
            method=_kl_method,
            )
          kl = kl * completion_mask
          rewards = rewards - self.ppo_config.beta * kl
          # ================= END MODIFICATION =================

        # ===== Compute advantages using Generalised Advantage Estimation ======
        advantages, returns = ppo_helpers.compute_gae_advantages(
            rewards=rewards,
            values=values,
            completion_mask=completion_mask,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
        )

        # ===== Metric logging ======
        step = self._get_metric_logging_steps(mode)

        # Log raw scores from the reward model fn
        self._actor_metrics_logger.log(
            "score/mean", np.mean(last_token_scores), mode, step
        )
        self._actor_metrics_logger.log(
            "score/max", np.max(last_token_scores), mode, step
        )
        self._actor_metrics_logger.log(
            "score/min", np.min(last_token_scores), mode, step
        )

        # Log final rewards (scores + KL penalty)
        sequence_rewards = rewards.sum(-1)
        self._actor_metrics_logger.log(
            "reward/mean", np.mean(sequence_rewards), mode, step
        )
        self._actor_metrics_logger.log(
            "reward/max", np.max(sequence_rewards), mode, step
        )
        self._actor_metrics_logger.log(
            "reward/min", np.min(sequence_rewards), mode, step
        )
        if self.ppo_config.beta != 0.0:
          # Average of the per-sequence mean KL
          per_sequence_mean_kl = ppo_helpers.masked_mean(
              kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
          )
          self._actor_metrics_logger.log(
              "reward_kl_penalty", per_sequence_mean_kl.mean(), mode, step
          )

        # Log multi-turn rollout metrics (from filter/meta) in the same block
        if mt_metrics_dict is not None:
          try:
            for name, value in mt_metrics_dict.items():
              self._actor_metrics_logger.log(name, float(value), mode, step)
          except Exception:
            pass
        if meta_metrics_dict is not None:
          try:
            for name, value in meta_metrics_dict.items():
              self._actor_metrics_logger.log(name, float(value), mode, step)
          except Exception:
            pass

        # Log completion lengths.
        agg_completion_mask = completion_mask.sum(axis=-1)
        self._actor_metrics_logger.log(
            "completions/mean_length",
            np.mean(agg_completion_mask),
            mode,
            self._get_metric_logging_steps(mode),
        )
        self._actor_metrics_logger.log(
            "completions/max_length",
            np.max(agg_completion_mask),
            mode,
            self._get_metric_logging_steps(mode),
        )
        self._actor_metrics_logger.log(
            "completions/min_length",
            np.min(agg_completion_mask),
            mode,
            self._get_metric_logging_steps(mode),
        )

        return TrainExampleExp(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            returns=returns,
            old_per_token_logps=old_per_token_logps,
            old_values=values,
            entropy_coeff=float(getattr(self.ppo_config, "entropy_coeff", 0.0)),
            aggs_mode=str(getattr(self.ppo_config, "aggs_mode", "token-mean")),
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
    def _validate(self, eval_ds: Iterable[_TrainingInputT] | None) -> None:
        """Run one validation rollout and log ONLY rollout metrics under EVAL.

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
                proceed_num_steps=1,
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
                if (
                    eval_ds
                    and not curr_eval_ds
                    and self.rl_cluster.actor_trainer.train_steps
                    % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                    == 0
                ):
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
        self.rl_cluster.actor_trainer.close()
        self.rl_cluster.critic_trainer.close()


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    vf_coef: float,
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
  return vf_coef * vf_loss, aux


def entropy_from_logits_jax(logits: jax.Array) -> jax.Array:
  logps = jax.nn.log_softmax(logits, axis=-1)
  ps = jnp.exp(logps)
  return -jnp.sum(ps * logps, axis=-1)


def agg_loss_jax(loss_mat: jax.Array, loss_mask: jax.Array, loss_agg_mode: str = "token-mean") -> jax.Array:
  loss_mask = loss_mask.astype(loss_mat.dtype)
  if loss_agg_mode == "token-mean":
    num = jnp.sum(loss_mat * loss_mask)
    den = jnp.sum(loss_mask) + 1e-8
    return num / den
  elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = jnp.sum(loss_mat * loss_mask, axis=-1)
    return jnp.mean(seq_losses)
  elif loss_agg_mode == "seq-mean-token-mean":
    token_counts = jnp.sum(loss_mask, axis=-1) + 1e-8
    seq_losses = jnp.sum(loss_mat * loss_mask, axis=-1) / token_counts
    return jnp.mean(seq_losses)
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
  """Computes the policy loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # Get log probs.
  per_token_logps, logits_slice = common.compute_per_token_logps_and_logits(
      model,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
  )

  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps
  coef_1 = jnp.exp(per_token_logps - old_per_token_logps)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)
  pg_losses1 = -coef_1 * jnp.expand_dims(advantages, 1)
  pg_losses2 = -coef_2 * jnp.expand_dims(advantages, 1)
  policy_loss = jnp.maximum(pg_losses1, pg_losses2)

  # "token mean" style of normalisation.
  policy_loss = ppo_helpers.masked_mean(policy_loss, completion_mask)

  aux = {
      "pg_clipfrac": ppo_helpers.masked_mean(
          (pg_losses2 > pg_losses1).astype(jnp.float32), completion_mask
      )
  }

  # Optional entropy regularization from logits
  entropy_coef = float(getattr(train_example, "entropy_coeff", 0.0))
  entropy_agg = str(getattr(train_example, "aggs_mode", "token-mean"))
  if entropy_coef != 0.0:
    token_entropy = entropy_from_logits_jax(logits_slice)
    entropy_loss = agg_loss_jax(token_entropy, completion_mask, entropy_agg)
    total_loss = policy_loss - entropy_coef * entropy_loss
    aux.update({
        "entropy/token_mean": (jnp.sum(token_entropy * completion_mask) / (jnp.sum(completion_mask) + 1e-8)),
        "loss/entropy": entropy_loss,
        "loss/total": total_loss,
    })
    # Best-effort visibility: print entropy metrics to stdout since aux isn't
    # auto-logged by the trainer. This helps verify entropy path execution.
    try:
      jax.debug.print(
        "[PPO/entropy] token_mean={:.6f} loss={:.6f} total={:.6f}",
        aux["entropy/token_mean"], aux["loss/entropy"], aux["loss/total"],
      )
    except Exception:
      pass
    return total_loss, aux

  aux["loss/total"] = policy_loss
  return policy_loss, aux

