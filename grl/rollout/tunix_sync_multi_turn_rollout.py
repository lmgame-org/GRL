# sync_multi_turn_rollout.py
from typing import List, Dict, Any, Union
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from grl.agents import get_agent_cls, REGISTERED_AGENTS
from grl.rollout.utils import RolloutBatch
import jax.numpy as jnp
import jax
import numpy as np
from tunix.rl.rl_cluster import Mode


class SyncMultiTurnRollout:
  """
  Vectorised, synchronous, multi-turn rollout manager.
  Batch size = cfg.agent_length (fallback to cfg.env_length).

  Each agent manages its own environment, history, and recorder internally.
  This class orchestrates the batch processing and LLM calls.
  """

  # ─────────────────── INITIALIZATION ───────────────────
  def __init__(self, rl_cluster, cfg, tokenizer, processor, validation=False):
    """evaluate the transferability to Tetris, Blocksworld, and GSM8K.
    Initialize rollout manager. Agent class is resolved from config.

    Args:
        actor_rollout_wg: Actor rollout worker group
        cfg: Configuration object
        tokenizer: Tokenizer for text processing
        processor: Data processor
    """
    self.cfg = cfg
    self.tokenizer = tokenizer
    self.processor = processor
    self.rl_cluster = rl_cluster

    # Calculate total agents from agent_group_num * agent_group_size
    if validation:
      self.agent_group_num_list = getattr(
          cfg.rollout, "validation_agent_group_num", [64]
      )
      self.agent_group_size_list = getattr(
          cfg.rollout, "validation_agent_group_size", [1]
      )
    else:
      self.agent_group_num_list = getattr(cfg.rollout, "agent_group_num", [4])
      self.agent_group_size_list = getattr(cfg.rollout, "agent_group_size", [2])

    self.n_agents_list = [
        agent_group_num * agent_group_size
        for agent_group_num, agent_group_size in zip(
            self.agent_group_num_list, self.agent_group_size_list
        )
    ]
    self.total_group_num = sum(self.agent_group_num_list)
    self.total_agent_num = sum(self.n_agents_list)
    self.validation = validation

    # Threading configuration (lightweight acceleration on CPU)
    # 0 or missing -> auto (use min(32, os.cpu_count()))
    def _auto_threads(x: int | None) -> int:
      if not x or x <= 0:
        return max(1, min(32, (os.cpu_count() or 1)))
      return x

    rollout_cfg = getattr(self.cfg, "rollout", None)
    self.num_prompt_threads = _auto_threads(
        getattr(rollout_cfg, "num_prompt_threads", 0)
    )
    self.num_env_threads = _auto_threads(
        getattr(rollout_cfg, "num_env_threads", 0)
    )
    self.num_init_threads = _auto_threads(
        getattr(rollout_cfg, "num_init_threads", 0)
    )
    self.show_tqdm = bool(getattr(rollout_cfg, "show_tqdm", False))

    # Initialize agent configuration from config
    self._setup_agent_config()
    self._init_batch_agents()

    # Global turn counter
    self.step_cnt = 0

  def _setup_agent_config(self):
    """
    Setup agent configuration for single agent type.
    Agent class is resolved from config and agent config is extracted.
    """
    # Get agent name from rollout config train list (first item)
    if self.validation:
      self.agent_names = getattr(
          self.cfg.rollout, "validation", ["simpleSokobanAgent"]
      )
    else:
      self.agent_names = getattr(
          self.cfg.rollout, "training", ["simpleSokobanAgent"]
      )

    self.agent_cls_list = []
    self.agent_config_list = []
    self.max_turns_list = []
    for agent_name in self.agent_names:
      agent_type = self.cfg[agent_name]["agent_type"]

      # Resolve agent class from registry
      self.agent_cls_list.append(get_agent_cls(agent_type))

      self.agent_config_list.append(self.cfg[agent_name])
      self.max_turns_list.append(
          self.cfg[agent_name]["agent_config"].get("max_turns", 5)
      )

    # Get max_turns from agent config
    self.max_turns = max(self.max_turns_list)

  def _init_batch_agents(self):
    """
    Build self.agents: List[Agent] without resetting them.
    Each agent handles its own history & recorder.
    Agents are grouped based on agent_group_size for training purposes.
    Actual reset happens in rollout() via _reset_batch_agents().
    """
    # Create agents of the same type
    if len(self.agent_cls_list) == 0:
      raise ValueError("agent_cls_list is None but trying to create agents")

    # Verify the math
    for i, agent_num in enumerate(self.n_agents_list):
      if (
          agent_num
          != self.agent_group_num_list[i] * self.agent_group_size_list[i]
      ):
        raise ValueError(
            f"Total agents ({agent_num}) != agent_group_num ({self.agent_group_num_list[i]}) × agent_group_size ({self.agent_group_size_list[i]})"
        )

    self.agents = []
    done_groups = 0
    agent_id_counter = 0

    # loop through all agent types
    for i, agent_cls in enumerate(self.agent_cls_list):
      group_num = self.agent_group_num_list[i]
      group_size = self.agent_group_size_list[i]
      cfg = self.agent_config_list[i]
      name = self.agent_names[i]
      # loop through all groups of the same agent type
      for local_group_id in range(group_num):
        global_group_id = local_group_id + done_groups
        # loop through group_size to initialize agents
        for _ in range(group_size):
          agent = agent_cls(
              config=cfg,
              agent_id=agent_id_counter,
              group_id=global_group_id,
              tag=name,
          )
          agent_id_counter += 1
          self.agents.append(agent)
      # update the done_groups for the next agent type
      done_groups += self.agent_group_num_list[i]

    # Initialize tracking structures - actual env_outs will be set in rollout()
    self.done_mask = np.zeros(self.total_agent_num, dtype=bool)
    self.env_outs = None  # Will be initialized in _reset_batch_agents()

  # ─────────────────── BATCH LLM PROMPTS ───────────────────
  def get_batch_llm_prompts(self, env_outputs: list) -> list[str]:
    """
    Generate a batch of prompt strings from environment outputs.

    Each agent provides messages via agent.get_llm_prompts(env_out), which
    are rendered with the tokenizer's chat template. Optionally appends
    a think/answer control token per agent config.

    Args:
        env_outputs: List of per-agent environment outputs (opaque objects
            consumed by the agents' get_llm_prompts).

    Returns:
        List[str]: Prompt strings, one per agent (left-padding handled later).
    """
    llm_prompts = [""] * len(env_outputs)

    def build_prompt(idx_env_out):
      idx, env_out = idx_env_out
      if self.done_mask[idx]:
        return idx, ""
      agent = self.agents[idx]
      messages = agent.get_llm_prompts(env_out)
      try:
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
      except Exception:
        prompt_str = "System error in chat template"
      if agent.agent_config.get("use_think_answer_token", True):
        prompt_str += (
            "<think>"
            if agent.agent_config.get("enable_think", True)
            else "<answer>"
        )
      return idx, prompt_str

    with ThreadPoolExecutor(max_workers=self.num_prompt_threads) as ex:
      iterator = ex.map(build_prompt, enumerate(env_outputs))
      if self.show_tqdm:
        iterator = tqdm(
            iterator,
            total=len(env_outputs),
            desc="rollout/prompts",
            leave=False,
        )
      for idx, prompt in iterator:
        llm_prompts[idx] = prompt

    return llm_prompts

  # ─────────────────── RESPONSE DECODING HELPERS ───────────────────
  def tokens_to_text(self, token_seqs) -> list[str]:
    """
    Convert a batch of token id sequences to strings using the tokenizer.
    Accepts array-like inputs (e.g., JAX or numpy arrays) and returns List[str].
    """
    arr = np.array(token_seqs)
    return self.tokenizer.batch_decode(arr, skip_special_tokens=True)

  def decode_llm_responses(self, llm_responses) -> list[str]:
    """
    Decode rollout responses into a list of strings.
    Supports objects with `.text` (List[str]) or `.tokens` fields
    like rl_cluster.generate() RolloutOutput, or a raw List[str].
    """
    if isinstance(llm_responses, list) and all(
        isinstance(x, str) for x in llm_responses
    ):
      return llm_responses
    if hasattr(llm_responses, "text") and isinstance(llm_responses.text, list):
      return llm_responses.text
    if hasattr(llm_responses, "tokens"):
      return self.tokens_to_text(llm_responses.tokens)
    raise TypeError(
        "Unsupported llm_responses type: expected List[str] or RolloutOutput with .text or .tokens."
    )

  # ─────────────────── BATCH ENV OUTPUTS ───────────────────
  def get_batch_env_outputs(self, llm_responses_str: list[str]):
    """
    Process LLM outputs and update environment outputs for all agents.

    Args:
        llm_responses_str: List[str] decoded responses from the model, one per agent.

    Returns:
        List: Updated environment outputs from all agents
    """
    # Ensure env_outs is initialized (should be done by _reset_batch_agents)
    if self.env_outs is None:
      raise RuntimeError(
          "env_outs not initialized. Call rollout() or _reset_batch_agents() first."
      )

    # Update environment outputs for all agents in parallel
    updated_env_outs = [None] * len(llm_responses_str)

    def step_one(i_response):
      idx, reply = i_response
      if self.done_mask[idx]:
        return idx, self.env_outs[idx], True
      agent = self.agents[idx]
      env_out = agent.get_env_outputs(reply)
      is_done = env_out.truncated or env_out.terminated
      return idx, env_out, is_done

    with ThreadPoolExecutor(max_workers=self.num_env_threads) as ex:
      iterator = ex.map(step_one, enumerate(llm_responses_str))
      if self.show_tqdm:
        iterator = tqdm(
            iterator,
            total=len(llm_responses_str),
            desc="rollout/env",
            leave=False,
        )
      for idx, env_out, is_done in iterator:
        updated_env_outs[idx] = env_out
        self.env_outs[idx] = env_out
        self.done_mask[idx] = is_done

    return updated_env_outs

  # ─────────────────── LLM GENERATION ───────────────────
  def generate_sequences(self, llm_prompts: list[str]):
    """
    Generate sequences using the rollout cluster from a batch of LLM prompts.

    Args:
        llm_prompts: List[str] prompts (already templated), one per agent

    Returns:
        RolloutOutput: The rollout output containing text/tokens for completions
    """
    rollout_mode = Mode.EVAL if self.validation else Mode.TRAIN
    return self.rl_cluster.generate(prompts=llm_prompts, mode=rollout_mode)

  # ─────────────────── MAIN ROLLOUT LOOP ───────────────────
  def rollout(self):
    """
    Main rollout loop using batch LLM prompts approach.
    Iterate cfg.agent.max_turn turns, breaking early if all done.
    """
    self._reset_batch_agents()

    for _ in range(self.max_turns):
      if self.done_mask.all():
        break

      # Generate batch of LLM prompts from current env outputs
      batch_llm_prompts = self.get_batch_llm_prompts(self.env_outs)

      # Generate responses using batch dispatch
      llm_responses = self.generate_sequences(batch_llm_prompts)

      # Decode responses and update environment outputs
      llm_responses_str = self.decode_llm_responses(llm_responses)
      self.env_outs = self.get_batch_env_outputs(llm_responses_str)

      self.step_cnt += 1

    final_rollout_states = self._collect_final_rollout_states()
    return self.build_rollout_batch(final_rollout_states)

  # ─────────────────── PPO BATCH BUILDING ───────────────────

  def get_masks_and_scores(
      self,
      input_ids: jnp.ndarray,
      all_scores: List[List[float]] | None = None,
      use_turn_scores: bool = False,
  ):
    """
    Get loss/response masks and per-token score tensor.

    Implemented with NumPy to avoid JAX slicing/metadata issues on TPU; results
    are converted back to JAX arrays before returning.

    Args:
        input_ids: array-like with shape (bsz, seq_len)
        all_scores: List of score lists for each agent
        use_turn_scores: Whether to use turn-based scores

    Returns:
        Tuple of (loss_mask, score_tensor, response_mask) as jax.Arrays
    """
    # Materialize on host with NumPy first
    np_input_ids = np.asarray(input_ids)
    bsz, seq_len = np_input_ids.shape

    special_token = int(self.tokenizer.encode("<|im_start|>")[0])
    reward_token = int(self.tokenizer.encode("<|im_end|>")[0])

    # Turn indicators: increment at every <|im_start|>
    turn_starts = np.where(np_input_ids == special_token, 1, 0).astype(np.int32)
    turn_indicators = np.cumsum(turn_starts, axis=-1).astype(np.int32)

    # Masks in NumPy
    response_mask_np = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    loss_mask_np = turn_indicators > 1

    # Score tensor in NumPy
    score_tensor_np = np.zeros((bsz, seq_len), dtype=np.float32)
    if all_scores is not None and len(all_scores) > 0:
      if use_turn_scores:
        # Per-turn rewards placed at the <|im_end|> of each assistant turn
        # Transpose list-of-lists to iterate per turn
        # all_scores: List[ List[float] ] with shape [bsz][num_turns]
        try:
          per_turn_iter = list(zip(*all_scores))
        except Exception:
          per_turn_iter = []
        for turn_idx, scores_per_turn in enumerate(per_turn_iter):
          scores_arr = np.asarray(scores_per_turn, dtype=np.float32).reshape(
              bsz, 1
          )
          turn_indicator_val = int(
              turn_idx * 2 + 3
          )  # 0: pad, 1: system, 2+2n: user, 3+2n: assistant
          reward_position = (np_input_ids == reward_token) & (
              turn_indicators == turn_indicator_val
          )
          score_tensor_np = score_tensor_np + np.where(
              reward_position, scores_arr, 0.0
          )
      else:
        # Accumulate per-sequence reward and place at last token
        seq_scores = np.asarray(
            [float(sum(x)) for x in all_scores], dtype=np.float32
        )
        score_tensor_np[:, -1] = seq_scores

    # Align with loss calculation: target is next-token prediction
    loss_mask_np = loss_mask_np[:, :-1]
    score_tensor_np = score_tensor_np[:, 1:]

    # Convert results back to JAX
    loss_mask = jnp.asarray(loss_mask_np)
    score_tensor = jnp.asarray(score_tensor_np)
    response_mask = jnp.asarray(response_mask_np)

    return loss_mask, score_tensor, response_mask

  def _normalize_score_tensor(
      self, score_tensor: jnp.ndarray, env_outputs: List[Dict]
  ) -> jnp.ndarray:
    """
    Normalize the score tensor to be between 0 and 1.
    NOTE: only support score at the last token for now
    """
    assert (
        self.cfg.rollout.use_turn_scores == False
    ), "Reward normalization is not supported for use_turn_scores == True"

    rn_cfg = self.cfg.rollout.reward_normalization
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if grouping == "state":
      group_tags = [env_output["group_id"] for env_output in env_outputs]
    elif grouping == "inductive":
      group_tags = [env_output["tag"] for env_output in env_outputs]
    elif grouping == "batch":
      group_tags = [1] * len(env_outputs)
    else:
      raise ValueError(f"Invalid grouping: {grouping}")

    # Build group -> indices mapping
    group2index = {}
    for i, env_tag in enumerate(group_tags):
      if env_tag not in group2index:
        group2index[env_tag] = []
      group2index[env_tag].append(i)

    # Normalize group-wise using JAX ops
    acc_scores = score_tensor[:, -1]
    normalized_acc_scores = acc_scores

    for _, indices in group2index.items():
      idx_arr = jnp.array(indices, dtype=jnp.int32)
      group_vals = acc_scores[idx_arr]
      if method == "mean_std":
        mu = jnp.mean(group_vals)
        sigma = jnp.std(group_vals)
        cond = jnp.abs(sigma) > 1e-6
        normed = jnp.where(
            cond, (group_vals - mu) / (sigma + 1e-6), jnp.zeros_like(group_vals)
        )
      elif method == "mean":
        mu = jnp.mean(group_vals)
        normed = group_vals - mu
      elif method == "asym_clip":
        mu = jnp.mean(group_vals)
        sigma = jnp.std(group_vals)
        cond = jnp.abs(sigma) > 1e-6
        normed = jnp.where(
            cond, (group_vals - mu) / (sigma + 1e-6), jnp.zeros_like(group_vals)
        )
        normed = jnp.clip(normed, a_min=-1.0, a_max=3.0)
      elif method == "identity":
        normed = group_vals
      else:
        raise ValueError(f"Invalid normalization method: {method}")

      normalized_acc_scores = normalized_acc_scores.at[idx_arr].set(normed)

    # Apply penalty (JAX)
    penalty = jnp.array(
        [env_output["penalty"] for env_output in env_outputs], dtype=jnp.float32
    )
    normalized_acc_scores = normalized_acc_scores + penalty

    # Write back to the last position
    score_tensor = score_tensor.at[:, -1].set(normalized_acc_scores)

    return score_tensor

  def filter_rollout_batch(
      self, rollout_batch: RolloutBatch
  ) -> tuple[RolloutBatch, dict]:
    """
    Filter rollout batch using JAX, no DataProto or numpy.

    Expects `rollout_batch` to be a dict-like with at least:
      - 'rm_scores': jnp.ndarray of shape [N, T]
    Optionally contains other fields (jnp.ndarray with leading dim N) and
    an optional 'non_tensor_batch' dict whose values are jnp.ndarray or lists
    of length N. These will be filtered consistently.

    Returns (filtered_rollout_batch, metrics_dict)
    """
    rollout_filter_ratio = self.cfg.rollout.rollout_filter_ratio

    # Determine grouping geometry (assumes single agent type in training)
    num_groups, group_size = (
        self.agent_group_num_list[0],
        self.agent_group_size_list[0],
    )

    # Scores: [N, T] → per-sample scalar [N] → reshape to [G, S]
    rm_scores = jnp.array(rollout_batch.reward_scores)  # [N, T]
    per_sample = jnp.sum(rm_scores, axis=-1)  # [N]
    group_scores = jnp.reshape(per_sample, (num_groups, group_size))  # [G, S]

    selected_groups = int(rollout_filter_ratio * num_groups)

    # Group statistics
    in_group_std = jnp.std(group_scores, axis=-1)  # [G]
    in_group_max = jnp.max(group_scores, axis=-1)  # [G]
    in_group_mean = jnp.mean(group_scores, axis=-1)  # [G]

    if rollout_filter_ratio == 1:
      metrics = {
          "rollout/in_group_std": float(jnp.mean(in_group_std)),
          "rollout/in_group_max": float(jnp.mean(in_group_max)),
          "rollout/in_group_mean": float(jnp.mean(in_group_mean)),
          "rollout/chosen_in_group_std": float(jnp.mean(in_group_std)),
          "rollout/chosen_in_group_max": float(jnp.mean(in_group_max)),
          "rollout/chosen_in_group_mean": float(jnp.mean(in_group_mean)),
      }
      return rollout_batch, metrics

    # Select top groups by std (or reverse)
    if self.cfg.rollout.rollout_filter_type == "std_rev":
      vals = -in_group_std
    elif self.cfg.rollout.rollout_filter_type == "std":
      vals = in_group_std
    else:
      raise ValueError(
          f"Invalid rollout filter type: {self.cfg.rollout.rollout_filter_type}"
      )

    _, top_idx = jax.lax.top_k(vals, k=selected_groups)  # [K]

    # Build flat indices to keep: each selected group keeps all its members
    base = top_idx * group_size  # [K]
    offsets = jnp.arange(group_size)  # [S]
    keep_idx = (base[:, None] + offsets[None, :]).reshape(-1)  # [K*S]

    # Filter tensor fields (leading dim N)
    N = per_sample.shape[0]
    # Filter tensor-like fields inside the dataclass
    input_ids = jnp.array(rollout_batch.input_ids)[keep_idx]
    loss_mask = jnp.array(rollout_batch.loss_mask)[keep_idx]
    reward_scores = jnp.array(rollout_batch.reward_scores)[keep_idx]

    # Filter non-tensor fields
    # Filter non-tensor agent_raw_data
    keep_idx_list = list(map(int, keep_idx.tolist()))
    agent_raw = {}
    for key, value in rollout_batch.agent_raw_data.items():
      if isinstance(value, np.ndarray) and value.shape[:1] == (N,):
        agent_raw[key] = value[keep_idx_list]
      elif isinstance(value, list) and len(value) == int(N):
        agent_raw[key] = [value[i] for i in keep_idx_list]
      else:
        agent_raw[key] = value

    # Metrics including chosen groups
    metrics = {
        "rollout/in_group_std": float(jnp.mean(in_group_std)),
        "rollout/in_group_max": float(jnp.mean(in_group_max)),
        "rollout/in_group_mean": float(jnp.mean(in_group_mean)),
        "rollout/chosen_in_group_std": float(jnp.mean(in_group_std[top_idx])),
        "rollout/chosen_in_group_max": float(jnp.mean(in_group_max[top_idx])),
        "rollout/chosen_in_group_mean": float(jnp.mean(in_group_mean[top_idx])),
    }
    filtered = RolloutBatch(
        input_ids=np.array(input_ids),
        loss_mask=np.array(loss_mask),
        reward_scores=np.array(reward_scores),
        agent_raw_data=agent_raw,
        meta_info=rollout_batch.meta_info,
    )

    return filtered, metrics

  def _collect_final_rollout_states(self) -> List[Dict]:
    """
    Collect final rollout states from all agents.

    Returns:
        List[Dict]: List of rollout state dictionaries from all agents
    """
    env_outputs = []
    for idx in range(self.total_agent_num):
      agent = self.agents[idx]
      rollout_state = agent.get_final_rollout_states()
      env_outputs.append(rollout_state)
    return env_outputs

  def build_rollout_batch(self, rollout_states: List[Dict]) -> RolloutBatch:
    """
    Build PPO batch from the final batch rollout states using numpy/JAX, no DataProto.
    """

    llm_input_texts = []
    messages_list = []

    # Loop through all agents to collect their LLM prompts
    for idx, agent in enumerate(self.agents):
      # Get the current environment output for this agent
      env_out = self.env_outs[idx] if self.env_outs else None

      if env_out is None:
        # Handle case where env_outs is not initialized
        llm_input_texts.append("")
        continue

      # Get messages from agent's get_messages method
      messages = agent.get_messages()

      # NOTE: this assertion is important for loss mask computation
      assert all(msg["role"] == "assistant" for msg in messages[2::2])

      messages_list.append(messages)

      # Apply chat template to convert messages to text
      try:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
      except Exception as e:
        # Fallback in case of chat template error
        prompt_text = "System error in chat template"

      llm_input_texts.append(prompt_text)

    # Tokenize via tokenizer; many HF tokenizers return numpy if as numpy requested
    inputs = self.tokenizer(
        llm_input_texts,
        return_tensors=None,
        padding=True,
        padding_side="left",
        truncation=False,
    )
    # Harmonize to numpy arrays
    input_ids = (
        np.array(inputs["input_ids"])
        if isinstance(inputs, dict)
        else np.array(inputs.input_ids)
    )
    attention_mask = (
        np.array(inputs["attention_mask"])
        if isinstance(inputs, dict)
        else np.array(inputs.attention_mask)
    )
    # position_ids = attention_mask.cumsum(axis=-1)
    scores = [
        [i["reward"] for i in env_output["history"]]
        for env_output in rollout_states
    ]

    # Convert to jnp for mask/score computation
    loss_mask, score_tensor, response_mask = self.get_masks_and_scores(
        jnp.array(input_ids),
        scores,
        use_turn_scores=self.cfg.rollout.use_turn_scores,
    )
    normalized_score_tensor = self._normalize_score_tensor(
        score_tensor, rollout_states
    )
    response_length = float(np.mean(np.sum(np.array(response_mask), axis=-1)))

    # Compose numpy reward_scores
    reward_scores_np = np.array(normalized_score_tensor)

    agent_raw_data = {
        "env_ids": np.array(
            [env_output["env_id"] for env_output in rollout_states],
            dtype=object,
        ),
        "group_ids": np.array(
            [env_output["group_id"] for env_output in rollout_states],
            dtype=object,
        ),
        "messages_list": np.array(messages_list, dtype=object),
    }

    metrics = {}
    n_agents_map = dict(zip(self.agent_names, self.n_agents_list))
    for env_output in rollout_states:
      for key, value in env_output["metrics"].items():
        if key not in metrics:
          metrics[key] = []
        metrics[key].append(value)
    metrics = {
        key: np.sum(value) / n_agents_map[key.split("/")[0]]
        for key, value in metrics.items()
    }
    metrics["response_length"] = response_length

    return RolloutBatch(
        input_ids=np.array(input_ids),
        loss_mask=np.array(loss_mask).astype(np.int32),
        reward_scores=reward_scores_np,
        agent_raw_data=agent_raw_data,
        meta_info={"metrics": metrics},
    )

  # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────

  def _reset_batch_agents(self, seed=None):
    """
    Reset all agents and collect the batch of initial env outputs.
    This function resets the rollout manager for new epoch/rollout.

    Args:
        seed: Optional base seed for reproducibility. If None, generates random seed.
    """
    import random

    # Generate base seed following reference implementation pattern
    if seed is not None:
      base_seed = seed
    elif self.validation:
      base_seed = self.cfg.rollout.validation_seed
    else:  # Generate random seed for training, consistent seed for validation
      base_seed = random.randint(0, 1000000)

    # Generate group seeds: agents within same group share environment
    # Different groups get different environments
    group_seeds = [
        base_seed + group_id for group_id in range(self.total_group_num)
    ]

    initial_env_outs = []

    for idx in range(self.total_agent_num):
      agent = self.agents[idx]

      # All agents in the same group use the same seed (same environment)
      group_seed = group_seeds[agent.group_id]

      # Reset agent with group-specific seed
      initial_env_out = agent.reset(seed=group_seed)
      initial_env_outs.append(initial_env_out)

    # Update tracking structures with batch of reset outputs
    self.done_mask = np.zeros(self.total_agent_num, dtype=bool)
    self.env_outs = initial_env_outs
    self.step_cnt = 0

  def reset(self, seed=None):
    """
    Public reset method for external use (e.g., called by trainer between epochs).
    Delegates to _reset_batch_agents() for actual reset logic.

    Args:
        seed: Optional base seed for reproducibility
    """
    self._reset_batch_agents(seed=seed)

  def close(self):
    """
    Clean up agents and environments for tidy teardown.
    """
    for idx in range(self.total_agent_num):
      agent = self.agents[idx]
      if hasattr(agent, "close"):
        agent.close()
      # If agent has separate env reference
      if hasattr(agent, "env") and hasattr(agent.env, "close"):
        agent.env.close()
