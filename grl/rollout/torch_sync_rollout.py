
"""
TorchSyncRollout

Dataflow: intialize rl dataset -> call rollout() - RolloutBatch in utils.py

Public API:


"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto


from grl.rollout.utils import RolloutBatch
from grl_agents.rl_dataset import RLDataset
from grl_agents.agent_group_builder import AgentGroupBuilder


class TorchSyncRollout:
  """
  Universal, synchronous multi-turn rollout for Torch backends.

  - Initializes groups via RLDataset/AgentGroupBuilder
  - Supports tool-calling rollouts like tests/grl_agents_tests/group_sokoban_coding_agent_test.py
  - Uses actor_rollout_wg to generate sequences (mimicking sync_multi_turn_rollout)
  - Builds RolloutBatch (numpy arrays) for PPO training
  """

  def __init__(self, actor_rollout_wg, cfg, tokenizer, validation=False):
    self.cfg = cfg
    self.tokenizer = tokenizer
    self.actor_wg = actor_rollout_wg
    self.validation = validation

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

    # Simplified configuration (no threading/tqdm)

    # Initialize agent configuration
    self._setup_agent_config()

    # Dataset/builders are created in reset with concrete seeds
    self.dataset: RLDataset | None = None
    self.builders: List[AgentGroupBuilder] = []
    self.agents = []

    # Runtime states
    self.done_mask: np.ndarray | None = None
    self.env_outs: List[Dict[str, Any]] | None = None
    self.step_cnt = 0

  # ─────────────────── AGENT CONFIG ───────────────────
  def _setup_agent_config(self):
    # Resolve agent type names used for this rollout
    if self.validation:
      self.agent_names = getattr(
          self.cfg.rollout, "validation", ["simpleSokobanAgent"]
      )
    else:
      self.agent_names = getattr(
          self.cfg.rollout, "training", ["simpleSokobanAgent"]
      )

    # Normalize to a list for downstream logic (handle OmegaConf ListConfig)
    try:
      from omegaconf import ListConfig
      is_list_like = isinstance(self.agent_names, (list, tuple, ListConfig))
    except Exception:
      is_list_like = isinstance(self.agent_names, (list, tuple))
    if is_list_like:
      self.agent_names = list(self.agent_names)
    else:
      self.agent_names = [self.agent_names]

    # Build per-type configs and basic limits for convenience
    self.agent_config_list = []
    self.max_turns_list = []
    self.max_steps_list = []
    for agent_name in self.agent_names:
      conf = self.cfg[agent_name]
      self.agent_config_list.append(conf)
      ac = conf.get("agent_config", {})
      self.max_turns_list.append(ac.get("max_turns", 5))
      self.max_steps_list.append(int(ac.get("max_steps", 10)))
    self.max_turns = max(self.max_turns_list) if self.max_turns_list else 1
    self.max_steps = max(self.max_steps_list) if self.max_steps_list else 10

  def _init_batch_agents(self):
    """
    Build self.agents: List[Agent] without resetting them.
    Each agent handles its own history & recorder.
    Agents are grouped based on agent_group_size for training purposes.
    Actual reset happens in rollout() via _reset_batch_agents().
    """
    # Validate counts
    for i, agent_num in enumerate(self.n_agents_list):
      if agent_num != self.agent_group_num_list[i] * self.agent_group_size_list[i]:
        raise ValueError(
            f"Total agents ({agent_num}) != agent_group_num ({self.agent_group_num_list[i]}) × agent_group_size ({self.agent_group_size_list[i]})"
        )

    # Initialize dataset and builders for one synthetic episode (no reset here)
    import random
    base_seed = random.randint(0, 1_000_000)

    # Per agent type seeds/configs
    base_configs = [self.agent_config_list[i] for i in range(len(self.agent_names))]
    seeds = [base_seed + i * 100000 for i in range(len(self.agent_names))]

    # RLDataset prebuilds AgentGroupBuilders and assigns group_id contiguously
    self.dataset = RLDataset(
        base_configs=base_configs,
        seeds=seeds,
        group_nums=self.agent_group_num_list,
        group_sizes=self.agent_group_size_list,
        agent_names=self.agent_names,
    )
    # Flat list of builders for all groups
    self.builders = self.dataset.get_batch()

    # Instantiate agents synchronously without environment reset
    agents: List[Any] = []
    for builder in self.builders:
      try:
        import asyncio
        new_agents = asyncio.run(builder.make_agents())
      except Exception:
        try:
          import asyncio
          loop = asyncio.get_event_loop()
          new_agents = loop.run_until_complete(builder.make_agents())
        except Exception:
          import asyncio
          new_agents = asyncio.new_event_loop().run_until_complete(builder.make_agents())

      # AgentGroupBuilder already sets group_id and a contiguous agent_id offset
      agents.extend(new_agents)

    self.agents = agents
    self.done_mask = np.zeros(self.total_agent_num, dtype=bool)
    self.env_outs = None

  # ─────────────────── PROMPT/ENV HELPERS ───────────────────
  def get_batch_llm_prompts(self, env_outputs: List[Any]) -> List[str]:
    """
    Build per-agent chat prompts from current env outputs and agent message history.
    Mirrors sync/tunix implementations.
    """
    llm_prompts = [""] * len(env_outputs)
    for idx, env_out in enumerate(env_outputs):
      if self.done_mask is not None and self.done_mask[idx]:
        llm_prompts[idx] = ""
        continue
      agent = self.agents[idx]
      messages = agent.get_llm_prompts(env_out)
      prompt_str = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
      )
      
      llm_prompts[idx] = prompt_str
    return llm_prompts

  def get_batch_env_outputs(self, llm_responses_str: List[str]) -> List[Any]:
    """
    Update each agent's environment with decoded model responses.
    """
    if self.env_outs is None:
      raise RuntimeError("env_outs not initialized. Call rollout() or _reset_batch_agents() first.")
    updated_env_outs = [None] * len(llm_responses_str)
    for idx, reply in enumerate(llm_responses_str):
      if self.done_mask is not None and self.done_mask[idx]:
        updated_env_outs[idx] = self.env_outs[idx]
        continue
      agent = self.agents[idx]
      env_out = agent.get_env_outputs(reply)
      is_done = bool(getattr(env_out, "truncated", False) or getattr(env_out, "terminated", False))
      updated_env_outs[idx] = env_out
      self.env_outs[idx] = env_out
      if self.done_mask is not None:
        self.done_mask[idx] = is_done
    return updated_env_outs

  # ─────────────────── MAIN ROLLOUT (TOOL-CALLING) ───────────────────
  def get_batch_tool_llm_prompts(self, active_indices: List[int]) -> List[str]:
    """
    Build tool prompts for active agents from their current message histories.
    """
    prompts = [""] * len(active_indices)
    for j, idx in enumerate(active_indices):
      agent = self.agents[idx]
      messages = agent.get_tool_llm_prompts() if hasattr(agent, "get_tool_llm_prompts") else agent.get_messages()
      p = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      prompts[j] = p
    return prompts

  def execute_batch_tool_call_requests(
      self,
      active_indices: List[int],
      replies: List[str],
      step_calls: List[int],
      max_steps_list: List[int],
  ) -> None:
    """
    For each active agent, execute one tool call (if present), or fallback to plain actions.
    Updates env_outs and done_mask in place without exception handling for clearer debugging.
    """
    for j, idx in enumerate(active_indices):
      # Respect already finished agents
      if self.done_mask is not None and self.done_mask[idx]:
        continue

      reply = replies[j] if j < len(replies) else ""
      agent = self.agents[idx]

      if not reply:
        # Route empty reply through tool-call handler to preserve assistant→user parity
        try:
          agent.execute_tool_call(reply)
        except Exception:
          # Fallback minimal parity maintenance
          agent.messages.append({"role": "assistant", "content": str(reply)})
          agent.messages.append({"role": "user", "content": "Empty model reply. Provide a valid tool call or <answer>."})
        continue

      # Increment step count; enforce max tool steps only after a non-empty reply
      step_calls[idx] += 1
      if step_calls[idx] > max_steps_list[idx]:
        # Exhausted budget → mark done and skip
        if self.done_mask is not None:
          self.done_mask[idx] = True
        continue

      # Attempt tool execution
      done, env_out_done = agent.execute_tool_call(reply)
      if done:
        if env_out_done is not None:
          self.env_outs[idx] = env_out_done
        if self.done_mask is not None:
          self.done_mask[idx] = True
        continue



  # ─────────────────── SIMPLE CONTROL HELPERS ───────────────────
  def _compute_max_steps_per_agent(self) -> List[int]:
    """
    Compute per-agent max tool-call steps from agent configs with sane defaults.
    """
    return [int(getattr(a, "agent_config", {}).get("max_steps", 10)) for a in self.agents]

  def _mark_zero_budget_as_done(self, max_steps_list: List[int]) -> None:
    """
    Immediately mark agents with non-positive budgets as done for this turn.
    """
    if self.done_mask is None:
      return
    for i, mx in enumerate(max_steps_list):
      if mx <= 0:
        self.done_mask[i] = True

  def _select_active_indices(self, step_calls: List[int], max_steps_list: List[int]) -> List[int]:
    """
    Select indices of agents that are not done and still have tool-call budget left.
    """
    if self.done_mask is None:
      return []
    return [
        i for i in range(len(self.agents))
        if (not self.done_mask[i]) and step_calls[i] < max_steps_list[i]
    ]

  def _all_done(self) -> bool:
    return bool(self.done_mask is not None and self.done_mask.size > 0 and self.done_mask.all())

  def _run_tool_phase_turn(self, max_steps_list: List[int], step_calls: List[int]) -> None:
    """
    Run the inner tool-calling loop for a single outer turn:
    - Build prompts for active agents
    - Generate model replies
    - Execute at most one tool call per active agent per iteration
    Stops when no active agents remain or all agents are done.
    """
    inner_budget = (max(max_steps_list) + 1) if max_steps_list else 0
    for _ in range(inner_budget):
      active_indices = self._select_active_indices(step_calls, max_steps_list)
      if not active_indices:
        break
      tool_prompts = self.get_batch_tool_llm_prompts(active_indices)
      lm_outputs_tools = self.generate_sequences(tool_prompts)
      replies_tools = self.tokenizer.batch_decode(
          lm_outputs_tools.batch["responses"], skip_special_tokens=True
      )
      self.execute_batch_tool_call_requests(active_indices, replies_tools, step_calls, max_steps_list)
      if self._all_done():
        break


  # ─────────────────── GENERATION ───────────────────
  def generate_sequences(self, prompts: List[str]):
    """
    Generate sequences using the actor worker group (copied from SyncMultiTurnRollout).
    Returns a DataProto with generated sequences.
    """
    lm_inputs = self._build_dataproto_from_prompts(prompts)

    try:
      from verl.trainer.ppo.ray_trainer import RayWorkerGroup
      from verl.utils.dataset.rl_dataset import pad_dataproto_to_divisor, unpad_dataproto
    except ImportError:
      RayWorkerGroup = None
      pad_dataproto_to_divisor = None
      unpad_dataproto = None

    if (
        RayWorkerGroup is not None
        and isinstance(self.actor_wg, RayWorkerGroup)
        and pad_dataproto_to_divisor is not None
        and unpad_dataproto is not None
    ):
      padded_lm_inputs, pad_size = pad_dataproto_to_divisor(
          lm_inputs, self.actor_wg.world_size
      )
      padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
      lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
      lm_outputs.meta_info = lm_inputs.meta_info
      lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
    else:
      lm_outputs = self.actor_wg.generate_sequences(lm_inputs)

    return lm_outputs

  def _build_dataproto_from_prompts(self, prompts: List[str]) -> DataProto:
    """
    Convert a list of prompt strings into a DataProto compatible with
    the actor worker group's generate_sequences.
    """
    # Ensure left padding for autoregressive decoding
    try:
      original_padding_side = getattr(self.tokenizer, "padding_side", "right")
      self.tokenizer.padding_side = "left"
    except Exception:
      original_padding_side = None

    try:
      inputs = self.tokenizer(
          prompts,
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=int(getattr(self.cfg, "max_prompt_length", 4096)),
      )
    finally:
      # Restore tokenizer padding_side if available
      try:
        if original_padding_side is not None:
          self.tokenizer.padding_side = original_padding_side
      except Exception:
        pass

    input_ids: torch.Tensor = inputs.input_ids
    attention_mask: torch.Tensor = inputs.attention_mask

    # Position ids consistent with verl
    from verl.utils.model import compute_position_id_with_mask

    position_ids = compute_position_id_with_mask(attention_mask)

    dp = DataProto()
    dp.batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )

    dp.meta_info = {
        "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
        # Generation knobs can be extended/read by workers if needed
        "recompute_log_prob": False,
    }
    return dp


  def rollout(self):
    """
    Simplified rollout focused on tool-calling only:
    - Reset to initialize agents and initial messages/observations
    - For each turn, repeatedly build tool prompts → generate → execute tool calls
    - Build PPO batch from final trajectories
    """
    # Initialize groups/agents for this episode
    self._reset_batch_agents()

    max_steps_list = self._compute_max_steps_per_agent()
    step_calls = [0 for _ in self.agents]
    max_turns = max(getattr(a, "max_turns", 1) for a in self.agents) or 1

    for _ in range(max_turns):
      if self._all_done():
        break

      # Immediately mark agents with zero budget as done; then run tool phase
      self._mark_zero_budget_as_done(max_steps_list)
      self._run_tool_phase_turn(max_steps_list, step_calls)

    # Collect final trajectories using dataset helper when available
    if self.dataset is not None:
      import asyncio
      per_group_agent_rollouts = asyncio.run(self.dataset.collect_group_trajectories())
      final_rollout_states = [state for group in per_group_agent_rollouts for state in group]
    else:
      final_rollout_states = self._collect_final_rollout_states()

    return self.build_rollout_batch(final_rollout_states)

  # ─────────────────── MASKS AND SCORES (Torch) ───────────────────
  def get_masks_and_scores(
      self,
      input_ids: torch.Tensor,
      all_scores: List[List[float]] | None = None,
      use_turn_scores: bool = False,
  ):
    special_token = self.tokenizer.encode("<|im_start|>")[0]
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    loss_mask = turn_indicators > 1

    reward_token = self.tokenizer.encode("<|im_end|>")[0]
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if all_scores is not None:
      if use_turn_scores:
        for idx, scores in enumerate(list(zip(*all_scores))):
          scores = torch.tensor(scores, dtype=torch.float32)
          turn_indicator = idx * 2 + 3
          reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
          score_tensor[reward_position] = scores
      else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    loss_mask = loss_mask[:, :-1]
    score_tensor = score_tensor[:, 1:]
    return loss_mask, score_tensor, response_mask

  def _normalize_score_tensor(
      self, score_tensor: torch.Tensor, env_outputs: List[Dict]
  ) -> torch.Tensor:
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

    if method == "mean_std":
      norm_func = (
          lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
          if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
          else torch.zeros_like(x)
      )
    elif method == "mean":
      norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
    elif method == "asym_clip":
      norm_func = lambda x: (
          (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
          if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
          else torch.zeros_like(x)
      ).clamp(min=-1, max=3)
    elif method == "identity":
      norm_func = lambda x: x
    else:
      raise ValueError(f"Invalid normalization method: {method}")

    group2index: Dict[Any, torch.Tensor] = {}
    for i, env_tag in enumerate(group_tags):
      if env_tag not in group2index:
        group2index[env_tag] = []
      group2index[env_tag].append(i)
    group2index = {k: torch.tensor(v) for k, v in group2index.items()}

    acc_scores = score_tensor[:, -1]
    normalized_acc_scores = acc_scores.clone()
    for _, index in group2index.items():
      normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

    penalty = torch.tensor(
        [env_output["penalty"] for env_output in env_outputs], dtype=torch.float32
    )
    normalized_acc_scores = normalized_acc_scores + penalty
    score_tensor[:, -1] = normalized_acc_scores
    return score_tensor

  # ─────────────────── BUILD ROLLOUT BATCH ───────────────────
  def _collect_final_rollout_states(self) -> List[Dict]:
    env_outputs = []
    for agent in self.agents:
      rollout_state = agent.get_final_rollout_states()
      env_outputs.append(rollout_state)
    return env_outputs

  def build_rollout_batch(self, rollout_states: List[Dict]) -> RolloutBatch:
    # Construct chat transcripts for each agent
    llm_input_texts: List[str] = []
    messages_list: List[List[Dict[str, Any]]] = []

    for agent_idx, agent in enumerate(self.agents):
      messages = agent.get_messages()
      bad_indices = [i for i, msg in enumerate(messages[2::2], start=2) if msg.get("role") != "assistant"]
      if bad_indices:
        # Debug output to visualize what happened before raising
        try:
          print(f"[DEBUG] build_rollout_batch: Agent {agent_idx} has non-assistant roles at positions {bad_indices}")
          roles_seq = [m.get("role") for m in messages]
          print(f"[DEBUG] build_rollout_batch: roles sequence = {roles_seq}")
          for k, m in enumerate(messages):
            content_preview = str(m.get("content"))
            content_preview = content_preview if content_preview is not None else ""
            if len(content_preview) > 200:
              content_preview = content_preview[:200] + "..."
            print(f"[DEBUG] build_rollout_batch: msg[{k}] role={m.get('role')} | content={content_preview}")
        except Exception:
          pass
        raise AssertionError(
            f"Expected 'assistant' at even turns starting from index 2. Got {[messages[i].get('role') for i in bad_indices]} at positions {bad_indices} for agent {agent_idx}."
        )
      messages_list.append(messages)
      try:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
      except Exception:
        prompt_text = "System error in chat template"
      llm_input_texts.append(prompt_text)

    # Tokenize (HF tokenizer) and convert to torch then numpy
    inputs = self.tokenizer(
        llm_input_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=False,
    )
    input_ids: torch.Tensor = inputs.input_ids
    attention_mask: torch.Tensor = inputs.attention_mask

    scores = [
        [i["reward"] for i in env_output["history"]]
        for env_output in rollout_states
    ]

    loss_mask, score_tensor, response_mask = self.get_masks_and_scores(
        input_ids, scores, use_turn_scores=self.cfg.rollout.use_turn_scores
    )
    normalized_score_tensor = self._normalize_score_tensor(
        score_tensor, rollout_states
    )

    # Metrics
    metrics: Dict[str, float] = {}
    n_agents_map = dict(zip(self.agent_names, self.n_agents_list))
    for env_output in rollout_states:
      for key, value in env_output["metrics"].items():
        metrics.setdefault(key, []).append(value)
    metrics = {
        key: float(np.sum(value) / n_agents_map[key.split("/")[0]])
        for key, value in metrics.items()
    }
    response_length = response_mask.sum(dim=-1).float().mean().item()
    metrics["response_length"] = response_length

    # Compose RolloutBatch
    return RolloutBatch(
        input_ids=input_ids.cpu().numpy(),
        loss_mask=loss_mask.cpu().numpy().astype(np.int32),
        reward_scores=normalized_score_tensor.cpu().numpy(),
        agent_raw_data={
            "agent_ids": np.array(
                [env_output["agent_id"] for env_output in rollout_states],
                dtype=object,
            ),
            "group_ids": np.array(
                [env_output["group_id"] for env_output in rollout_states],
                dtype=object,
            ),
            "messages_list": np.array(messages_list, dtype=object),
        },
        meta_info={"metrics": metrics},
    )

  def build_ppo_batch(self, rollout_states: List[Dict]) -> DataProto:
    """
    Build a DataProto for PPO training from final rollout states.
    Mirrors SyncMultiTurnRollout.build_ppo_batch for compatibility.
    """
    llm_input_texts: List[str] = []
    messages_list: List[List[Dict[str, Any]]] = []

    for agent_idx, agent in enumerate(self.agents):
      messages = agent.get_messages()
      bad_indices = [i for i, msg in enumerate(messages[2::2], start=2) if msg.get("role") != "assistant"]
      if bad_indices:
        # Debug output to visualize what happened before raising
        try:
          print(f"[DEBUG] build_ppo_batch: Agent {agent_idx} has non-assistant roles at positions {bad_indices}")
          roles_seq = [m.get("role") for m in messages]
          print(f"[DEBUG] build_ppo_batch: roles sequence = {roles_seq}")
          for k, m in enumerate(messages):
            content_preview = str(m.get("content"))
            content_preview = content_preview if content_preview is not None else ""
            if len(content_preview) > 200:
              content_preview = content_preview[:200] + "..."
            print(f"[DEBUG] build_ppo_batch: msg[{k}] role={m.get('role')} | content={content_preview}")
        except Exception:
          pass
        raise AssertionError(
            f"Expected 'assistant' at even turns starting from index 2. Got {[messages[i].get('role') for i in bad_indices]} at positions {bad_indices} for agent {agent_idx}."
        )
      messages_list.append(messages)
      try:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
      except Exception:
        prompt_text = "System error in chat template"
      llm_input_texts.append(prompt_text)

    # Tokenize batched transcripts
    try:
      original_padding_side = getattr(self.tokenizer, "padding_side", "right")
      self.tokenizer.padding_side = "left"
    except Exception:
      original_padding_side = None
    try:
      inputs = self.tokenizer(
          llm_input_texts,
          return_tensors="pt",
          padding=True,
          truncation=False,
      )
    finally:
      try:
        if original_padding_side is not None:
          self.tokenizer.padding_side = original_padding_side
      except Exception:
        pass

    input_ids: torch.Tensor = inputs.input_ids
    attention_mask: torch.Tensor = inputs.attention_mask
    position_ids: torch.Tensor = attention_mask.cumsum(dim=-1)

    scores = [
        [i["reward"] for i in env_output["history"]]
        for env_output in rollout_states
    ]

    loss_mask, score_tensor, response_mask = self.get_masks_and_scores(
        input_ids, scores, use_turn_scores=self.cfg.rollout.use_turn_scores
    )
    normalized_score_tensor = self._normalize_score_tensor(
        score_tensor, rollout_states
    )
    response_length = response_mask.sum(dim=-1).float().mean().item()

    llm_inputs = DataProto()
    llm_inputs.batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],
            "loss_mask": loss_mask,
            "rm_scores": normalized_score_tensor,
        },
        batch_size=input_ids.shape[0],
    )

    llm_inputs.non_tensor_batch = {
        "agent_ids": np.array(
            [env_output["agent_id"] for env_output in rollout_states],
            dtype=object,
        ),
        "group_ids": np.array(
            [env_output["group_id"] for env_output in rollout_states],
            dtype=object,
        ),
        "messages_list": np.array(messages_list, dtype=object),
    }

    metrics: Dict[str, float] = {}
    n_agents_map = dict(zip(self.agent_names, self.n_agents_list))
    for env_output in rollout_states:
      for key, value in env_output["metrics"].items():
        metrics.setdefault(key, []).append(value)
    metrics = {
        key: float(np.sum(value) / n_agents_map[key.split("/")[0]])
        for key, value in metrics.items()
    }
    metrics["response_length"] = response_length
    llm_inputs.meta_info = {"metrics": metrics}

    return llm_inputs

  # ─────────────────── RESET / CLOSE ───────────────────
  def _reset_batch_agents(self, seed=None):
    import random

    # Base seed
    if seed is not None:
      base_seed = int(seed)
    elif self.validation:
      base_seed = int(self.cfg.rollout.validation_seed)
    else:
      base_seed = random.randint(0, 1_000_000)

    # Build per-type seeds and builders using dataset one-shot API
    base_configs = []
    type_base_seeds = []
    for i in range(len(self.agent_names)):
      base_configs.append(self.agent_config_list[i])
      type_base_seeds.append(base_seed + i * 100000)
    self.dataset = RLDataset(
        base_configs=base_configs,
        seeds=type_base_seeds,
        group_nums=self.agent_group_num_list,
        group_sizes=self.agent_group_size_list,
    )
    self.builders = self.dataset.get_batch(index=None, agent_name=self.agent_names[0])

    # Instantiate agents synchronously
    agents: List[Any] = []
    # Precompute per-builder group seeds in the same order as builders were created
    group_seeds: List[int] = []
    for i, num_groups in enumerate(self.agent_group_num_list):
      base_s = type_base_seeds[i]
      for j in range(int(num_groups)):
        group_seeds.append(int(base_s + j))

    for global_group_id, builder in enumerate(self.builders):
      try:
        import asyncio
        new_agents = asyncio.run(builder.make_agents())
      except Exception:
        try:
          # Fallback for existing loop context
          import asyncio
          loop = asyncio.get_event_loop()
          new_agents = loop.run_until_complete(builder.make_agents())
        except Exception:
          # Last resort: run in separate thread
          import asyncio
          new_agents = asyncio.new_event_loop().run_until_complete(builder.make_agents())

      # Fix group_id to global id and agent_id continuity
      for local_id, agent in enumerate(new_agents):
        agent.group_id = global_group_id
        agent.agent_id = len(agents)
        # Reset with group seed, capture EnvOutput
        env_out = agent.reset(seed=group_seeds[global_group_id])
        agents.append(agent)

    self.agents = agents
    self.done_mask = np.zeros(len(self.agents), dtype=bool)
    # Populate initial env_outs using each agent's reset observation
    self.env_outs = [None] * len(self.agents)  # type: ignore
    for idx, agent in enumerate(self.agents):
      # Agent.reset already produced the initial observation/state in messages; render minimal EnvOutput
      try:
        from grl_agents.utils import EnvOutput
        obs_txt = agent.env.render()
        self.env_outs[idx] = EnvOutput(truncated=False, terminated=False, state=obs_txt, reward=0.0, info={})
      except Exception:
        from grl_agents.utils import EnvOutput
        self.env_outs[idx] = EnvOutput()
    self.step_cnt = 0

  def reset(self, seed=None):
    self._reset_batch_agents(seed=seed)

  def close(self):
    for agent in self.agents:
      if hasattr(agent, "close"):
        agent.close()