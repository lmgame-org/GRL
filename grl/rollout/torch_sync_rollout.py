
"""
TorchSyncRollout

Dataflow: intialize rl dataset -> call rollout() - RolloutBatch in utils.py

Public API:


"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch


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
    if self.validation:
      self.agent_names = getattr(
          self.cfg.rollout, "validation", ["simpleSokobanAgent"]
      )
    else:
      self.agent_names = getattr(
          self.cfg.rollout, "training", ["simpleSokobanAgent"]
      )

    self.agent_config_list = []
    self.max_turns_list = []
    self.max_steps_list = []
    for agent_name in self.agent_names:
      self.agent_config_list.append(self.cfg[agent_name])
      self.max_turns_list.append(
          self.cfg[agent_name]["agent_config"].get("max_turns", 5)
      )
      self.max_steps_list.append(
          int(self.cfg[agent_name]["agent_config"].get("max_steps", 10))
      )
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
    base_configs = []
    seeds = []
    for i in range(len(self.agent_names)):
      base_configs.append(self.agent_config_list[i])
      seeds.append(base_seed + i * 100000)  # distinct base seed per type
    self.dataset = RLDataset(
        base_configs=base_configs,
        seeds=seeds,
        group_nums=self.agent_group_num_list,
        group_sizes=self.agent_group_size_list,
    )
    # Build all builders in one shot via get_batch(index=None)
    self.builders = self.dataset.get_batch(index=None, agent_name=self.agent_names[0])

    # Instantiate agents synchronously without environment reset
    agents: List[Any] = []
    for global_group_id, builder in enumerate(self.builders):
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

      for local_id, agent in enumerate(new_agents):
        agent.group_id = global_group_id
        agent.agent_id = len(agents)
        agents.append(agent)

    self.agents = agents
    self.done_mask = torch.zeros(self.total_agent_num, dtype=torch.bool)
    self.env_outs = None

  # ─────────────────── PROMPT/ENV HELPERS ───────────────────
  def get_batch_llm_prompts(self, env_outputs: List[Any]) -> List[str]:
    """
    Build per-agent chat prompts from current env outputs and agent message history.
    Mirrors sync/tunix implementations.
    """
    llm_prompts = [""] * len(env_outputs)
    for idx, env_out in enumerate(env_outputs):
      agent = self.agents[idx]
      messages = agent.get_llm_prompts(env_out)
      prompt_str = self._messages_to_prompt(messages, add_generation_prompt=True)
      llm_prompts[idx] = prompt_str
    return llm_prompts

  def get_batch_env_outputs(self, llm_responses_str: List[str]) -> List[Any]:
    """
    Update each agent's environment with decoded model responses.
    """
    updated_env_outs = [None] * len(llm_responses_str)
    for idx, reply in enumerate(llm_responses_str):
      agent = self.agents[idx]
      env_out = agent.get_env_outputs(reply)
      is_done = getattr(env_out, "truncated", False) or getattr(env_out, "terminated", False)
      updated_env_outs[idx] = env_out
      self.env_outs[idx] = env_out
      if self.done_mask is not None:
        self.done_mask[idx] = is_done
    return updated_env_outs

  # ─────────────────── GENERATION ───────────────────
  def generate_sequences(self, prompts: List[str]):
    """
    Use actor worker group to generate sequences from raw prompt strings.
    Mirrors sync_multi_turn_rollout.generate_sequences with DataProto padding when available.
    """
    # Prepare DataProto
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

    # Decode to strings for tool pipeline
    replies = self.tokenizer.batch_decode(
        lm_outputs.batch["responses"], skip_special_tokens=True
    )
    return replies

  # ─────────────────── MAIN ROLLOUT (TOOL-CALLING) ───────────────────
  def get_batch_tool_prompts(self, active_indices: List[int]) -> List[str]:
    """
    Build prompts for a batch of active agents during the tool-calling loop.
    """
    prompts = [""] * len(active_indices)
    for j, idx in enumerate(active_indices):
      agent = self.agents[idx]
      p = self._messages_to_prompt(agent.messages, add_generation_prompt=True)
      prompts[j] = p
    return prompts

  def process_batch_tool_responses(
      self,
      active_indices: List[int],
      replies: List[str],
      step_calls: List[int],
      max_steps_list: List[int],
  ) -> None:
    """
    For each active agent, execute one tool call (if present), or fallback to plain actions.
    Adds a 'Tool calls left: <k>' line to the most recent user feedback message when continuing.
    Updates env_outs and done_mask in place.
    """
    for j, idx in enumerate(active_indices):
      reply = replies[j] if j < len(replies) else ""
      step_calls[idx] += 1
      agent = self.agents[idx]

      # Debug: show raw reply snippet
      try:
        print(f"[DEBUG][process] agent={idx} step_calls={step_calls[idx]} has_reply={bool(reply)} reply_snippet={repr(str(reply)[:200])}")
      except Exception:
        pass

      if not reply:
        continue

      before_len = len(agent.get_messages())
      done, env_out_done = agent.execute_tool_call(reply)

      # Debug: outcome of execute_tool_call
      try:
        if done:
          r = getattr(env_out_done, "reward", None)
          term = getattr(env_out_done, "terminated", None)
          trunc = getattr(env_out_done, "truncated", None)
          print(f"[DEBUG][process] agent={idx} execute_tool_call done=True reward={r} terminated={term} truncated={trunc}")
      except Exception:
        pass

      if done:
        if env_out_done is not None:
          self.env_outs[idx] = env_out_done
        if self.done_mask is not None:
          self.done_mask[idx] = True
        continue

      # Fallback: execute plain actions if no function block
      if ("<function=" not in reply) and ("<answer>" in reply or "||" in reply):
        try:
          print(f"[DEBUG][process] agent={idx} fallback_to_env_step has_answer={("<answer>" in reply)} has_sep={("||" in reply)}")
        except Exception:
          pass
        env_out3 = agent.get_env_outputs(reply)
        self.env_outs[idx] = env_out3
        if self.done_mask is not None:
          self.done_mask[idx] = True
        try:
          print(f"[DEBUG][process] agent={idx} env_step reward={getattr(env_out3, 'reward', None)} terminated={getattr(env_out3, 'terminated', None)} truncated={getattr(env_out3, 'truncated', None)}")
        except Exception:
          pass
        continue

      # Add remaining budget hint to the last user feedback message (if any)
      remaining = max(0, max_steps_list[idx] - step_calls[idx])
      new_msgs = agent.get_messages()[before_len:]
      for m in reversed(new_msgs):
        if isinstance(m, dict) and m.get("role") == "user":
          try:
            m["content"] = f"{m.get('content', '')}\nTool calls left: {remaining}"
            print(f"[DEBUG][process] agent={idx} budget_hint_added remaining={remaining}")
          except Exception:
            pass
          break

  def rollout(self):
    """
    Tool-calling rollout loop modeled after tests/grl_agents_tests/group_sokoban_coding_agent_test.py.
    For each group/agent:
      - Reset with deterministic group seed
      - Repeatedly call the LLM (via actor_wg) and execute at most max_steps tool calls
      - If <function=finish|submit> is invoked, execute final actions and stop
      - Fallback: parse <answer>/plain actions and step the environment, then stop
    Finally builds a RolloutBatch for PPO.
    """
    # Build groups and agents for this episode based on seeds
    self._reset_batch_agents()

    # Batched tool-calling loop across all agents
    # Prepare per-agent step budgets
    max_steps_list = [int(a.agent_config.get("max_steps", 10)) for a in self.agents]
    step_calls = [0 for _ in self.agents]
    max_turns = max(getattr(a, "max_turns", 1) for a in self.agents) or 1

    # Multi-step, single-turn style (loop turns, with inner max-steps) but we typically run 1 turn
    for turn_idx in range(max_turns):
      for step_idx in range(max(max_steps_list) if max_steps_list else 0):
        active_indices = [
            i for i, a in enumerate(self.agents)
            if (self.done_mask is not None and not self.done_mask[i]) and step_calls[i] < max_steps_list[i]
        ]
        try:
          print(f"[DEBUG][rollout] turn={turn_idx} step={step_idx} active_indices={active_indices}")
        except Exception:
          pass
        if not active_indices:
          break

        prompts = self.get_batch_tool_prompts(active_indices)
        try:
          # Debug: show prompt snippets
          try:
            for j, idx in enumerate(active_indices):
              print(f"[DEBUG][prompt] agent={idx} prompt_snippet={repr(str(prompts[j])[:200])}")
          except Exception:
            pass
          replies = self.generate_sequences(prompts)
        except Exception:
          replies = [""] * len(active_indices)

        # Debug: show reply presence
        try:
          for j, idx in enumerate(active_indices):
            r = replies[j] if j < len(replies) else ""
            print(f"[DEBUG][reply] agent={idx} has_reply={bool(r)} contains_function={("<function=" in str(r))} contains_finish={("<function=finish>" in str(r))}")
          
        except Exception:
          pass

        self.process_batch_tool_responses(active_indices, replies, step_calls, max_steps_list)
        if self.done_mask is not None and bool(self.done_mask.all()):
          break

    # After all groups/agents finish, collect trajectories and build PPO batch
    final_rollout_states = self._collect_final_rollout_states()

    # Debug/trace printout of final rollout states (repr), tailored for SokobanCodingAgent
    try:
      print("\n=== Final Rollout States (repr) ===")
      for i, st in enumerate(final_rollout_states):
        try:
          print(f"[Agent {i}]", repr(st))
        except Exception:
          print(f"[Agent {i}] <unprintable state>")
    except Exception:
      pass

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

    for agent in self.agents:
      messages = agent.get_messages()
      assert all(msg["role"] == "assistant" for msg in messages[2::2])
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
    seeds = []
    for i in range(len(self.agent_names)):
      base_configs.append(self.agent_config_list[i])
      seeds.append(base_seed + i * 100000)
    self.dataset = RLDataset(
        base_configs=base_configs,
        seeds=seeds,
        group_nums=self.agent_group_num_list,
        group_sizes=self.agent_group_size_list,
    )
    self.builders = self.dataset.get_batch(index=None, agent_name=self.agent_names[0])

    # Instantiate agents synchronously
    agents: List[Any] = []
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
        env_out = agent.reset(seed=seeds[global_group_id])
        agents.append(agent)

    self.agents = agents
    self.done_mask = np.zeros(len(self.agents), dtype=bool)
    # Populate initial env_outs from fresh reset states by prompting a no-op action
    self.env_outs = [None] * len(self.agents)  # type: ignore
    for idx, agent in enumerate(self.agents):
      try:
        # Prefer the EnvOutput returned by reset; reconstruct minimal if needed
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