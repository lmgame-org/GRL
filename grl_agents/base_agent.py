from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import random


@dataclass
class EnvOutput:
  truncated: bool
  terminated: bool
  state: Any
  reward: float
  info: Dict[str, Any]


@dataclass
class SingleTurnTrajectory:
  state: Any
  actions_left: int
  actions: List[int]
  reward: float
  info: Dict[str, Any]
  llm_response: str
  llm_raw_response: str


class MultiTurnTrajectory:

  def __init__(self, max_length: int = 5):
    from collections import deque

    self._deque = deque(maxlen=max_length)
    self.max_length = max_length

  def add(self, traj: SingleTurnTrajectory) -> None:
    self._deque.append(traj)

  def get(self):
    return self._deque

  def clear(self) -> None:
    self._deque.clear()


class BaseAgent:
  """Abstract base class for GRL agents with multi-turn rollout helpers."""

  def __init__(
      self, config: Dict[str, Any], group_id=0, agent_id=0, seed=None, tag=None
  ):
    self.group_id = group_id
    self.agent_id = agent_id
    self.tag = tag
    self.cur_turn = 0
    self.seed = random.randint(0, 2**32 - 1) if seed is None else seed

    self.agent_config = config.get("agent_config", {})
    self.env_config = config.get("env_config", {})

    # Hyperparameters
    self.max_turns = self.agent_config.get("max_turns", 1)
    self.max_actions_all_turns = self.agent_config.get(
        "max_actions_all_turns", 1
    )
    self.max_actions_per_turn = self.agent_config.get("max_actions_per_turn", 1)
    self.max_tokens = self.agent_config.get("max_tokens", 100)
    self.format_penalty = self.agent_config.get("format_penalty", -0.1)
    self.enable_think = self.agent_config.get("enable_think", True)
    self.system_prompt = self.agent_config.get(
        "system_prompt", "You are a helpful AI assistant."
    )
    self.prompt = self.agent_config.get(
        "prompt", "Please respond appropriately."
    )
    self.action_separator = self.agent_config.get("action_separator", "||")

    if self.enable_think:
      self.turn_prompt_template = (
          """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. """
          """Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. """
          """Strictly follow this format. Max response length: {max_tokens} tokens.\n"""
      )
    else:
      self.turn_prompt_template = (
          """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. """
          """Always output: <answer> [your answer] </answer> with no extra text. """
          """Strictly follow this format. Max response length: {max_tokens} tokens.\n"""
      )

    self.trajectory_history = MultiTurnTrajectory(max_length=self.max_turns)
    self.raw_response_list: List[str] = []
    self.messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.prompt},
    ]
    self.total_actions_consumed = 0
    self.penalty = 0.0

  def get_llm_prompts(self, env_out: EnvOutput):
    if not getattr(self, "messages", None):
      self.messages = [
          {"role": "system", "content": self.system_prompt},
          {"role": "user", "content": self.prompt},
      ]

    actions_remaining = max(
        0, self.max_actions_all_turns - self.total_actions_consumed
    )
    turn_content = self.turn_prompt_template.format(
        turn_number=self.cur_turn + 1,
        state=env_out.state,
        actions_remaining=actions_remaining,
        max_tokens=self.max_tokens,
    )
    turn_msg = {"role": "user", "content": turn_content}

    if (
        self.cur_turn == 0
        and len(self.messages) == 2
        and self.messages[1]["role"] == "user"
    ):
      self.messages[1]["content"] = (
          self.messages[1]["content"] + "\n" + turn_content
      )
    else:
      reward_msg = f"Reward: \n{env_out.reward}\n"
      turn_msg["content"] = reward_msg + " " + turn_msg["content"]
      self.messages.append(turn_msg)
    return self.messages

  def parse_llm_response(
      self, llm_response: str, enable_think: bool = True
  ) -> Tuple[str, List[str]]:
    import re

    if self.agent_config.get("use_think_answer_token", True):
      if enable_think:
        llm_response = "<think>" + llm_response
      else:
        llm_response = "<answer>" + llm_response

    pattern = (
        r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
        if enable_think
        else r"<answer>(.*?)</answer>"
    )
    match = re.search(pattern, llm_response, re.DOTALL)
    if not match:
      return llm_response, []
    if enable_think:
      think_content, action_content = match.group(1), match.group(2)
    else:
      think_content, action_content = "", match.group(1)
    special_tokens = [
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    for tok in special_tokens:
      action_content = action_content.replace(tok, "").strip()
      think_content = think_content.replace(tok, "").strip()
    actions = [
        a.strip()
        for a in action_content.split(self.action_separator)
        if a.strip()
    ]
    if len(actions) > self.max_actions_per_turn:
      actions = actions[: self.max_actions_per_turn]
      action_content = self.action_separator.join(actions)
    processed = (
        f"<think>{think_content}</think><answer>{action_content}</answer>"
        if enable_think
        else f"<answer>{action_content}</answer>"
    )
    return processed, actions

  def get_final_rollout_states(self) -> Dict[str, Any]:
    history = []
    dq = self.trajectory_history.get()
    for traj in dq:
      history.append(
          {
              "state": traj.state,
              "actions_left": traj.actions_left,
              "actions": traj.actions,
              "reward": traj.reward,
              "info": traj.info,
              "llm_response": traj.llm_response,
              "llm_raw_response": traj.llm_raw_response,
          }
      )

    metrics: Dict[str, Any] = {}
    success_values = [traj.info.get("success", False) for traj in dq]
    metrics[f"{self.tag or 'baseAgent'}/success"] = float(any(success_values))
    total_actions = sum(len(traj.actions) for traj in dq)
    metrics[f"{self.tag or 'baseAgent'}/num_actions"] = total_actions
    action_is_effective_values = [
        traj.info.get("action_is_effective", False) for traj in dq
    ]
    metrics[f"{self.tag or 'baseAgent'}/action_is_effective"] = (
        (sum(action_is_effective_values) / len(action_is_effective_values))
        if action_is_effective_values
        else 0.0
    )
    action_is_valid_values = [
        traj.info.get("action_is_valid", False) for traj in dq
    ]
    metrics[f"{self.tag or 'baseAgent'}/action_is_valid"] = (
        (sum(action_is_valid_values) / len(action_is_valid_values))
        if action_is_valid_values
        else 1.0
    )

    if dq:
      last_traj = dq[-1]
      if "metrics" in last_traj.info:
        for k, v in last_traj.info["metrics"].items():
          metrics[k] = v

    return {
        "env_id": self.agent_id,
        "history": history,
        "group_id": self.group_id,
        "tag": self.tag or "baseAgent",
        "metrics": metrics,
        "penalty": self.penalty,
    }

  def reset(self, seed: int | None = None) -> EnvOutput:
    reset_seed = random.randint(0, 1_000_000) if seed is None else seed
    obs = self.env.reset(seed=reset_seed)
    if not obs:
      obs = self.env.render()
    self.cur_turn = 0
    self.trajectory_history.clear()
    self.raw_response_list = []
    self.total_actions_consumed = 0
    self.penalty = 0.0
    self.messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.prompt},
    ]
    return EnvOutput(
        truncated=False, terminated=False, state=obs, reward=0.0, info={}
    )

  def close(self) -> None:
    if hasattr(self, "env") and hasattr(self.env, "close"):
      self.env.close()

  def get_messages(self):
    return self.messages

  # ─────────────────── Interface for subclasses ───────────────────
  def initialize_env(self) -> None:
    raise NotImplementedError

  def get_env_outputs(self, llm_response: str) -> EnvOutput:
    raise NotImplementedError
