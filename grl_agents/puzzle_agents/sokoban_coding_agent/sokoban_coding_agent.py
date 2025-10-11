from typing import Any, Dict, List, Tuple, Optional
import re

from grl_agents.base_agent import BaseAgent, SingleTurnTrajectory, EnvOutput
from .sokoban_env import SokobanEnv


class SokobanCodingAgent(BaseAgent):
  """
  Sokoban agent that manages environment interactions and conversation history.
  Compatible with GRL multi-turn rollout interface.
  """

  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    super().__init__(config, group_id, agent_id, seed, tag)
    self.prompt = self._build_enhanced_prompt(self.prompt)
    self.initialize_env()

  def _build_enhanced_prompt(self, base_prompt: str) -> str:
    enhanced_prompt = base_prompt
    if self.env_config.get("grid_vocab"):
      symbols = [f"{k}: {v}" for k, v in self.env_config["grid_vocab"].items()]
      enhanced_prompt += (
          "\nThe meaning of each symbol in the state is:\n "
          + ", ".join(symbols)
      )
    if self.env_config.get("action_lookup"):
      actions = list(self.env_config["action_lookup"].values())
      enhanced_prompt += "\nYour available actions are:\n" + ", ".join(actions)
    enhanced_prompt += f"\nYou can make up to {self.max_actions_all_turns} actions, and each action is separated by '{self.action_separator}'."
    return enhanced_prompt

  def initialize_env(self) -> None:
    self.env = SokobanEnv(self.env_config)

  # Simplified action parsing that accepts plain string or <answer> blocks
  def parse_llm_response(self, llm_response: str, enable_think: bool = False):
    text = (
        str(llm_response) if not isinstance(llm_response, str) else llm_response
    )
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
      action_content = m.group(1).strip()
    else:
      action_content = text.strip()
    # normalize separators
    normalized = action_content.replace("| |", "||").replace("|||", "||")
    parts = [
        p.strip() for p in normalized.split(self.action_separator) if p.strip()
    ]
    if len(parts) > self.max_actions_per_turn:
      parts = parts[: self.max_actions_per_turn]
    processed = f"<answer>{' || '.join(parts)}</answer>"
    return processed, parts

  def get_env_outputs(self, llm_response):
    llm_raw_response = llm_response
    self.raw_response_list.append(llm_raw_response)
    self.cur_turn += 1

    processed_llm_response, actions = self.parse_llm_response(
        str(llm_raw_response), enable_think=self.enable_think
    )
    self.messages.append(
        {"role": "assistant", "content": processed_llm_response}
    )

    obs = self.env.render()
    total_reward = 0.0
    done = False
    executed_actions: List[int] = []
    info: Dict[str, Any] = {}

    action_lookup_reverse = {
        v: k for k, v in self.env_config["action_lookup"].items()
    }
    action_lookup_reverse_lower = {
        v.lower(): k for k, v in self.env_config["action_lookup"].items()
    }

    valid_actions: List[int] = []
    invalid_actions: List[str] = []

    for action_str in actions:
      try:
        action_str_clean = action_str.strip()
        if action_str_clean in action_lookup_reverse:
          action = action_lookup_reverse[action_str_clean]
          if action in self.env_config["action_lookup"]:
            valid_actions.append(action)
          else:
            invalid_actions.append(action_str)
        elif action_str_clean.lower() in action_lookup_reverse_lower:
          action = action_lookup_reverse_lower[action_str_clean.lower()]
          if action in self.env_config["action_lookup"]:
            valid_actions.append(action)
          else:
            invalid_actions.append(action_str)
        else:
          action = int(action_str_clean)
          if action in self.env_config["action_lookup"]:
            valid_actions.append(action)
          else:
            invalid_actions.append(action_str)
      except (ValueError, KeyError, TypeError):
        invalid_actions.append(action_str)
        continue

    if (
        len(actions) == 0
        or invalid_actions
        or len(valid_actions) != len(actions)
    ):
      self.penalty += self.format_penalty

    for a in valid_actions:
      try:
        obs, reward, done, step_info = self.env.step(a)
        total_reward += reward
        executed_actions.append(a)
        info.update(step_info)
        if done:
          break
      except Exception:
        continue

    self.total_actions_consumed += len(executed_actions)
    actions_left = max(
        0, self.max_actions_all_turns - self.total_actions_consumed
    )
    if (
        self.cur_turn >= self.max_turns
        or self.total_actions_consumed >= self.max_actions_all_turns
    ):
      done = True

    self.trajectory_history.add(
        SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions=executed_actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=str(llm_raw_response),
        )
    )

    return EnvOutput(
        truncated=done,
        terminated=done,
        state=obs,
        reward=total_reward,
        info=info,
    )
