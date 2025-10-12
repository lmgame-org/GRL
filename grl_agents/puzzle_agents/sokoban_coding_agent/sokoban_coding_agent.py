from typing import Any, Dict, List, Tuple, Optional, Union
import re
import os
from pathlib import Path

from grl_agents.base_agent import BaseAgent
from grl_agents.utils import SingleTurnTrajectory, EnvOutput
from grl.agents import register_agent
from .sokoban_env import SokobanEnv
from grl_agents.tools import build_default_tool_manager


@register_agent("sokobanCodingAgent")
class SokobanCodingAgent(BaseAgent):
  """
  Sokoban agent that manages environment interactions and conversation history.
  Compatible with GRL multi-turn rollout interface.
  """

  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    super().__init__(config, group_id, agent_id, seed, tag)
    # Resolve per-agent workspace path from config and ensure it exists
    base_workspace = self.agent_config.get("workspace_path")
    if base_workspace:
      per_agent = Path(str(base_workspace)).resolve() / f"{self.group_id}_{self.agent_id}"
      try:
        per_agent.mkdir(parents=True, exist_ok=True)
      except Exception:
        pass
      self.workspace_path = str(per_agent)
      # Route tools to this per-agent workspace directory
      os.environ["GRL_WORKSPACE_ROOT"] = self.workspace_path
    else:
      self.workspace_path = None  # type: ignore[assignment]
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
    # Initialize tool manager for function-call protocol
    try:
      self.tool_manager = build_default_tool_manager()
    except Exception:
      self.tool_manager = None

  # ─────────────────── TOOL-CALL PROTOCOL HELPERS ───────────────────
  def _parse_function_blocks(self, text: str) -> List[str]:
    if not isinstance(text, str):
      return []
    pattern = re.compile(r"<function\s*=\s*[^>]+>.*?</function>", re.DOTALL)
    return pattern.findall(text)

  def _parse_function_call(self, block: str) -> Tuple[str, Dict[str, str]]:
    try:
      fn_match = re.search(r"<function\s*=\s*([^>]+)>", block)
      function_name = fn_match.group(1).strip() if fn_match else ""
      params: Dict[str, str] = {}
      for key, val in re.findall(r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", block, flags=re.DOTALL):
        params[key.strip()] = val.strip()
      return function_name, params
    except Exception:
      return "", {}

  def _format_tool_observation(self, function_name: str, tool_out: Dict[str, Any]) -> str:
    output = str(tool_out.get("output", ""))
    exit_code = str(tool_out.get("exit_code", ""))
    if function_name in {"execute_bash", "bash"}:
      return f"Exit code: {exit_code}\nExecution output of [{function_name}]:\n{output}"
    return f"Execution output of [{function_name}]:\n{output}"

  # Simplified action parsing that accepts plain string, <answer>, or finish blocks
  def parse_llm_response(self, llm_response: str, enable_think: bool = False):
    text = str(llm_response) if not isinstance(llm_response, str) else llm_response
    # Prefer <function=finish> result if present
    blocks = self._parse_function_blocks(text)
    if blocks:
      for block in blocks:
        fn_name, params = self._parse_function_call(block)
        if fn_name.lower() in {"finish", "submit"}:
          action_line = params.get("result", "").split("\n", 1)[0].split("---", 1)[0].strip()
          action_content = action_line
          break
      else:
        action_content = text
    else:
      m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
      if m:
        action_content = m.group(1).strip()
      else:
        action_content = text.strip()
    # normalize separators
    action_content = action_content.replace("| |", "||").replace("|||", "||")
    parts = [p.strip() for p in action_content.split(self.action_separator) if p.strip()]
    if len(parts) > self.max_actions_per_turn:
      parts = parts[: self.max_actions_per_turn]
    processed = f"<answer>{' || '.join(parts)}</answer>"
    return processed, parts

  def get_env_outputs(self, llm_response: Union[str, List[str]]):
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

  def execute_tool_call(self, llm_response: str) -> Tuple[bool, Optional[EnvOutput]]:
    """Process exactly one tool call from the given LLM response.

    - If the call is <function=finish>, execute final actions and return (True, EnvOutput).
    - Otherwise, execute one tool (if available), append feedback, and return (False, None).
    - If no function block is present, return (False, None).
    """
    function_blocks = self._parse_function_blocks(llm_response)
    if not function_blocks:
      return False, None

    block = function_blocks[0]
    fn_name, params = self._parse_function_call(block)
    if not fn_name:
      return False, None

    # Log assistant call
    self.messages.append({"role": "assistant", "content": block})

    if fn_name.lower() in {"finish", "submit"}:
      result_text = params.get("result", "")
      action_line = result_text.split("\n", 1)[0].split("---", 1)[0].strip()
      env_out = self.get_env_outputs(action_line)
      return True, env_out

    tool_out: Dict[str, Any] = {"output": "", "exit_code": "0"}
    try:
      if getattr(self, "tool_manager", None) is not None:
        tool_out = self.tool_manager.execute(fn_name, params)
      else:
        tool_out = {"output": f"Tool manager unavailable for {fn_name}.", "exit_code": "-1"}
    except Exception as e:
      tool_out = {"output": f"Error executing tool {fn_name}: {e}", "exit_code": "-1"}

    feedback = self._format_tool_observation(fn_name, tool_out)
    self.messages.append({"role": "user", "content": feedback})
    return False, None
