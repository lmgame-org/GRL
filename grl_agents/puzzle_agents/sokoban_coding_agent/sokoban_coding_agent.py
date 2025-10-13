from typing import Any, Dict, List, Tuple, Optional, Union
import re
import os
from pathlib import Path

from grl_agents.base_agent import BaseAgent
from grl_agents.utils import SingleTurnTrajectory, EnvOutput, SingleTurnToolCallTrajectory
from grl.agents import register_agent
from .sokoban_env import SokobanEnv
from grl_agents.tools import build_default_tool_manager


"""
SokobanCodingAgent
"""


@register_agent("sokobanCodingAgent")
class SokobanCodingAgent(BaseAgent):
  """
  Sokoban agent that manages environment interactions and conversation history.
  Compatible with GRL multi-turn rollout interface.
  """

  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    super().__init__(config, group_id, agent_id, seed, tag)
    # Whether to enable tool-use protocol and tracking
    self.tool_use: bool = bool(self.agent_config.get("tool_use", False))
    # Track per-turn tool calls from config (max_steps budget)
    self.max_tool_steps: int = int(self.agent_config.get("max_steps", 10))
    self.tool_calls_this_turn: int = 0
    # Resolve per-agent workspace path from config and ensure it exists
    base_workspace = self.agent_config.get("workspace_path")
    if base_workspace:
      base_root = Path(str(base_workspace)).resolve()
      per_agent = base_root / f"group_{self.group_id}" / f"agent_{self.agent_id}_{self.seed}"
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
    # Initialize per-episode tool-call message recorder when tool_use is enabled
    self.tool_trajectory: Optional[SingleTurnToolCallTrajectory] = (
        SingleTurnToolCallTrajectory() if self.tool_use else None
    )
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
    # Initialize tool manager for function-call protocol only when enabled
    try:
      if self.tool_use:
        self.tool_manager = build_default_tool_manager()
      else:
        self.tool_manager = None
    except Exception:
      self.tool_manager = None

  def _build_initial_user_prompt(self, state_text: str) -> str:
    # Build initial user prompt mirroring tests/grl_agents_tests/single_sokoban_coding_agent_test.py
    symbols = self.env_config.get("grid_vocab", {}) or {}
    symbols_txt = ", ".join([f"{k}: {v}" for k, v in symbols.items()]) if symbols else ""
    actions_txt = ", ".join(self.env_config.get("action_lookup", {}).values())
    parts = [
        self.prompt,
        "",
        "Initial Sokoban state:",
        str(state_text),
        "",
        f"The meaning of each symbol is: {symbols_txt}",
        f"Your available actions are: {actions_txt}",
        f"Separator: '{self.action_separator}'",
        f"Max actions total: {self.max_actions_all_turns}",
    ]
    if self.tool_use:
      max_steps = int(self.agent_config.get("max_steps", 10))
      parts.append(
          "Tool-call budget: at most "
          f"{max_steps} tool calls this turn. Include a line 'Tool calls left: <k>' in every response and call <function=finish> on the final step."
      )
    return "\n".join(parts)

  def reset(self, seed: int | None = None) -> EnvOutput:
    # Use base reset to clear history/counters and get initial observation
    env_out = super().reset(seed=seed)
    # Rebuild initial messages to mirror external test prompt structure
    initial_user = self._build_initial_user_prompt(env_out.state)
    self.messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": initial_user},
    ]
    # Reset tool-call trajectory if enabled and capture the initial two messages
    if self.tool_trajectory is not None:
      self.tool_trajectory.clear()
      self.tool_trajectory.extend(self.messages)
    # Reset tool-call counter for the new turn
    self.tool_calls_this_turn = 0
    return env_out

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

    # Normalize action names to 0-based indices expected by SokobanEnv/Gym (0..3)
    # Allow config to define 1-based mapping; convert here.
    configured = self.env_config["action_lookup"]
    name_to_index0 = {}
    for k, name in configured.items():
      try:
        k_int = int(k)
      except Exception:
        k_int = k
      # Map to 0-based index
      if k_int in (1, 2, 3, 4):
        idx0 = k_int - 1
      elif k_int in (0, 1, 2, 3):
        idx0 = k_int
      else:
        # Fallback: attempt common ordering
        order = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
        idx0 = order.get(str(name), None)
      if idx0 is not None:
        name_to_index0[str(name)] = idx0
        name_to_index0[str(name).lower()] = idx0


    valid_actions: List[int] = []
    invalid_actions: List[str] = []

    for action_str in actions:
      try:
        action_str_clean = action_str.strip()
        # Prefer name-based mapping to 0-based
        if action_str_clean in name_to_index0:
          idx0 = name_to_index0[action_str_clean]
          valid_actions.append(idx0)
        elif action_str_clean.lower() in name_to_index0:
          idx0 = name_to_index0[action_str_clean.lower()]
          valid_actions.append(idx0)
        else:
          # Treat as numeric; convert to 0-based if in 1..4 else expect 0..3
          num = int(action_str_clean)
          if num in (1, 2, 3, 4):
            idx0 = num - 1
          else:
            idx0 = num
          if idx0 in (0, 1, 2, 3):
            valid_actions.append(idx0)
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
    if self.tool_trajectory is not None:
      self.tool_trajectory.add({"role": "assistant", "content": block})

    if fn_name.lower() in {"finish", "submit"}:
      result_text = params.get("result", "")
      action_line = result_text.split("\n", 1)[0].split("---", 1)[0].strip()
      env_out = self.get_env_outputs(action_line)
      # Turn finalized; reset counter for potential next turn
      self.tool_calls_this_turn = 0
      return True, env_out

    tool_out: Dict[str, Any] = {"output": "", "exit_code": "0"}
    try:
      if self.tool_use and getattr(self, "tool_manager", None) is not None:
        tool_out = self.tool_manager.execute(fn_name, params)
      else:
        tool_out = {"output": f"Tool manager unavailable for {fn_name}.", "exit_code": "-1"}
    except Exception as e:
      tool_out = {"output": f"Error executing tool {fn_name}: {e}", "exit_code": "-1"}

    feedback = self._format_tool_observation(fn_name, tool_out)
    self.messages.append({"role": "user", "content": feedback})
    if self.tool_trajectory is not None:
      self.tool_trajectory.add({"role": "user", "content": feedback})
    # Count this tool call; if budget exhausted without finish, force finalize with empty answer
    try:
      self.tool_calls_this_turn += 1
    except Exception:
      self.tool_calls_this_turn = self.tool_calls_this_turn if isinstance(self.tool_calls_this_turn, int) else 0
      self.tool_calls_this_turn += 1
    if self.tool_calls_this_turn >= self.max_tool_steps:
      # Force a final step with empty actions
      env_out = self.get_env_outputs("")
      # Reset counter for a potential next turn
      self.tool_calls_this_turn = 0
      return True, env_out
    return False, None

  def get_final_rollout_states(self) -> Dict[str, Any]:
    # Extend base rollout state with tool_msg transcript when tool_use is enabled
    base = super().get_final_rollout_states()
    if self.tool_trajectory is not None:
      base["tool_msg"] = self.tool_trajectory.get()
    else:
      base["tool_msg"] = []
    return base
