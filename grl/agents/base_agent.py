from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random
from grl.agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput


class BaseAgent:
  """
  Abstract base class for agents. Provides high-level method signatures for agent lifecycle, environment interaction, LLM interface, trajectory management, and rollout collection.
  """

  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    """Initialize the agent with configuration and identifiers."""
    # initialize base agent
    self.group_id = group_id
    self.agent_id = agent_id
    self.tag = tag
    self.cur_turn = 0
    if seed is None:
      self.seed = random.randint(0, 2**32 - 1)
    else:
      self.seed = seed
    self.agent_config = config["agent_config"]
    self.env_config = config["env_config"]

    # handle config hyperparameters
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

    # Define turn prompt template based on enable_think (Qwen3 format)
    self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. Max response length: {max_tokens} tokens.\n"""

    self.trajectory_history = MultiTurnTrajectory(max_length=self.max_turns)
    self.raw_response_list = []  # Store all raw LLM responses for debugging
    self.messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.prompt},
    ]
    self.total_actions_consumed = 0
    self.penalty = 0.0  # Track accumulated penalty

  def get_feedback_between_turns(self, reward: float) -> str:
    """Get textual feedback for the agent."""
    return f"Reward: \n{reward}\n"

  # ─────────────────── LLM INTERFACE ───────────────────
  def get_llm_prompts(self, env_out):
    """Convert environment outputs to LLM prompts following SyncMultiTurnRollout interface."""

    # Ensure messages are initialized
    if not hasattr(self, "messages") or not self.messages:
      self.messages = [
          {"role": "system", "content": self.system_prompt},
          {"role": "user", "content": self.prompt},
      ]

    # Calculate actions remaining based on max_actions_all_turns
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

    # In the first turn, we merge the turn_content into the prompt
    if (
        self.cur_turn == 0
        and len(self.messages) == 2
        and self.messages[1]["role"] == "user"
    ):
      self.messages[1]["content"] = (
          self.messages[1]["content"] + "\n" + turn_content
      )
    else:
      reward_msg = self.get_feedback_between_turns(env_out.reward)
      turn_msg["content"] = reward_msg + " " + turn_msg["content"]
      self.messages.append(turn_msg)

    # Validate final messages before returning
    if not self.messages:
      # Emergency fallback
      self.messages = [
          {"role": "system", "content": "You are a helpful AI assistant."},
          {"role": "user", "content": "Please respond appropriately."},
      ]

    return self.messages

  def parse_llm_response(self, llm_response, enable_think=True):
    """
    Parse model response into processed llm_response and action list.
    Simple parsing that handles enable_think cases and limits actions to max_actions_per_turn.

    Args:
        llm_response: Raw LLM response string
        enable_think: Whether to expect <think> tags

    Returns:
        Tuple[str, List[str]]: (processed_llm_response, actions_list)
    """
    import re

    if enable_think:
      llm_response_with_prefix = "<think>" + llm_response
      pattern = r"<think>(.*?)</think>\s*Answer:\s*([^\n]+)"
      match = re.search(pattern, llm_response_with_prefix, re.DOTALL)

      if not match:
        # No valid pattern found, return original response with empty actions
        processed_response, actions = llm_response, [llm_response]
        return processed_response, actions
      
      think_content, action_content = match.group(1).strip(), match.group(2).strip()
        
        # Clean up special tokens
      special_tokens = [
          "<think>",
          "</think>",
          "<|im_start|>",
          "<|im_end|>",
      ]
      for special_token in special_tokens:
        action_content = action_content.replace(special_token, "").strip()
        think_content = think_content.replace(special_token, "").strip()
    else:
      match = re.search(r"Answer:\s*(\S+)", llm_response)
      if match:
          action_content = match.group(1)

      # Clean up special tokens
      special_tokens = [
          "<|im_start|>",
          "<|im_end|>",
      ]
      for special_token in special_tokens:
        action_content = action_content.replace(special_token, "").strip()

    # Parse actions using || separator
    actions = [
        action.strip()
        for action in action_content.split(self.action_separator)
        if action.strip()
    ]

    # Limit actions to max_actions_per_turn
    if len(actions) > self.max_actions_per_turn:
      actions = actions[: self.max_actions_per_turn]
      action_content = self.action_separator.join(actions)
      
    processed_response = (
      action_content if not enable_think 
      else f"<think>{think_content}</think>{action_content}"
    )
    
    return processed_response, actions

  # ─────────────────── ROLLOUT STATE COLLECTION ───────────────────
  def get_final_rollout_states(self):
    """Get final rollout states for PPO training."""
    history = []
    trajectory_deque = self.trajectory_history.get()
    for traj in trajectory_deque:
      history_entry = {
          "state": traj.state,
          "actions_left": traj.actions_left,
          "actions": traj.actions,
          "reward": traj.reward,
          "info": traj.info,
          "llm_response": traj.llm_response,
          "llm_raw_response": traj.llm_raw_response,
      }
      history.append(history_entry)

    metrics = {}

    success_values = [
        traj.info.get("success", False) for traj in trajectory_deque
    ]
    metrics[f'{self.tag or "baseAgent"}/success'] = float(any(success_values))

    total_actions = sum(len(traj.actions) for traj in trajectory_deque)
    metrics[f'{self.tag or "baseAgent"}/num_actions'] = total_actions

    action_is_effective_values = [
        traj.info.get("action_is_effective", False) for traj in trajectory_deque
    ]
    if action_is_effective_values:
      metrics[f'{self.tag or "baseAgent"}/action_is_effective'] = sum(
          action_is_effective_values
      ) / len(action_is_effective_values)
    else:
      metrics[f'{self.tag or "baseAgent"}/action_is_effective'] = 0.0

    action_is_valid_values = [
        traj.info.get("action_is_valid", False) for traj in trajectory_deque
    ]
    if action_is_valid_values:
      metrics[f'{self.tag or "baseAgent"}/action_is_valid'] = sum(
          action_is_valid_values
      ) / len(action_is_valid_values)
    else:
      metrics[f'{self.tag or "baseAgent"}/action_is_valid'] = 1.0

    if trajectory_deque:
      last_traj = trajectory_deque[-1]
      if "metrics" in last_traj.info:
        for key, value in last_traj.info["metrics"].items():
          metrics[key] = value

    row_dict = {
        "env_id": self.agent_id,
        "history": history,
        "group_id": self.group_id,
        "tag": self.tag or "baseAgent",
        "metrics": metrics,
        "penalty": self.penalty,
    }

    return row_dict

  # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────
  def reset(self, seed=None):
    """Reset agent state for new episode and return initial environment outputs."""
    # Implement group-based seeding following reference implementation
    # Agents within the same group should have the same environment (same seed)
    # Different groups should have different environments (different seeds)
    if seed is None:
      # Generate a unique seed only if no seed provided
      reset_seed = random.randint(0, 1000000)
    else:
      # Use the provided group seed directly - all agents in same group get same seed
      reset_seed = seed

    obs = self.env.reset(seed=reset_seed)
    if not obs:
      obs = self.env.render()

    self.cur_turn = 0

    self.trajectory_history.clear()
    self.raw_response_list = []
    self.total_actions_consumed = 0
    self.penalty = 0.0  # Reset penalty for new episode

    self.messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.prompt},
    ]

    # Return initial environment outputs for the rollout loop
    return EnvOutput(
        truncated=False, terminated=False, state=obs, reward=0.0, info={}
    )

  def close(self):
    """Clean up agent resources."""
    if hasattr(self, "env") and hasattr(self.env, "close"):
      self.env.close()

  def get_messages(self):
    """Get messages for debugging."""
    return self.messages

  # ─────────────────── ENVIRONMENT INTERFACE ───────────────────
  def initialize_env(self):
    """Initialize the environment for the agent."""
    pass

  def get_env_outputs(self, llm_response):
    """Process LLM outputs and get environment outputs."""
    pass

  # ─────────────────── DEBUG UTILITIES ───────────────────
  def print_processed_llm(self, processed_llm_response, actions):
    """Pretty-print processed LLM response and actions using repr."""
    try:
      sep = "=" * 50
      agent_name = self.tag or self.__class__.__name__
      print(sep)
      print(
          f"[{agent_name}] processed_llm_response: {repr(processed_llm_response)}"
      )
      print(f"[{agent_name}] actions({len(actions)}): {repr(actions)}")
      print(sep)
    except Exception:
      pass


if __name__ == "__main__":
  import argparse
  from omegaconf import OmegaConf
  from grl.agents import get_agent_cls, list_registered_agents

  def print_separator(title=""):
    sep = "=" * 60
    if title:
      print(f"\n{sep}\n{title}\n{sep}")
    else:
      print(sep)

  def print_messages(messages):
    """Pretty print the messages list."""
    for i, msg in enumerate(messages):
      role = msg.get("role", "unknown")
      content = msg.get("content", "")
      print(f"\n[{i}] Role: {role}")
      print("-" * 40)
      print(content)
      print("-" * 40)

  def main():
    parser = argparse.ArgumentParser(description="Debug Agent interactively with real environment")
    parser.add_argument(
        "--config",
        type=str,
        default="gsm8k_5_turn",
        help="Config name from agents.yaml (default: gsm8k_1_turn)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/agents.yaml",
        help="Path to config file (default: configs/agents.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment (default: 42)",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=0,
        help="Agent ID / data index (default: 0)",
    )
    args = parser.parse_args()

    # Load config
    print_separator("Loading Configuration")
    all_configs = OmegaConf.load(args.config_file)
    if args.config not in all_configs:
      print(f"Error: Config '{args.config}' not found in {args.config_file}")
      print(f"Available configs: {list(all_configs.keys())}")
      return
    config = all_configs[args.config]
    print(f"Config name: {args.config}")
    print(f"Config contents:\n{OmegaConf.to_yaml(config)}")

    # Get agent class from registry
    print_separator("Creating Agent")
    agent_name = config.get("agent_name") or config.get("agent_type")
    if not agent_name:
      print(f"Error: Config must have 'agent_name' or 'agent_type'")
      print(f"Available agents: {list_registered_agents()}")
      return

    print(f"Agent type: {agent_name}")
    print(f"Available agents: {list_registered_agents()}")

    try:
      agent_cls = get_agent_cls(agent_name)
    except KeyError as e:
      print(f"Error: {e}")
      return

    # Create agent instance
    agent = agent_cls(config, group_id=0, agent_id=args.agent_id, seed=args.seed, tag="debug")
    print(f"Agent created with:")
    print(f"  - max_turns: {agent.max_turns}")
    print(f"  - max_actions_per_turn: {agent.max_actions_per_turn}")
    print(f"  - max_actions_all_turns: {agent.max_actions_all_turns}")
    print(f"  - enable_think: {agent.enable_think}")
    print(f"  - max_tokens: {agent.max_tokens}")

    # Reset environment to get initial state
    print_separator("Resetting Environment")
    env_out = agent.reset(seed=args.seed)
    print(f"Initial state:\n{env_out.state}")

    print_separator("Starting Interactive Debug Loop")
    print("Commands:")
    print("  - Type your LLM response and press Enter")
    print("  - Type 'quit' or 'q' to exit")
    print("  - Type 'reset' to reset the agent and environment")
    print("  - Type 'history' to show trajectory history")
    print("  - Type 'messages' to show current conversation messages")

    while not env_out.terminated and not env_out.truncated:
      print_separator(f"Turn {agent.cur_turn + 1}")

      # Get LLM prompts
      messages = agent.get_llm_prompts(env_out)

      print("\n>>> LLM PROMPT (Messages to send to LLM):")
      print_messages(messages)

      # Get user input for LLM response
      print("\n>>> Enter LLM response (or command):")
      try:
        user_input = input("> ").strip()
      except EOFError:
        print("\nEOF received, exiting...")
        break

      if not user_input:
        print("Empty input, please try again.")
        continue

      # Handle commands
      if user_input.lower() in ["quit", "q", "exit"]:
        print("Exiting debug loop...")
        break

      if user_input.lower() == "reset":
        print("Resetting agent and environment...")
        agent = agent_cls(config, group_id=0, agent_id=args.agent_id, seed=args.seed, tag="debug")
        env_out = agent.reset(seed=args.seed)
        print(f"Initial state:\n{env_out.state}")
        continue

      if user_input.lower() == "history":
        print("\n>>> Trajectory History:")
        for i, traj in enumerate(agent.trajectory_history.get()):
          state_preview = traj.state[:80] + "..." if len(traj.state) > 80 else traj.state
          print(f"  [{i}] state: {state_preview}")
          print(f"      actions: {traj.actions}")
          print(f"      reward: {traj.reward}")
          print(f"      info: {traj.info}")
        continue

      if user_input.lower() == "messages":
        print("\n>>> Current Messages:")
        print_messages(agent.messages)
        continue

      # Process LLM response through the actual environment
      llm_response = user_input
      env_out = agent.get_env_outputs(llm_response)

      print("\n>>> Environment Output:")
      print(f"  State: {env_out.state}")
      print(f"  Reward: {env_out.reward}")
      print(f"  Info: {env_out.info}")
      print(f"  Terminated: {env_out.terminated}")
      print(f"  Truncated: {env_out.truncated}")

      print(f"\n>>> Agent State:")
      print(f"  Current turn: {agent.cur_turn}/{agent.max_turns}")
      print(f"  Total actions consumed: {agent.total_actions_consumed}/{agent.max_actions_all_turns}")
      print(f"  Penalty: {agent.penalty}")

      if env_out.terminated or env_out.truncated:
        print("\n>>> Episode ended!")

    print_separator("Debug Session Complete")
    print(f"Final agent state:")
    print(f"  - Turns completed: {agent.cur_turn}")
    print(f"  - Total actions: {agent.total_actions_consumed}")
    print(f"  - Penalty: {agent.penalty}")

    print("\n>>> Final Trajectory History:")
    for i, traj in enumerate(agent.trajectory_history.get()):
      state_preview = traj.state[:80] + "..." if len(traj.state) > 80 else traj.state
      print(f"  [{i}] state: {state_preview}")
      print(f"      actions: {traj.actions}")
      print(f"      reward: {traj.reward}")
      print(f"      info: {traj.info}")

    # Get final rollout states
    print("\n>>> Final Rollout States:")
    rollout_states = agent.get_final_rollout_states()
    print(f"  Metrics: {rollout_states.get('metrics', {})}")

  main()