import os
import json
import re
from pathlib import Path
from pprint import pprint


def _find_repo_root(start: Path) -> Path:
  cur = start.resolve()
  for _ in range(10):
    if (cur / "pyproject.toml").exists() or (cur / ".git").exists() or (cur / "grl_agents").exists():
      return cur
    if cur.parent == cur:
      break
    cur = cur.parent
  return start.resolve()


def _load_env_from_dotenv(repo_root: Path):
  dotenv_path = repo_root / ".env"
  if not dotenv_path.exists():
    return
  try:
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      if line.lower().startswith("export "):
        line = line[len("export "):].strip()
      if "=" not in line:
        continue
      key, val = line.split("=", 1)
      key = key.strip()
      val = val.strip().strip('"').strip("'")
      if not os.getenv(key):
        os.environ[key] = val
  except Exception:
    pass


def _parse_function_blocks(text: str):
  try:
    return re.findall(r"<function\s*=\s*[^>]+>.*?</function>", text, flags=re.DOTALL)
  except Exception:
    return []


 


def main():
  # Resolve repo root and required directories
  repo_root = _find_repo_root(Path(__file__).resolve())
  cache_dir = repo_root / "cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  log_file = cache_dir / "tool_demo_log.txt"
  try:
    log_file.write_text("", encoding="utf-8")
  except Exception:
    pass

  # Agent manages its own per-agent workspace; no global override here

  # Import agent and config
  from grl_agents.puzzle_agents.sokoban_coding_agent.config import (
      get_sokoban_coding_agent_config,
  )
  from grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent import (
      SokobanCodingAgent,
  )
  from grl_agents.puzzle_agents.sokoban_coding_agent.prompts import (
      system_prompt as sokoban_system_prompt,
      prompt as sokoban_user_prompt,
  )
  conf = get_sokoban_coding_agent_config()

  # Build agent and reset
  agent = SokobanCodingAgent(config=conf, tag="toolCallingDemo-script")
  # Ensure prompts match prompts.py explicitly
  agent.system_prompt = sokoban_system_prompt
  agent.prompt = sokoban_user_prompt
  env_out = agent.reset(seed=123)

  # Compose initial user message mirroring external test
  symbols = agent.env_config.get("grid_vocab", {})
  symbols_txt = (
      ", ".join([f"{k}: {v}" for k, v in symbols.items()]) if symbols else ""
  )
  actions_txt = ", ".join(agent.env_config.get("action_lookup", {}).values())
  initial_user = (
      f"{sokoban_user_prompt}\n\n"
      f"Initial Sokoban state:\n{env_out.state}\n\n"
      f"The meaning of each symbol is: {symbols_txt}\n"
      f"Your available actions are: {actions_txt}\n"
      f"Separator: '{agent.action_separator}'\n"
      f"Max actions total: {agent.max_actions_all_turns}\n"
      f"Tool-call budget: at most {agent.agent_config.get('max_steps', 10)} tool calls this turn. Include a line 'Tool calls left: <k>' in every response and call <function=finish> on the final step.\n"
  )
  agent.messages = [
      {"role": "system", "content": sokoban_system_prompt},
      {"role": "user", "content": initial_user},
  ]

  # Print brief state and helpers
  print("\n=== Initial Game State ===")
  _append_log(log_file, "\n=== Initial Game State ===")
  print(env_out.state)
  _append_log(log_file, str(env_out.state))
  print("Symbols:", symbols_txt)
  _append_log(log_file, f"Symbols: {symbols_txt}")
  print("Actions:", actions_txt)
  _append_log(log_file, f"Actions: {actions_txt}")
  print("Separator:", agent.action_separator)
  _append_log(log_file, f"Separator: {agent.action_separator}")
  print("Max actions total:", agent.max_actions_all_turns)
  _append_log(log_file, f"Max actions total: {agent.max_actions_all_turns}")
  print("Agent workspace:", getattr(agent, "workspace_path", "(none)"))
  _append_log(log_file, f"Agent workspace: {getattr(agent, 'workspace_path', '(none)')}")
  print("Tool-call budget (max steps):", agent.agent_config.get("max_steps", 10))
  _append_log(
      log_file, f"Tool-call budget (max steps): {agent.agent_config.get('max_steps', 10)}"
  )

  # Provider/model setup
  provider = "openai"
  model = "gpt-5"
  print(f"Model provider={provider} model={model or '(default)'}")
  _append_log(log_file, f"=== Model provider={provider} model={model or '(default)'} ===")

  # Load OPENAI_API_KEY if present
  _load_env_from_dotenv(repo_root)
  if not os.getenv("OPENAI_API_KEY") and provider == "openai":
    msg = (
        "OPENAI_API_KEY is not set. Please add it to .env at repo root or export it."
    )
    print(msg)
    _append_log(log_file, msg)
    return

  # Tool schemas from agent's tool manager (if available)
  tm = getattr(agent, "tool_manager", None)
  tool_schemas = tm.get_schemas() if tm is not None else None

  # LLM provider wrapper
  from grl_agents.api_serving.api_providers import chat_completion, LLMProviderError

  MAX_TURNS = getattr(agent, "max_turns", 1)
  MAX_STEPS = agent.agent_config.get("max_steps", 10)
  final_env_out = None
  finished = False

  for turn_idx in range(MAX_TURNS):
    step_calls = 0
    while step_calls < MAX_STEPS and not finished:
      step_calls += 1
      print(f"\n=== Tool Step {turn_idx+1}.{step_calls} ===")
      _append_log(log_file, f"\n=== Tool Step {turn_idx+1}.{step_calls} ===")

      try:
        extra_args = {"tools": tool_schemas} if tool_schemas else None
        llm_response_raw = chat_completion(
            messages=agent.messages,
            provider=provider,
            model=model,
            temperature=1,
            extra_args=extra_args,
        )
      except (Exception, LLMProviderError) as e:
        print(f"LLM error: {e}")
        _append_log(log_file, f"LLM error: {e}")
        return

      llm_response = (
          llm_response_raw if isinstance(llm_response_raw, str) else str(llm_response_raw)
      )
      if not llm_response:
        print("\nLLM returned empty response; continuing.")
        _append_log(
            log_file,
            f"=== LLM returned empty response at step {turn_idx+1}.{step_calls} ===",
        )
        continue

      # Log declared tool calls
      try:
        tool_names = re.findall(r"<function\s*=\s*([^>]+)>", llm_response)
      except Exception:
        tool_names = []
      tool_names_no_finish = [n for n in tool_names if n.lower() not in {"finish", "submit"}]
      for fn in tool_names:
        _append_log(log_file, f"=== Tool call step {turn_idx+1}.{step_calls}: {fn} ===")
        print(f"[ToolCall] step {turn_idx+1}.{step_calls}: {fn}")

      before_len = len(agent.get_messages())
      finished, env_out = agent.execute_tool_call(llm_response)
      if finished:
        final_env_out = env_out
        print("\nExecuted final actions in environment.")
        _append_log(log_file, "\nExecuted final actions in environment.")
        _append_log(log_file, f"=== User messages before finish step {turn_idx+1}.{step_calls} ===")
        _append_log(log_file, _user_messages_repr(agent.get_messages()))
        if final_env_out is not None:
          _append_log(log_file, "=== Final observation ===")
          _append_log(log_file, str(final_env_out.state))
        break

      if llm_response and ("||" in llm_response) and ("<function=" not in llm_response) and ("<answer>" not in llm_response):
        final_env_out = agent.get_env_outputs(llm_response)
        print("\nExecuted actions from plain string.")
        _append_log(log_file, "\nExecuted actions from plain string.")
        _append_log(log_file, f"=== Executed plain actions at step {turn_idx+1}.{step_calls} ===")
        _append_log(log_file, llm_response)
        finished = True
        break

      remaining = max(0, MAX_STEPS - step_calls)
      for m in agent.get_messages()[before_len:]:
        if isinstance(m, dict) and m.get("role") == "user":
          m["content"] = f"{m.get('content', '')}\nTool calls left: {remaining}"

      new_msgs = agent.get_messages()[before_len:]
      new_user_feedback = [m.get("content", "") for m in new_msgs if isinstance(m, dict) and m.get("role") == "user"]
      for idx, fn in enumerate(tool_names_no_finish):
        if idx < len(new_user_feedback):
          _append_log(log_file, f"=== Feedback step {turn_idx+1}.{step_calls}: {fn} ===")
          _append_log(log_file, new_user_feedback[idx])
          print(f"\n[ToolFeedback] step {turn_idx+1}.{step_calls} tool={fn}\n{new_user_feedback[idx]}")

  if final_env_out is None:
    print("\nNo finish received within iteration cap.")
    _append_log(log_file, "\nNo finish received within iteration cap.")
  else:
    print("\nFinal Observation:\n", final_env_out.state)
    _append_log(log_file, "\nFinal Observation:")
    _append_log(log_file, str(final_env_out.state))
    print("Reward:", final_env_out.reward)
    _append_log(log_file, f"Reward: {final_env_out.reward}")
    print("Info:", final_env_out.info)
    _append_log(log_file, f"Info: {final_env_out.info}")
    _append_log(log_file, "=== Final observation ===")
    _append_log(log_file, str(final_env_out.state))

  row = agent.get_final_rollout_states()
  print("\nMetrics:")
  _append_log(log_file, "\nMetrics:")
  metrics_obj = row.get("metrics", {})
  pprint(metrics_obj)
  try:
    _append_log(log_file, str(metrics_obj))
  except Exception:
    _append_log(log_file, "<metrics not serializable>")
  print("\nHistory length:", len(row.get("history", [])))
  _append_log(log_file, f"\nHistory length: {len(row.get('history', []))}")


def _append_log(path: Path, content: str):
  try:
    with path.open("a", encoding="utf-8") as f:
      f.write(content)
      if not content.endswith("\n"):
        f.write("\n")
  except Exception:
    pass


def _user_messages_repr(messages):
  try:
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return repr(users)
  except Exception:
    return repr(messages)


if __name__ == "__main__":
  main()


