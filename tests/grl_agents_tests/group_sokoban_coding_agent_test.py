import os
import json
import re
import shutil
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


def _run_async(coro):
  import asyncio
  return asyncio.run(coro)


def main():
  # Resolve repo root and required directories
  repo_root = _find_repo_root(Path(__file__).resolve())
  cache_dir = repo_root / "cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  log_file = cache_dir / "group_tool_demo_log.txt"
  try:
    log_file.write_text("", encoding="utf-8")
  except Exception:
    pass

  # Per-repo workspace where each agent gets its own subfolder
  workspace_root = repo_root / "workspace"
  workspace_root.mkdir(parents=True, exist_ok=True)

  # Import agent config and prompts
  from grl_agents.puzzle_agents.sokoban_coding_agent.config import (
      get_sokoban_coding_agent_config,
  )
  from grl_agents.puzzle_agents.sokoban_coding_agent.prompts import (
      system_prompt as sokoban_system_prompt,
      prompt as sokoban_user_prompt,
  )

  # Dataset/group setup
  from grl_agents.rl_dataset import RLDataset
  from grl_agents.agent_group_builder import AgentGroupBuilder

  base_conf = get_sokoban_coding_agent_config()
  groups_per_batch = 1
  seeds_per_group = 2  # can scale to 4 if desired
  seed_start = 123

  dataset = RLDataset(
      base_config=base_conf,
      groups_per_batch=groups_per_batch,
      seeds_per_group=seeds_per_group,
      seed_start=seed_start,
  )

  builders = dataset.get_batch(index=0)
  _append_log(log_file, f"=== Group test: groups={len(builders)} agents_per_group={seeds_per_group} ===")

  # LLM provider setup (same as single-agent test)
  provider = "openai"
  model = "gpt-5"
  _load_env_from_dotenv(repo_root)
  have_key = bool(os.getenv("OPENAI_API_KEY")) if provider == "openai" else True

  # Tools and API wrapper
  from grl_agents.tools import build_default_tool_manager
  from grl_agents.api_serving.api_providers import chat_completion, LLMProviderError

  # Iterate groups
  for g_idx, builder in enumerate(builders):
    # Create agents for the group
    agents = _run_async(builder.make_agents(parallel=True, max_workers=4))
    _append_log(log_file, f"Group {g_idx}: spawned {len(agents)} agents")

    # Run each agent with its own isolated workspace
    for agent in agents:
      agent_folder = workspace_root / f"group_{g_idx}" / f"agent_{agent.agent_id}"
      agent_folder.mkdir(parents=True, exist_ok=True)

      # Point tools to agent-specific workspace
      os.environ["GRL_WORKSPACE_ROOT"] = str(agent_folder)

      # Ensure prompts match prompts.py explicitly
      agent.system_prompt = sokoban_system_prompt
      agent.prompt = sokoban_user_prompt

      # Reset environment with deterministic per-group seed
      group_seed = seed_start + g_idx
      env_out = agent.reset(seed=group_seed)

      # Build initial user message mirroring single-agent test
      symbols = agent.env_config.get("grid_vocab", {})
      symbols_txt = ", ".join([f"{k}: {v}" for k, v in symbols.items()]) if symbols else ""
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

      # Build and attach tool manager per agent
      tm = build_default_tool_manager()
      agent.tool_manager = tm
      tool_schemas = tm.get_schemas()

      _append_log(log_file, f"Agent {agent.agent_id} workspace: {agent_folder}")
      _append_log(log_file, f"Model provider={provider} model={model or '(default)'}")

      # If API key absent, skip calling model for this agent (but continue others)
      if not have_key:
        _append_log(log_file, "OPENAI_API_KEY not set; skipping model calls for this agent.")
        continue

      # Limited single-turn tool-calling loop per agent (mirrors single-agent test)
      MAX_TURNS = getattr(agent, "max_turns", 1)
      MAX_STEPS = agent.agent_config.get("max_steps", 10)
      finished = False

      # Helper functions copied inline from single test
      def _parse_function_blocks(text: str):
        try:
          return re.findall(r"<function\s*=\s*[^>]+>.*?</function>", text, flags=re.DOTALL)
        except Exception:
          return []

      def _parse_function_call(block: str):
        try:
          m = re.search(r"<function\s*=\s*([^>]+)>", block)
          fn = m.group(1).strip() if m else ""
          params = {}
          for key, val in re.findall(r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", block, flags=re.DOTALL):
            key = key.strip()
            raw = val.strip()
            try:
              params[key] = json.loads(raw)
            except Exception:
              params[key] = raw
          return fn, params
        except Exception:
          return "", {}

      def _format_tool_observation(function_name: str, tool_out: dict) -> str:
        output = str(tool_out.get("output", ""))
        exit_code = str(tool_out.get("exit_code", ""))
        if function_name in {"execute_bash", "bash"}:
          return f"Exit code: {exit_code}\nExecution output of [{function_name}]:\n{output}"
        return f"Execution output of [{function_name}]:\n{output}"

      def _execute_single_tool_call(agent_local, tm_local, llm_response: str):
        blocks = _parse_function_blocks(llm_response)
        if not blocks:
          return False, None
        block = blocks[0]
        fn_name, params = _parse_function_call(block)
        if not fn_name:
          return False, None
        agent_local.messages.append({"role": "assistant", "content": block})
        if fn_name.lower() in {"finish", "submit"}:
          result_text = params.get("result", "")
          action_line = result_text.split("\n", 1)[0].split("---", 1)[0].strip()
          env_out2 = agent_local.get_env_outputs(action_line)
          return True, env_out2
        try:
          tool_out = tm_local.execute(fn_name, params)
        except Exception as e:
          tool_out = {"output": f"Error executing tool {fn_name}: {e}", "exit_code": "-1"}
        feedback = _format_tool_observation(fn_name, tool_out)
        agent_local.messages.append({"role": "user", "content": feedback})
        return False, None

      for turn_idx in range(MAX_TURNS):
        step_calls = 0
        while step_calls < MAX_STEPS and not finished:
          step_calls += 1
          _append_log(log_file, f"[G{g_idx} A{agent.agent_id}] Tool Step {turn_idx+1}.{step_calls}")

          try:
            llm_response_raw = chat_completion(
                messages=agent.messages,
                provider=provider,
                model=model,
                temperature=1,
                extra_args={"tools": tool_schemas},
            )
          except (Exception, LLMProviderError) as e:
            _append_log(log_file, f"LLM error: {e}")
            break

          llm_response = (
              llm_response_raw if isinstance(llm_response_raw, str) else str(llm_response_raw)
          )
          if not llm_response:
            _append_log(log_file, f"LLM returned empty response at step {turn_idx+1}.{step_calls}")
            continue

          try:
            tool_names = re.findall(r"<function\s*=\s*([^>]+)>", llm_response)
          except Exception:
            tool_names = []
          for fn in tool_names:
            _append_log(log_file, f"[ToolCall] G{g_idx} A{agent.agent_id} step {turn_idx+1}.{step_calls}: {fn}")

          before_len = len(agent.messages)
          finished, env_out2 = _execute_single_tool_call(agent, tm, llm_response)
          if finished:
            _append_log(log_file, f"Executed final actions for G{g_idx} A{agent.agent_id}.")
            if env_out2 is not None:
              _append_log(log_file, "=== Final observation ===")
              _append_log(log_file, str(env_out2.state))
            break

          if llm_response and ("||" in llm_response) and ("<function=" not in llm_response) and ("<answer>" not in llm_response):
            env_out3 = agent.get_env_outputs(llm_response)
            _append_log(log_file, f"Executed actions from plain string for G{g_idx} A{agent.agent_id}.")
            _append_log(log_file, str(env_out3.state))
            finished = True
            break

          remaining = max(0, MAX_STEPS - step_calls)
          for m in agent.messages[before_len:]:
            if isinstance(m, dict) and m.get("role") == "user":
              m["content"] = f"{m.get('content', '')}\nTool calls left: {remaining}"

      # Cleanup the agent-specific workspace directory after run
      try:
        shutil.rmtree(agent_folder, ignore_errors=True)
      except Exception:
        pass

    # After the group run, collect trajectories for these agents
    group_rows = _run_async(builder.generate_trajectories(agents=agents, reset=False, max_workers=4))
    _append_log(log_file, "=== Group rollouts (existing agents) ===")
    try:
      _append_log(log_file, json.dumps(group_rows))
    except Exception:
      _append_log(log_file, str(group_rows))

  # Also run dataset-level collection (fresh agents) and log
  ds_rows = _run_async(dataset.generate_group_trajectories(index=0, reset=True, max_workers=4))
  _append_log(log_file, "=== Dataset rollouts (fresh agents) ===")
  try:
    _append_log(log_file, json.dumps(ds_rows))
  except Exception:
    _append_log(log_file, str(ds_rows))



if __name__ == "__main__":
  main()