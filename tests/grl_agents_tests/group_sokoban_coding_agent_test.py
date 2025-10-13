import os
import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


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
  # Build dataset as list of groups with individual seeds
  dataset = RLDataset(base_configs=[base_conf], seeds=[123])

  # Build two groups; each group will have two agents sharing the same seed
  builders = [
      AgentGroupBuilder(seed=123, config=base_conf, group_num=2),
      AgentGroupBuilder(seed=456, config=base_conf, group_num=2),
  ]
  _append_log(log_file, f"=== Group test: groups={len(builders)} agents_per_group={2} ===")

  # LLM provider setup (same as single-agent test)
  provider = "openai"
  model = "gpt-5-mini"
  _load_env_from_dotenv(repo_root)
  have_key = bool(os.getenv("OPENAI_API_KEY")) if provider == "openai" else True

  # Tools and API wrapper
  from grl_agents.tools import build_default_tool_manager
  from grl_agents.api_serving.api_providers import chat_completion, LLMProviderError

  # Iterate groups
  all_group_rows = []
  for g_idx, builder in enumerate(builders):
    # Create agents for the group
    agents = _run_async(builder.make_agents())
    _append_log(log_file, f"Group {g_idx}: spawned {len(agents)} agents")

    def _run_agent(agent):
      agent_folder = workspace_root / f"group_{g_idx}" / f"agent_{agent.agent_id}"
      agent_folder.mkdir(parents=True, exist_ok=True)
      # Ensure all tool/file operations happen under this per-agent workspace
      os.environ["GRL_WORKSPACE_ROOT"] = str(agent_folder)

      # Prepare per-agent logs in cache
      interaction_log = cache_dir / f"group{g_idx}_agent{agent.agent_id}_interaction.log"
      tools_log = cache_dir / f"group{g_idx}_agent{agent.agent_id}_tools.log"
      try:
        interaction_log.write_text("", encoding="utf-8")
        tools_log.write_text("", encoding="utf-8")
      except Exception:
        pass

      # Ensure prompts match prompts.py explicitly
      try:
        agent.system_prompt = sokoban_system_prompt
        agent.prompt = sokoban_user_prompt
      except Exception:
        pass

      # Reset environment with deterministic per-group seed
      group_seed = builders[0].seed
      env_out = agent.reset(seed=group_seed)
      _append_log(interaction_log, f"Initial observation: {env_out.state}")

      # Build initial user message
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
      # Bind tool manager to the per-thread workspace root
      try:
        tm.bind_workspace(agent_folder)
      except Exception:
        pass

      _append_log(log_file, f"Agent {agent.agent_id} workspace: {agent_folder}")
      _append_log(log_file, f"Model provider={provider} model={model or '(default)'}")
      _append_log(interaction_log, f"Init messages: {json.dumps(agent.messages) if isinstance(agent.messages, list) else str(agent.messages)}")

      # If API key absent, skip calling model for this agent but keep going
      if not have_key:
        _append_log(log_file, "OPENAI_API_KEY not set; skipping model calls for this agent.")
        return

      MAX_TURNS = getattr(agent, "max_turns", 1)
      MAX_STEPS = agent.agent_config.get("max_steps", 10)

      def _format_tool_observation(function_name: str, tool_out: dict) -> str:
        output = str(tool_out.get("output", ""))
        exit_code = str(tool_out.get("exit_code", ""))
        if function_name in {"execute_bash", "bash"}:
          return f"Exit code: {exit_code}\nExecution output of [{function_name}]:\n{output}"
        return f"Execution output of [{function_name}]:\n{output}"

      # Multi-turn loop using execute_tool_call where applicable
      for turn_idx in range(MAX_TURNS):
        step_calls = 0
        finished = False
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
            _append_log(interaction_log, f"LLM error: {e}")
            break

          llm_response = llm_response_raw if isinstance(llm_response_raw, str) else str(llm_response_raw)
          if not llm_response:
            _append_log(log_file, f"LLM returned empty response at step {turn_idx+1}.{step_calls}")
            _append_log(interaction_log, f"Empty LLM response at step {turn_idx+1}.{step_calls}")
            continue

          # Prefer execute_tool_call helper
          _append_log(interaction_log, f"LLM response: {llm_response[:2000]}")
          try:
            tool_names = re.findall(r"<function\s*=\s*([^>]+)>", llm_response)
          except Exception:
            tool_names = []
          if tool_names:
            _append_log(tools_log, f"Tool blocks detected: {', '.join(tool_names)}")
          done, env_out_done = agent.execute_tool_call(llm_response)
          if done:
            _append_log(log_file, f"Executed final actions for G{g_idx} A{agent.agent_id}.")
            _append_log(interaction_log, "Final actions executed via <function=finish>.")
            if env_out_done is not None:
              _append_log(log_file, "=== Final observation ===")
              _append_log(log_file, str(env_out_done.state))
              _append_log(interaction_log, f"Final observation: {env_out_done.state}")
            break

          # Fallback: direct actions parsing when no function blocks
          if ("<function=" not in llm_response) and ("<answer>" in llm_response or "||" in llm_response):
            env_out3 = agent.get_env_outputs(llm_response)
            _append_log(log_file, f"Executed actions from plain string for G{g_idx} A{agent.agent_id}.")
            _append_log(log_file, str(env_out3.state))
            _append_log(interaction_log, f"Executed plain actions, obs: {env_out3.state}")
            finished = True
            break

      # Per-agent cleanup is deferred; group folder will be removed after all agents finish
      try:
        tm.unbind_workspace()
      except Exception:
        pass

    # Run all agents in the group concurrently using threads
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as executor:
      list(executor.map(_run_agent, agents))

    # After the group run, collect trajectories for these agents
    group_rows = _run_async(builder.generate_full_trajectories(agents=agents))
    # Accumulate all agent rollouts across groups for a combined log/file
    try:
      all_group_rows.extend(list(group_rows))
    except Exception:
      # Fallback if group_rows is not iterable as expected
      try:
        all_group_rows.append(group_rows)
      except Exception:
        pass
    _append_log(log_file, "=== Group rollouts (existing agents) ===")
    try:
      _append_log(log_file, json.dumps(group_rows))
    except Exception:
      _append_log(log_file, str(group_rows))

    # Persist a single rollout file containing all agents' rollout states for this group
    try:
      group_rollout_file = cache_dir / f"group{g_idx}_rollouts.json"
      group_rollout_file.write_text(json.dumps(group_rows), encoding="utf-8")
    except Exception:
      try:
        group_rollout_file.write_text(str(group_rows), encoding="utf-8")
      except Exception:
        pass

    # Cleanup: remove the entire group workspace directory now that all agents are done
    try:
      group_workspace = workspace_root / f"group_{g_idx}"
      shutil.rmtree(group_workspace, ignore_errors=True)
    except Exception:
      pass

  # Also run dataset-level collection (fresh agents) and log
  ds_rows = _run_async(dataset.collect_group_trajectories(index=0))
  _append_log(log_file, "=== Dataset rollouts (fresh agents) ===")
  try:
    _append_log(log_file, json.dumps(ds_rows))
  except Exception:
    _append_log(log_file, str(ds_rows))

  # Persist a combined rollout file containing all agents across all groups
  try:
    combined_file = cache_dir / "group_rollouts.json"
    combined_file.write_text(json.dumps(all_group_rows), encoding="utf-8")
    _append_log(log_file, f"=== Combined group rollouts saved: {len(all_group_rows)} agents ===")
  except Exception:
    try:
      combined_file.write_text(str(all_group_rows), encoding="utf-8")
    except Exception:
      pass



if __name__ == "__main__":
  main()