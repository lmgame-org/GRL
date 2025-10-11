## Env · Tool · Agent — Lite, Elegant, Insightful

This directory defines a compact RL interface for multi‑turn LLM agents interacting with environments, optionally augmented by Tools. It’s designed to be:

- Lite but complete for RL rollout and training
- Modular to swap environments and agents
- Tool‑friendly to enable function‑calling workflows


### Philosophy
- Env is stateful and episode‑scoped. It exposes reset/step/render.
- Agent is the orchestrator. It manages prompts, parses LLM outputs, calls Tools when needed, steps the Env, and records trajectories.
- Tools are pluggable capabilities available to agents (e.g., file editing, search, bash). They are optional but powerful for coding or planning tasks.


## Core APIs

### Env API
File: `grl_agents/base_env.py`

```python
class BaseEnv:
  def reset(self, seed: int | None = None, **kwargs) -> Any: ...
  def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...
  def render(self, mode: str = "text") -> Any: ...
  def close(self) -> None: ...
```

Contract:
- reset returns the initial observation
- step returns (observation, reward, done, info)
- render returns a human‑readable view (text or rgb array)

### Agent API
File: `grl_agents/base_agent.py`

Key types:
```python
@dataclass
class EnvOutput:
  truncated: bool
  terminated: bool
  state: Any
  reward: float
  info: dict

@dataclass
class SingleTurnTrajectory:
  state: Any
  actions_left: int
  actions: list[int]
  reward: float
  info: dict
  llm_response: str
  llm_raw_response: str
```

Class skeleton:
```python
class BaseAgent:
  def __init__(self, config: dict, group_id=0, agent_id=0, seed=None, tag=None): ...

  # lifecycle
  def initialize_env(self) -> None: ...           # set self.env
  def reset(self, seed: int | None = None) -> EnvOutput: ...
  def close(self) -> None: ...

  # LLM IO
  def get_llm_prompts(self, env_out: EnvOutput) -> list[dict]: ...
  def parse_llm_response(self, llm_response: str, enable_think: bool) -> tuple[str, list[str]]: ...

  # stepping
  def get_env_outputs(self, llm_response: str) -> EnvOutput: ...

  # summaries
  def get_final_rollout_states(self) -> dict: ...
```

### Tool API
Directory: `grl_agents/tools/`

Core building blocks:
```python
class tool: ...                      # decorator to register a method as a tool
class ToolGroup: ...                 # groups related tools and exposes schemas
class ToolManager: ...               # aggregates groups; dispatches execution

# Example tool group
class ExecuteBashTools(ToolGroup):
  @tool(schema={...}, description="...")
  def execute_bash(self, args: dict) -> dict: ...

# Build a manager with default tools
from grl_agents.tools import build_default_tool_manager
tm = build_default_tool_manager()
schemas = tm.get_schemas()           # for LLM function-calling schemas
out = tm.execute("execute_bash", {"cmd": "echo hi"})
```

Included groups:
- `execute_bash`: run shell commands within the workspace
- `search`: grep‑like search across files/dirs
- `file_editor`: simple file view/edit/insert/undo
- `finish`: signal completion and return a final result string

Workspace root resolution can be overridden via `GRL_WORKSPACE_ROOT`.

## Putting It Together: Agent as the Control Center
An agent integrates Tools and Env to form the interaction loop:

1) Reset env: `env_out = agent.reset(seed)`
2) Build prompts: `messages = agent.get_llm_prompts(env_out)`
3) Call an LLM (optionally with tool schemas from `ToolManager`)
4) Parse response: `processed, actions = agent.parse_llm_response(text, enable_think)`
5) If tools are requested, execute via `ToolManager` and append observations to `messages`
6) Execute environment actions: `env_out = agent.get_env_outputs(text)`
7) Repeat until done or action/turn budget exhausted
8) Export metrics/trajectories via `get_final_rollout_states()`

This architecture makes the Agent the “main control” that flexibly coordinates LLM, Tools, and Env.


## Batching: Dataset and Group Builder
Files:
- `grl_agents/agent_group_builder.py` — `AgentGroupBuilder` exposes `make_agents(parallel=True, max_workers=4)`, which builds a group of fresh agents (and their environments) from seeds. It uses a lightweight process pool to preprocess seeds in parallel, then constructs agents in-process.
- `grl_agents/rl_dataset.py` — `RLDataset.get_batch(index)` produces a list of `AgentGroupBuilder` instances with deterministic seeds.

```python
from grl_agents.rl_dataset import RLDataset
import asyncio

cfg = {"agent_config": {...}, "env_config": {...}}
ds = RLDataset(base_config=cfg, groups_per_batch=2, seeds_per_group=4, seed_start=0)
groups = ds.get_batch(index=0)

# Build agents for the first group
agents = asyncio.run(groups[0].make_agents(parallel=True, max_workers=4))
```


## Example: Sokoban Coding Agent
- Env: `SokobanEnv` (text grid rendering)
- Agent: `SokobanCodingAgent` (parses action sequences like `Right || Up` and steps the env)
- Tools: available via `grl_agents.tools` (optional, useful for coding/analysis workflows)

You can implement your own agent by subclassing `BaseAgent`, implementing `initialize_env` and `get_env_outputs`, and (optionally) wiring a `ToolManager` for function‑calling.


## Design Goals
- Elegant: small, composable interfaces with sensible defaults
- Lite: minimal ceremony to plug in new agents/envs
- Insightful: clear separation of concerns; Agents as orchestrators; Tools as first‑class capabilities
