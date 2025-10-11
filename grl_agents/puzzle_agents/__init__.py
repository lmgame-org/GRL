# Puzzle agents namespace

from typing import Dict

# Track puzzle agents that failed to import
UNAVAILABLE_PUZZLE_AGENTS: Dict[str, str] = {}


def _safe_import(import_fn, agent_key: str) -> None:
  try:
    import_fn()
  except Exception as e:
    UNAVAILABLE_PUZZLE_AGENTS[agent_key] = str(e)


def list_unavailable_puzzle_agents() -> Dict[str, str]:
  return dict(UNAVAILABLE_PUZZLE_AGENTS)


# Safe-import puzzle agents to trigger registration
_safe_import(
  lambda: __import__(
    "grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent",
    fromlist=["SokobanCodingAgent"],
  ),
  "sokobanCodingAgent",
)
