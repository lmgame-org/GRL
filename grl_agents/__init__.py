# Namespace package for GRL agents

from typing import Type
import warnings
import importlib
import sys


def _safe_import(import_fn, agent_key: str):
  try:
    import_fn()
  except Exception as e:
    warnings.warn(f"Skipping local agent '{agent_key}' due to import error: {e}")


# Eagerly import local agents so they self-register with `grl.agents.register_agent`
_safe_import(
    lambda: __import__(
        "grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent",
        fromlist=["SokobanCodingAgent"],
    ),
    "sokobanCodingAgent",
)


def get_agent_cls(name: str) -> Type:
  """
  Resolve an agent class by name, ensuring local agents are imported first.

  Delegates to `grl.agents.get_agent_cls` after triggering local imports.
  """
  # Ensure local modules are imported (no-op if already imported)
  _safe_import(
      lambda: __import__(
          "grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent",
          fromlist=["SokobanCodingAgent"],
      ),
      "sokobanCodingAgent",
  )
  # Prefer direct import of local modules first so tests can stub them
  fallback_map = {
      "sokobanCodingAgent": (
          "grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent",
          "SokobanCodingAgent",
      ),
  }
  if name in fallback_map:
    module_path, class_name = fallback_map[name]
    try:
      # If tests have stubbed the module in sys.modules, this returns the stub
      mod = sys.modules.get(module_path) or importlib.import_module(module_path)
      if hasattr(mod, class_name):
        return getattr(mod, class_name)
    except Exception:
      pass
  # Fallback to global registry
  from grl.agents import get_agent_cls as _core_get
  try:
    return _core_get(name)
  except KeyError:
    # Last resort: attempt direct import again to raise a clearer error
    if name in fallback_map:
      module_path, class_name = fallback_map[name]
      try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
      except Exception as e:
        raise KeyError(f"Agent '{name}' not found and fallback import failed: {e}")
    raise
