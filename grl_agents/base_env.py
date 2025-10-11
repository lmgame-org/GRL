from __future__ import annotations

from typing import Any, Dict, Tuple


class BaseEnv:
  """
  Minimal reference implementation for an environment class.

  Every concrete environment should inherit from this and implement
  `reset`, `step`, `render`, and `close`.
  """

  def __init__(self, config: Dict[str, Any] | None = None, **kwargs):
    self.config = config or {}

  # ──────────────────────────────────────────────────────────
  # Required API (no-op stubs)
  # ──────────────────────────────────────────────────────────
  def reset(self, seed: int | None = None, **kwargs) -> Any:
    """Reset the environment to an initial state and return an observation."""
    raise NotImplementedError

  def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Advance the environment by one timestep using `action`.
    Returns: observation, reward, done, info
    """
    raise NotImplementedError

  def render(self, mode: str = "text") -> Any:
    """Return a human-readable representation of the current state."""
    raise NotImplementedError

  def close(self) -> None:
    """Clean up resources (files, sockets, etc.)."""
    pass


