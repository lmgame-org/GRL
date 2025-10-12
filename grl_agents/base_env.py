from __future__ import annotations

from typing import Any, Dict, Tuple
import asyncio


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

  # ─────────────────── Optional async wrappers ───────────────────
  async def areset(self, seed: int | None = None, **kwargs) -> Any:
    """Async wrapper around reset."""
    return await asyncio.to_thread(self.reset, seed, **kwargs)

  async def astep(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Async wrapper around step."""
    return await asyncio.to_thread(self.step, action)

  async def arender(self, mode: str = "text") -> Any:
    """Async wrapper around render."""
    return await asyncio.to_thread(self.render, mode)

  async def aclose(self) -> None:
    """Async wrapper around close."""
    await asyncio.to_thread(self.close)
