from typing import Tuple, List, Dict, Any, Optional
import re
from dataclasses import dataclass, field
import random
import numpy as np
from contextlib import contextmanager
from collections import deque


# ─────────────────── DATA STRUCTURES ───────────────────
@dataclass
class EnvOutput:
  """Simple container for environment outputs that SyncMultiTurnRollout expects."""

  truncated: bool = False
  terminated: bool = False
  state: str = ""
  reward: float = 0.0
  info: Dict[str, Any] = None  # type: ignore

  def __post_init__(self):
    if self.info is None:
      self.info = {}


@dataclass
class SingleTurnTrajectory:
  """Simple trajectory class for storing a single step's information."""

  state: str = ""
  actions_left: int = 0
  actions: List[int] = field(default_factory=list)
  reward: float = 0.0
  info: Dict[str, Any] = field(default_factory=dict)
  llm_response: str = ""
  llm_raw_response: str = ""


@dataclass
class MultiTurnTrajectory:
  """Trajectory class for storing multiple turns' information."""

  trajectories: deque = field(default_factory=lambda: deque(maxlen=5))
  max_length: int = 5

  def __post_init__(self):
    self.trajectories = deque(self.trajectories, maxlen=self.max_length)

  def add(self, trajectory: SingleTurnTrajectory) -> None:
    """Add a new trajectory to the deque."""
    self.trajectories.append(trajectory)

  def get(self) -> deque:
    """Get the trajectory at the specified index."""
    return self.trajectories

  def __len__(self) -> int:
    """Return the number of trajectories."""
    return len(self.trajectories)

  def clear(self) -> None:
    """Clear all trajectories."""
    self.trajectories.clear()


@dataclass
class SingleTurnToolCallTrajectory:
  """Container for logging one-turn tool-call interactions as chat messages.

  This mirrors the chat message format used by the rollout utilities and
  external providers, e.g.:
    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}

  Intended to capture LLM tool call blocks and corresponding tool feedbacks
  within a single episode/turn.
  """

  messages: List[Dict[str, Any]] = field(default_factory=list)

  def add(self, message: Dict[str, Any]) -> None:
    """Append a single chat message dict."""
    if isinstance(message, dict) and "role" in message and "content" in message:
      self.messages.append({"role": str(message["role"]), "content": str(message["content"])})

  def extend(self, messages: List[Dict[str, Any]]) -> None:
    """Append a list of chat message dicts."""
    for m in messages:
      self.add(m)

  def get(self) -> List[Dict[str, Any]]:
    """Return the list of captured messages."""
    return list(self.messages)

  def clear(self) -> None:
    """Clear all captured messages."""
    self.messages.clear()


@contextmanager
def all_seed(seed):
  """Context manager to set random seeds temporarily."""
  random_state = random.getstate()
  np_random_state = np.random.get_state()

  try:
    random.seed(seed)
    np.random.seed(seed)
    yield
  finally:
    random.setstate(random_state)
    np.random.set_state(np_random_state)


def debug_printout_in_env_output(messages, actions, tag):
  print("=" * 40)
  print("DEBUG: Messages and Actions")
  print("=" * 40)
  print(f"Agent Tag: {tag}")

  print("Messages:")
  for i, message in enumerate(messages):
    if isinstance(message, dict):
      role = message.get("role", "unknown")
      content = message.get("content", "")
      print(f"  [{i}] {role}: {repr(content)}")
    else:
      print(f"  [{i}] {repr(message)}")

  print("\nActions:")
  for i, action in enumerate(actions):
    print(f"  [{i}] {repr(action)}")

  print("=" * 40)
