"""Math environment registry."""

from typing import Dict, Type

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv

# Environment registry
MATH_ENV_REGISTRY: Dict[str, Type] = {}


def register_math_env(name: str):
  """Decorator to register a math environment."""

  def decorator(cls):
    if name in MATH_ENV_REGISTRY:
      raise ValueError(f"Math environment '{name}' already registered")
    MATH_ENV_REGISTRY[name] = cls
    return cls

  return decorator


def get_math_env_cls(name: str) -> Type:
  """Get math environment class by name."""
  if name not in MATH_ENV_REGISTRY:
    raise KeyError(
        f"Math environment '{name}' not found. Available: {list(MATH_ENV_REGISTRY.keys())}"
    )
  return MATH_ENV_REGISTRY[name]


def list_math_envs() -> list:
  """List all registered math environments."""
  return list(MATH_ENV_REGISTRY.keys())


# Import and register all math environments
from grl.agents.mathAgent.envs.gsm8k_env import GSM8KEnv
from grl.agents.mathAgent.envs.aime24_env import AIME24Env
from grl.agents.mathAgent.envs.aime25_env import AIME25Env
from grl.agents.mathAgent.envs.amc23_env import AMC23Env
from grl.agents.mathAgent.envs.math500_env import Math500Env
from grl.agents.mathAgent.envs.minervamath_env import MinervamathEnv

# Register environments
register_math_env("gsm8k")(GSM8KEnv)
register_math_env("aime24")(AIME24Env)
register_math_env("aime25")(AIME25Env)
register_math_env("amc23")(AMC23Env)
register_math_env("math500")(Math500Env)
register_math_env("minervamath")(MinervamathEnv)

__all__ = [
    "BaseMathEnv",
    "GSM8KEnv",
    "AIME24Env",
    "AIME25Env",
    "AMC23Env",
    "Math500Env",
    "MinervamathEnv",
    "MATH_ENV_REGISTRY",
    "register_math_env",
    "get_math_env_cls",
    "list_math_envs",
]
