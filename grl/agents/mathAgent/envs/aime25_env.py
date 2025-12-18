"""AIME 2025 environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class AIME25Env(BaseMathEnv):
  """AIME 2025 math competition environment."""

  DEFAULT_DATASET_PATH = "math-ai/aime25"
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "problem"
  ANSWER_FIELD = "answer"
