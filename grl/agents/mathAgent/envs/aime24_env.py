"""AIME 2024 environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class AIME24Env(BaseMathEnv):
  """AIME 2024 math competition environment."""

  DEFAULT_DATASET_PATH = "math-ai/aime24"
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "problem"
  ANSWER_FIELD = "solution"
