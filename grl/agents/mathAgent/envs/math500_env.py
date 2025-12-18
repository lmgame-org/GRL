"""MATH-500 environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class Math500Env(BaseMathEnv):
  """MATH-500 benchmark environment."""

  DEFAULT_DATASET_PATH = "HuggingFaceH4/MATH-500"
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "problem"
  ANSWER_FIELD = "answer"
