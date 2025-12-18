"""Minerva Math environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class MinervamathEnv(BaseMathEnv):
  """Minerva Math benchmark environment."""

  DEFAULT_DATASET_PATH = "math-ai/minervamath"
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "question"
  ANSWER_FIELD = "answer"
