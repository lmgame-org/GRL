"""AMC 2023 environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class AMC23Env(BaseMathEnv):
  """AMC 2023 math competition environment."""

  DEFAULT_DATASET_PATH = "math-ai/amc23"
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "question"
  ANSWER_FIELD = "answer"
