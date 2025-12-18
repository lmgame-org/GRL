"""Base math environment with shared LaTeX parsing logic."""

from datasets import load_dataset
import random
from grl.agents.agent_utils import all_seed
from grl.agents.base_env import BaseEnv
from grl.agents.mathAgent.math_utils import strip_tex, answers_equal
import re

class BaseMathEnv(BaseEnv):
  """
  Base class for math environments with shared LaTeX/answer parsing utilities.
  Subclasses only need to specify dataset details and field mappings.
  """

  # Subclasses should override these
  DEFAULT_DATASET_PATH = None
  DEFAULT_DATASET_CONFIG = None
  DEFAULT_SPLIT = "test"
  QUESTION_FIELD = "problem"  # Field name for question in dataset
  ANSWER_FIELD = "answer"  # Field name for answer in dataset

  def __init__(self, config, **kwargs):
    super(BaseMathEnv, self).__init__()
    self.config = config
    self.dataset = load_dataset(
        self.config.get("dataset_path", self.DEFAULT_DATASET_PATH),
        self.config.get("dataset_config", self.DEFAULT_DATASET_CONFIG),
        split=self.config.get("split", self.DEFAULT_SPLIT),
    )
    self.current_sample = None
    self.current_unique_id = None
    self.current_question = None
    self.correct_answer = None
    self.step_num = None
    self.render_cache = None

  def extract_answer(self, answer):
    """Extract answer from dataset. Override for custom extraction logic."""
    return str(answer).strip()

  def reset(self, seed=None):
    with all_seed(seed):
      question_data = random.choice(self.dataset)
    self.current_sample = question_data
    self.current_question = question_data.get(self.QUESTION_FIELD, "")
    self.correct_solution = question_data.get(self.ANSWER_FIELD, "")
    self.correct_answer = self.extract_answer(self.correct_solution)
    self.render_cache = self.current_question
    self.step_num = 0
    return self.render_cache

  def step(self, action):
    is_correct, is_valid = self._check_answer(action)
    reward = 10.0 if is_correct else -0.1
    if is_correct:
      done = True
    else:
      done = False
    self.step_num += 1
    info = {
        "action_is_effective": True,
        "action_is_valid": is_valid,
        "success": is_correct,
    }
    return self.render_cache, reward, done, info

  def _check_answer(self, user_answer):
    """Check if user answer matches correct answer. Override for custom logic."""
    # print(f"user_answer: {user_answer}")
    # print(f"correct_answer: {self.correct_answer}")
    user_answer = "" if user_answer is None else str(user_answer)
    gold = self.correct_answer if self.correct_answer is not None else ""
    is_valid = len(strip_tex(user_answer)) > 0
    is_correct = answers_equal(gold, user_answer)
    return is_correct, is_valid

  def render(self):
    return self.render_cache

  def close(self) -> None:
    self._question = self._answer = None
