"""GSM8K environment."""

from grl.agents.mathAgent.envs.base_math_env import BaseMathEnv


class GSM8KEnv(BaseMathEnv):
  """GSM8K math environment."""

  DEFAULT_DATASET_PATH = "openai/gsm8k"
  DEFAULT_DATASET_CONFIG = "main"
  DEFAULT_SPLIT = "train"
  QUESTION_FIELD = "question"
  ANSWER_FIELD = "answer"

  def extract_answer(self, answer):
    if "####" in answer:
      answer = answer.split("####")[-1].strip()
    else:
      answer = answer.strip()

    for remove_char in [",", "$", "%", "g"]:
      answer = answer.replace(remove_char, "")

    try:
      return int(answer)
    except ValueError:
      return answer