from datasets import load_dataset
import re
import random
from lmgamerl.agents.agent_utils import all_seed
from lmgamerl.agents.base_env import BaseEnv

class GSM8KEnv(BaseEnv):
    def __init__(self, config, **kwargs):
        super(GSM8KEnv, self).__init__()
        
        self.config = config
        self.dataset = load_dataset(self.config.get('dataset_path', 'openai/gsm8k'), 'main', split=self.config.get('split', 'train'))
        # # Group problems by subject
        self.current_sample = None
        self.current_unique_id = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None

        # if not os.path.exists(os.path.dirname(self.log_file)):
        #     os.makedirs(os.path.dirname(self.log_file))

    def extract_answer(self, answer):
        if '####' in answer:
            answer = answer.split('####')[-1].strip()
        else:
            answer = answer.strip()

        for remove_char in [',', '$', '%', 'g']:
            answer = answer.replace(remove_char, '')

        try:
            return int(answer)
        except ValueError:
            return answer


    def reset(self,seed=None):
        # print("[DEBUG] seed ", seed)
        with all_seed(seed):
            question_data = random.choice(self.dataset)

        self.current_sample = question_data
        self.current_question = question_data['question']
        self.correct_solution = question_data['answer']
        self.correct_answer = self.extract_answer(question_data['answer'])
        self.render_cache = self.current_question
        self.step_num = 0
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        # reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        reward = 10.0 if is_correct else -0.1
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False
        self.step_num += 1
        info = {"action_is_effective": True, "action_is_valid": is_valid, "success": is_correct}
        self.render_cache = observation
        return self.render_cache, reward, done, info
    
    def _check_answer(self, user_answer):
        # print("[DEBUG] question:", self.current_question)
        # print("[DEBUG] correct_answer:", self.correct_answer)
        # print("[DEBUG] user_answer:", user_answer)
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        matches = re.findall(r'\d+', user_answer)
        if not matches:                         # no integer at all → invalid
            return False, False

        last_integer = int(matches[-1]) if matches else None
        # print("[DEBUG] user_answer_integer:", last_integer)

        is_correct = last_integer == self.correct_answer
        is_valid = last_integer != ""
        return is_correct, is_valid

    def render(self):
        return self.render_cache

    def close(self) -> None:
        self._question = self._answer = None
