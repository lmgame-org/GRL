"""Unified Math Agent that works with any registered math environment."""

from grl.agents.agent_utils import SingleTurnTrajectory, EnvOutput
from grl.agents.base_agent import BaseAgent
from grl.agents.mathAgent.envs import get_math_env_cls
from grl.agents import register_agent


@register_agent("mathAgent")
class MathAgent(BaseAgent):
  """
  Unified Math Agent that can work with any registered math environment.
  
  The environment is specified via `env_name` in the config:
    - gsm8k: GSM8K dataset (integer answers)
    - aime24: AIME 2024 competition problems
    - aime25: AIME 2025 competition problems
    - amc23: AMC 2023 competition problems
    - math500: MATH-500 benchmark
    - minervamath: Minerva Math benchmark
  """

  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    super().__init__(config, group_id, agent_id, seed, tag)
    
    # Get environment name from config (required)
    self.env_name = config.get("env_name")
    if not self.env_name:
      raise ValueError(
          "MathAgent requires 'env_name' in config. "
          "Available: gsm8k, aime24, aime25, amc23, math500, minervamath"
      )
    
    self.initialize_env()
    
    # Setup custom prompt template if enabled
    self.turn_prompt_template = """Question: {state}\n"""

  def initialize_env(self):
    """Initialize the math environment based on env_name."""
    env_cls = get_math_env_cls(self.env_name)
    self.env = env_cls(self.env_config)

  def get_feedback_between_turns(self, reward: float) -> str:
    """Get textual feedback for the agent."""
    if reward > 0:
      return f"Feedback: Correct.\n"
    else:
      return f"Feedback: Incorrect. Please try again.\n"

  def get_env_outputs(self, llm_response: str):
    """Process LLM response and get environment outputs."""
    print(f"llm_response: {llm_response}")
    llm_raw_response = llm_response
    self.raw_response_list.append(llm_raw_response)
    self.cur_turn += 1

    processed_llm_response, actions = self.parse_llm_response(
        llm_raw_response, enable_think=self.enable_think
    )

    self.messages.append(
        {"role": "assistant", "content": processed_llm_response}
    )

    obs = self.env.render()
    total_reward = 0
    done = False
    info = {}

    if len(actions) != 0:
      obs, reward, done, step_info = self.env.step(actions[-1])
      total_reward += reward
      info.update(step_info)
    else:
      self.penalty += self.format_penalty

    self.total_actions_consumed += len(actions)
    actions_left = max(
        0, self.max_actions_all_turns - self.total_actions_consumed
    )

    if (
        self.cur_turn >= self.max_turns
        or self.total_actions_consumed >= self.max_actions_all_turns
    ):
      done = True

    self.trajectory_history.add(
        SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions=actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response,
        )
    )

    return EnvOutput(
        truncated=done,
        terminated=done,
        state=obs,
        reward=total_reward,
        info=info,
    )
