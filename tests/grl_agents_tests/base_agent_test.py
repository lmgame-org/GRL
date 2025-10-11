import pytest

from grl_agents.base_agent import BaseAgent
from grl_agents.utils import SingleTurnTrajectory, EnvOutput


class FakeEnv:
  def __init__(self, env_config):
    self.env_config = env_config
    self.last_seed = None
    self.closed = False

  def reset(self, seed=None):
    self.last_seed = seed
    return "obs"

  def render(self):
    return "rendered"

  def step(self, action):
    # next_state, reward, done, info
    return "obs", 0.0, False, {"action": action}

  def close(self):
    self.closed = True


class DummyAgent(BaseAgent):
  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    super().__init__(config, group_id, agent_id, seed, tag)
    self.initialize_env()

  def initialize_env(self) -> None:
    self.env = FakeEnv(self.env_config)

  def get_env_outputs(self, llm_response: str) -> EnvOutput:
    # Minimal environment loop: process and append a message, no-op reward
    llm_raw_response = llm_response
    self.raw_response_list.append(llm_raw_response)
    self.cur_turn += 1
    processed_llm_response, _ = self.parse_llm_response(
        str(llm_raw_response), enable_think=self.enable_think
    )
    self.messages.append({"role": "assistant", "content": processed_llm_response})
    obs = self.env.render()
    return EnvOutput(
        truncated=False,
        terminated=False,
        state=obs,
        reward=0.0,
        info={},
    )


def make_config(overrides: dict | None = None):
  cfg = {
      "agent_config": {
          "max_turns": 3,
          "max_actions_all_turns": 4,
          "max_actions_per_turn": 3,
          "max_tokens": 128,
          "format_penalty": -0.1,
          "enable_think": True,
          "system_prompt": "You are a helpful AI assistant.",
          "prompt": "Please respond appropriately.",
          "action_separator": "||",
          "use_think_answer_token": True,
      },
      "env_config": {},
  }
  if overrides:
    # shallow merge for test convenience
    for k, v in overrides.items():
      if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
        cfg[k].update(v)
      else:
        cfg[k] = v
  return cfg


def test_init_and_reset():
  cfg = make_config()
  agent = DummyAgent(cfg, group_id=7, agent_id=42, seed=123, tag="TestAgent")
  # Before reset
  assert agent.group_id == 7
  assert agent.agent_id == 42
  assert agent.tag == "TestAgent"
  assert agent.cur_turn == 0
  assert len(agent.messages) == 2

  # Reset with fixed seed
  out = agent.reset(seed=999)
  assert isinstance(out, EnvOutput)
  assert out.truncated is False and out.terminated is False
  assert out.reward == 0.0
  assert agent.cur_turn == 0
  assert len(agent.messages) == 2  # reinitialized
  assert agent.total_actions_consumed == 0
  assert agent.penalty == 0.0


def test_get_llm_prompts_first_turn_merges():
  cfg = make_config()
  agent = DummyAgent(cfg)
  env_out = EnvOutput(truncated=False, terminated=False, state="S", reward=0.0, info={})
  msgs = agent.get_llm_prompts(env_out)
  assert len(msgs) == 2  # merged into initial user message
  content = msgs[1]["content"]
  assert "Turn 1:" in content
  assert "State:" in content and "S" in content
  assert "actions remaining" in content
  assert str(agent.max_tokens) in content


def test_get_llm_prompts_subsequent_turn_appends_with_reward():
  cfg = make_config()
  agent = DummyAgent(cfg)
  # Simulate we're on turn 2
  agent.cur_turn = 1
  env_out = EnvOutput(truncated=False, terminated=False, state="S2", reward=1.5, info={})
  msgs = agent.get_llm_prompts(env_out)
  assert len(msgs) == 3
  last = msgs[-1]
  assert last["role"] == "user"
  assert last["content"].startswith("Reward:")
  assert "Turn 2:" in last["content"]


def test_parse_llm_response_enable_think_and_clamp_actions():
  cfg = make_config({"agent_config": {"max_actions_per_turn": 2}})
  agent = DummyAgent(cfg)
  processed, actions = agent.parse_llm_response("A || B || C", enable_think=True)
  assert processed.startswith("<think>") and "<answer>" in processed
  assert actions == ["A", "B"]  # clamped to 2


def test_parse_llm_response_disable_think():
  cfg = make_config({"agent_config": {"enable_think": False}})
  agent = DummyAgent(cfg)
  processed, actions = agent.parse_llm_response("foo||bar", enable_think=False)
  assert processed == "<answer>foo||bar</answer>"
  assert actions == ["foo", "bar"]


def test_history_and_metrics_aggregation():
  cfg = make_config()
  agent = DummyAgent(cfg, agent_id=5, group_id=2, tag="MetricAgent")

  # Build two trajectories with different flags
  agent.trajectory_history.add(
      SingleTurnTrajectory(
          state="s0",
          actions_left=3,
          actions=[1, 2],
          reward=1.0,
          info={"success": False, "action_is_effective": 1, "action_is_valid": 1},
          llm_response="<answer>x</answer>",
          llm_raw_response="x",
      )
  )
  agent.trajectory_history.add(
      SingleTurnTrajectory(
          state="s1",
          actions_left=2,
          actions=[3],
          reward=2.0,
          info={
              "success": True,
              "action_is_effective": 0,
              "action_is_valid": 1,
              "metrics": {"custom/metric": 7.0},
          },
          llm_response="<answer>y</answer>",
          llm_raw_response="y",
      )
  )

  out = agent.get_final_rollout_states()
  assert set(out.keys()) == {"agent_id", "history", "group_id", "tag", "metrics", "penalty"}
  assert out["agent_id"] == 5
  assert out["group_id"] == 2
  assert out["tag"] == "MetricAgent"
  assert isinstance(out["history"], list) and len(out["history"]) == 2

  metrics = out["metrics"]
  assert pytest.approx(metrics["MetricAgent/success"], rel=1e-6) == 1.0  # any True
  assert metrics["MetricAgent/num_actions"] == 3  # 2 + 1
  # action_is_effective average: (1 + 0) / 2
  assert pytest.approx(metrics["MetricAgent/action_is_effective"], rel=1e-6) == 0.5
  # action_is_valid average: (1 + 1) / 2
  assert pytest.approx(metrics["MetricAgent/action_is_valid"], rel=1e-6) == 1.0
  # merged custom metrics from last traj
  assert metrics["custom/metric"] == 7.0


def test_messages_accessor_and_close():
  cfg = make_config()
  agent = DummyAgent(cfg)
  msgs = agent.get_messages()
  assert isinstance(msgs, list) and len(msgs) == 2

  # ensure close calls underlying env.close() if present
  env_ref = agent.env
  agent.close()
  assert getattr(env_ref, "closed", False) is True


