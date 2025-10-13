import sys
import types
import importlib


def _stub_sokoban_agent_module():
  """Inject a lightweight SokobanCodingAgent stub before importing the builder."""
  module_name = (
      "grl_agents.puzzle_agents.sokoban_coding_agent.sokoban_coding_agent"
  )
  stub_module = types.ModuleType(module_name)

  class MockAgent:
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
      self.config = config
      self.group_id = group_id
      self.agent_id = agent_id
      self.seed = seed
      self.tag = tag
    def get_final_rollout_states(self):
      return {
          "agent_id": self.agent_id,
          "group_id": self.group_id,
          "tag": self.tag,
          "history": [],
          "metrics": {f"mock/{self.agent_id}": 1.0},
          "penalty": 0.0,
      }
    async def aget_final_rollout_states(self):
      return self.get_final_rollout_states()

  stub_module.SokobanCodingAgent = MockAgent
  sys.modules[module_name] = stub_module
  # Ensure a clean import of the target module(s)
  for name in ["grl_agents.agent_group_builder"]:
    if name in sys.modules:
      del sys.modules[name]

  return MockAgent


def _import_builder_with_stub():
  MockAgent = _stub_sokoban_agent_module()
  agb = importlib.import_module("grl_agents.agent_group_builder")
  return agb, MockAgent


def _run_async(coro):
  import asyncio

  loop = asyncio.get_event_loop()
  return loop.run_until_complete(coro)


def test_make_agents_simple():
  agb, MockAgent = _import_builder_with_stub()

  seed = 10
  group_num = 3
  builder = agb.AgentGroupBuilder(
      seed=seed,
      group_num=group_num,
      config={"foo": "bar"},
      agent_cls=MockAgent,
      agent_name="mock",
  )

  agents = _run_async(builder.make_agents())

  assert len(agents) == group_num
  for i, agent in enumerate(agents):
    assert isinstance(agent, MockAgent)
    assert agent.group_id == 0
    assert agent.agent_id == i
    # All agents should share the same seed and tag in the group
    assert agent.seed == seed
    assert agent.tag == f"mock-{seed}"
    assert agent.config == {"foo": "bar"}


def test_generate_full_trajectories_no_reset():
  agb, MockAgent = _import_builder_with_stub()

  seed = 1
  group_num = 4
  builder = agb.AgentGroupBuilder(
      seed=seed,
      group_num=group_num,
      config={"alpha": 1},
      agent_cls=MockAgent,
      agent_name="mock",
  )
  agents = _run_async(builder.make_agents())
  out = _run_async(builder.generate_full_trajectories(agents=agents))
  assert isinstance(out, list)
  assert len(out) == group_num
  # Shared seed reflected in tags
  assert all(row["tag"].endswith(f"mock-{seed}") for row in out)

