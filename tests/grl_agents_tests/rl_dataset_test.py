import sys
import types
import importlib


def _stub_sokoban_agent_module():
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
  for name in ["grl_agents.agent_group_builder", "grl_agents.rl_dataset"]:
    if name in sys.modules:
      del sys.modules[name]

  return MockAgent


def test_rl_dataset_get_batch_shape_and_seeds():
  _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  base_configs = [{"x": 1}, {"x": 2}, {"x": 3}]
  seeds = [100, 200, 300]
  dataset = rl_dataset.RLDataset(base_configs=base_configs, seeds=seeds)

  # Get batch for a single index
  builders = dataset.get_batch(index=1, agent_name="mock", group_num=2)
  assert len(builders) == 1
  b = builders[0]
  assert b.seed == 200
  assert b.group_num == 2


def test_rl_dataset_builders_create_mock_agents_simple():
  MockAgent = _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  dataset = rl_dataset.RLDataset(base_configs=[{"k": "v"}], seeds=[10])
  builders = dataset.get_batch(index=0, group_num=3)

  # Build agents synchronously; ensure properties are wired through
  all_agents = []
  for builder in builders:
    agents = _run_async(builder.make_agents())
    assert len(agents) == 3
    for a_idx, agent in enumerate(agents):
      assert isinstance(agent, MockAgent)
      assert agent.group_id == 0
      assert agent.agent_id == a_idx
      assert agent.seed == 10 + a_idx
      assert agent.tag == f"sokobanCodingAgent-{agent.seed}"
      assert agent.config == {"k": "v"}
    all_agents.extend(agents)

  assert len(all_agents) == 3


def test_collect_group_trajectories_async():
  _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  dataset = rl_dataset.RLDataset(base_configs=[{"z": 3}, {"z": 4}], seeds=[5, 15])

  # Async call: returns list per group, each a list per agent
  results = _run_async(dataset.collect_group_trajectories(index=1, group_num=2))
  assert isinstance(results, list)
  assert len(results) == 1
  # Group has 2 agents' rollouts
  assert len(results[0]) == 2


def _run_async(coro):
  import asyncio

  loop = asyncio.get_event_loop()
  return loop.run_until_complete(coro)


