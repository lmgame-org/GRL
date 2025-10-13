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
  dataset = rl_dataset.RLDataset(
      base_configs=base_configs,
      seeds=seeds,
      group_nums=[1, 2, 1],
      group_sizes=[1, 2, 3],
  )

  # Get flat list of builders (replicated per index via group_nums)
  builders = dataset.get_batch()
  # Total builders = 1 + 2 + 1 = 4
  assert len(builders) == 4
  # Order: i=0 (seed=100), i=1 (seed=200), i=1 (seed=201), i=2 (seed=300)
  assert builders[0].seed == 100 and builders[0].group_num == 1 and builders[0].group_id == 0
  assert builders[1].seed == 200 and builders[1].group_num == 2 and builders[1].group_id == 1
  assert builders[2].seed == 201 and builders[2].group_num == 2 and builders[2].group_id == 2
  assert builders[3].seed == 300 and builders[3].group_num == 3 and builders[3].group_id == 3


def test_rl_dataset_builders_create_mock_agents_simple():
  MockAgent = _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  dataset = rl_dataset.RLDataset(
      base_configs=[{"k": "v"}],
      seeds=[10],
      group_nums=[1],
      group_sizes=[3],
  )
  builders = dataset.get_batch()

  # Build agents synchronously; ensure properties are wired through
  all_agents = []
  for builder in builders:
    agents = _run_async(builder.make_agents())
    assert len(agents) == 3
    for a_idx, agent in enumerate(agents):
      assert isinstance(agent, MockAgent)
      assert agent.group_id == builder.group_id
      assert agent.agent_id == a_idx
      # All agents in the group share the same seed and tag
      assert agent.seed == 10
      assert agent.tag == f"sokobanCodingAgent-10"
      assert agent.config == {"k": "v"}
    all_agents.extend(agents)

  assert len(all_agents) == 3


def test_collect_group_trajectories_async():
  _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  # Build dataset with two groups (seeds 15 and 16), each size 2
  dataset = rl_dataset.RLDataset(
      base_configs=[{"z": 4}],
      seeds=[15],
      group_nums=[2],
      group_sizes=[2],
  )

  # Async call: returns list per group (here 2), each a list per agent
  results = _run_async(dataset.collect_group_trajectories())
  assert isinstance(results, list)
  assert len(results) == 2
  # Each group has 2 agents' rollouts
  assert len(results[0]) == 2 and len(results[1]) == 2
  # Shared seeds reflected in tags per agent (15 then 16)
  assert all(row["tag"].endswith("sokobanCodingAgent-15") for row in results[0])
  assert all(row["tag"].endswith("sokobanCodingAgent-16") for row in results[1])


def _run_async(coro):
  import asyncio

  loop = asyncio.get_event_loop()
  return loop.run_until_complete(coro)


