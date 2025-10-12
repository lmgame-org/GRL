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
      self._reset_called = False

    def reset(self, seed=None):
      self._reset_called = True
      self.seed = seed
      return "obs"

    def get_final_rollout_states(self):
      return {
          "agent_id": self.agent_id,
          "group_id": self.group_id,
          "tag": self.tag,
          "history": [],
          "metrics": {f"mock/{self.agent_id}": 1.0},
          "penalty": 0.0,
      }

  stub_module.SokobanCodingAgent = MockAgent
  sys.modules[module_name] = stub_module
  for name in ["grl_agents.agent_group_builder", "grl_agents.rl_dataset"]:
    if name in sys.modules:
      del sys.modules[name]

  return MockAgent


def test_rl_dataset_get_batch_shape_and_seeds():
  _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  base_config = {"x": 1}
  dataset = rl_dataset.RLDataset(
      base_config=base_config,
      groups_per_batch=3,
      seeds_per_group=2,
      seed_start=100,
  )

  builders = dataset.get_batch(index=5)
  assert len(builders) == 3
  # For index=5, seeds should be [100 + 5*2 + i for i in range(2)] per group
  expected_seeds = [110, 111]
  for b in builders:
    assert [*b.seeds] == expected_seeds
    assert b.config is base_config


def test_rl_dataset_builders_create_mock_agents_non_parallel():
  MockAgent = _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  dataset = rl_dataset.RLDataset(
      base_config={"k": "v"}, groups_per_batch=2, seeds_per_group=3, seed_start=7
  )
  builders = dataset.get_batch(index=1)

  # Build agents synchronously; ensure properties are wired through
  all_agents = []
  for g_idx, builder in enumerate(builders):
    agents = _run_async(builder.make_agents(parallel=False))
    assert len(agents) == 3
    for a_idx, agent in enumerate(agents):
      assert isinstance(agent, MockAgent)
      assert agent.group_id == 0
      assert agent.agent_id == a_idx
      # seeds_per_group=3, seed_start=7, index=1 -> seeds [10,11,12]
      assert agent.seed == 10 + a_idx
      assert agent.tag == f"sokobanAgent-{agent.seed}"
      assert agent.config == {"k": "v"}
    all_agents.extend(agents)

  assert len(all_agents) == 6


def test_generate_group_trajectories_async():
  _stub_sokoban_agent_module()
  rl_dataset = importlib.import_module("grl_agents.rl_dataset")

  base_config = {"z": 3}
  dataset = rl_dataset.RLDataset(
      base_config=base_config,
      groups_per_batch=2,
      seeds_per_group=2,
      seed_start=5,
  )

  # Async call: returns list per group, each a list per agent
  results = _run_async(dataset.generate_group_trajectories(index=0, reset=True, max_workers=2))
  assert isinstance(results, list)
  assert len(results) == 2
  # Each group should have 2 agents' rollouts
  assert all(len(group_rows) == 2 for group_rows in results)


def _run_async(coro):
  import asyncio

  loop = asyncio.get_event_loop()
  return loop.run_until_complete(coro)


