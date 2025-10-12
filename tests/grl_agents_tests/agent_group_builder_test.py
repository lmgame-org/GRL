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


def test_make_agents_non_parallel():
  agb, MockAgent = _import_builder_with_stub()

  seeds = [10, 11, 12]
  builder = agb.AgentGroupBuilder(
      seeds=seeds,
      config={"foo": "bar"},
      agent_cls=MockAgent,
      agent_name="mock",
  )

  agents = _run_async(builder.make_agents(parallel=False))

  assert len(agents) == len(seeds)
  for i, (seed, agent) in enumerate(zip(seeds, agents)):
    assert isinstance(agent, MockAgent)
    assert agent.group_id == 0
    assert agent.agent_id == i
    assert agent.seed == seed
    assert agent.tag == f"mock-{seed}"
    assert agent.config == {"foo": "bar"}


def test_make_agents_parallel_with_fake_executor():
  agb, MockAgent = _import_builder_with_stub()

  class FakePool:
    def map(self, func, iterable):
      # behave like built-in map but return list immediately
      return list(map(func, iterable))

  class FakeExecutor:
    def __init__(self, max_workers=None):
      self.max_workers = max_workers

    def __enter__(self):
      return FakePool()

    def __exit__(self, exc_type, exc, tb):
      return False

  # Monkeypatch the executor used inside the module
  agb.ProcessPoolExecutor = FakeExecutor  # type: ignore[attr-defined]

  seeds = [1, 2, 3, 4]
  builder = agb.AgentGroupBuilder(
      seeds=seeds,
      config={"alpha": 1},
      agent_cls=MockAgent,
      agent_name="mock",
  )

  agents = _run_async(builder.make_agents(parallel=True, max_workers=2))

  assert len(agents) == len(seeds)
  for i, (seed, agent) in enumerate(zip(seeds, agents)):
    assert isinstance(agent, MockAgent)
    assert agent.group_id == 0
    assert agent.agent_id == i
    assert agent.seed == seed
    assert agent.tag == f"mock-{seed}"
    assert agent.config == {"alpha": 1}


def test_generate_trajectories_concurrent_reset_and_collect():
  agb, MockAgent = _import_builder_with_stub()

  seeds = [101, 102, 103]
  builder = agb.AgentGroupBuilder(
      seeds=seeds,
      config={"cfg": 1},
      agent_cls=MockAgent,
      agent_name="mock",
  )

  agents = _run_async(builder.make_agents(parallel=False))

  # Collect without reset first
  out = _run_async(builder.generate_trajectories(agents=agents, reset=False, max_workers=2))
  assert isinstance(out, list)
  assert len(out) == len(agents)
  # Then with reset=True to ensure seeds are applied
  out2 = _run_async(builder.generate_trajectories(agents=agents, reset=True, max_workers=2))
  assert len(out2) == len(agents)
  for idx, agent in enumerate(agents):
    assert agent._reset_called is True
    assert agent.seed == seeds[idx]

