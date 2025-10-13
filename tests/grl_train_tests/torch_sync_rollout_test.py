import types
import numpy as np
import pytest


class _MockRolloutCfg:
  def __init__(self,
               training,
               validation,
               agent_group_num,
               agent_group_size,
               validation_agent_group_num=None,
               validation_agent_group_size=None,
               validation_seed=123):
    # Mimic configs/base.yaml (subset)
    self.training = training
    self.validation = validation
    self.agent_group_num = agent_group_num
    self.agent_group_size = agent_group_size
    self.validation_agent_group_num = validation_agent_group_num or agent_group_num
    self.validation_agent_group_size = validation_agent_group_size or agent_group_size
    self.validation_seed = validation_seed


class _MockCfg:
  def __init__(self, rollout_cfg, agents_map):
    # Attribute access for rollout, mapping access for agent configs
    self.rollout = rollout_cfg
    self._agents_map = dict(agents_map)

  def __getitem__(self, key):
    return self._agents_map[key]


def _build_mock_cfg():
  # Reference grl_agents/puzzle_agents/sokoban_coding_agent/config.py
  from grl_agents.puzzle_agents.sokoban_coding_agent.config import (
    get_sokoban_coding_agent_config,
  )

  training_agents = ["simpleSokobanAgent"]
  validation_agents = ["simpleSokobanAgent", "largeSokobanAgent"]

  rollout_cfg = _MockRolloutCfg(
    training=training_agents,
    validation=validation_agents,
    agent_group_num=[2],
    agent_group_size=[1],
    validation_agent_group_num=[2, 2],
    validation_agent_group_size=[1, 1],
  )

  # Map both names to a valid sokoban coding agent config for simplicity
  sokoban_conf = get_sokoban_coding_agent_config()
  agents_map = {
    "simpleSokobanAgent": sokoban_conf,
    "largeSokobanAgent": sokoban_conf,
  }
  return _MockCfg(rollout_cfg, agents_map)


def test_setup_agent_config_normalizes_and_limits():
  from grl.rollout.torch_sync_rollout import TorchSyncRollout

  cfg = _build_mock_cfg()

  # actor_wg and tokenizer are not used in setup; pass dummies
  rollout = TorchSyncRollout(actor_rollout_wg=object(), cfg=cfg, tokenizer=object(), validation=False)
  # setup is already called in __init__, but call again to assert idempotence
  rollout._setup_agent_config()

  assert rollout.agent_names == ["simpleSokobanAgent"]
  assert len(rollout.agent_config_list) == 1

  # From get_sokoban_coding_agent_config(): max_turns=1, max_steps=10
  assert rollout.max_turns == 1
  assert rollout.max_steps == 10


def test_init_batch_agents_builds_expected_counts(monkeypatch):
  import grl.rollout.torch_sync_rollout as tsr
  from grl.rollout.torch_sync_rollout import TorchSyncRollout

  cfg = _build_mock_cfg()

  # Fake builder/agent to avoid async complexity
  class FakeAgent:
    pass

  class FakeBuilder:
    def __init__(self, group_size):
      self.group_size = int(group_size)

    async def make_agents(self):
      # Return exactly group_size agents
      return [FakeAgent() for _ in range(self.group_size)]

  class FakeRLDataset:
    def __init__(self, base_configs, seeds, group_nums, group_sizes, agent_names=None):
      # Simulate one builder per group, each yielding group_size agents
      self._builders = []
      for i, num_groups in enumerate(group_nums):
        for _ in range(int(num_groups)):
          self._builders.append(FakeBuilder(group_sizes[i]))

    def get_batch(self, index=None, agent_name=None, group_num=None):
      return list(self._builders)

  # Monkeypatch the dataset used by the rollout module
  monkeypatch.setattr(tsr, "RLDataset", FakeRLDataset)

  rollout = TorchSyncRollout(actor_rollout_wg=object(), cfg=cfg, tokenizer=object(), validation=False)
  # setup already ran; now initialize agents without reset
  rollout._init_batch_agents()

  # With agent_group_num=[2], agent_group_size=[1] and one agent type -> 2 agents total
  assert rollout.total_agent_num == 2
  assert len(rollout.agents) == 2

  # done_mask and env_outs as set by _init_batch_agents
  assert isinstance(rollout.done_mask, np.ndarray)
  assert rollout.done_mask.dtype == bool
  assert rollout.done_mask.shape == (2,)
  assert rollout.env_outs is None



def test_rollout_edge_cases_four_agents(monkeypatch):
  import grl.rollout.torch_sync_rollout as tsr
  from grl.rollout.torch_sync_rollout import TorchSyncRollout

  cfg = _build_mock_cfg()

  # Simple tokenizer stub
  class Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
      return " | ".join([str(m.get("content", "")) for m in messages])

    def batch_decode(self, responses, skip_special_tokens=True):
      return list(responses)

  tokenizer = Tok()

  # Fake EnvOutput
  class EOut:
    def __init__(self, truncated=False, terminated=False, state="s", reward=0.0):
      self.truncated = truncated
      self.terminated = terminated
      self.state = state
      self.reward = reward

  # Fake Agent to simulate edge cases
  class FA:
    def __init__(self, agent_id, group_id, behavior, max_steps):
      self.agent_id = agent_id
      self.group_id = group_id
      self.behavior = behavior
      self.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
      self.agent_config = {"max_steps": max_steps, "max_turns": 1, "enable_think": False, "use_think_answer_token": False}
      self._tool_msgs = []

    def get_llm_prompts(self, env_out):
      return self.messages

    def get_messages(self):
      return self.messages

    def get_tool_llm_prompts(self):
      return self.get_messages()

    def execute_tool_call(self, reply):
      if self.behavior == "invalid":
        # Append feedback; not done
        self.messages.append({"role": "user", "content": "Invalid tool-call format. Please include <function=...>...</function>."})
        return False, None
      if self.behavior == "finish":
        return True, EOut(truncated=True, terminated=True, state="done", reward=1.0)
      if self.behavior == "no_finish":
        return False, None
      if self.behavior == "budget":
        return False, None
      return False, None

    def get_env_outputs(self, reply):
      return EOut(truncated=True, terminated=True, state="stepped", reward=0.5)

    def reset(self, seed=None):
      return EOut(truncated=False, terminated=False, state="init", reward=0.0)

    def get_final_rollout_states(self):
      return {"agent_id": self.agent_id, "group_id": self.group_id, "metrics": {"dummy/success": 1.0}}

  # Patch reset to create 4 agents in 2 groups of size 2
  def _fake_reset(self, seed=None):
    self.agents = [
      FA(0, 0, "invalid", 1),
      FA(1, 0, "finish", 1),
      FA(2, 1, "no_finish", 1),
      FA(3, 1, "budget", 0),
    ]
    import numpy as np
    self.done_mask = np.zeros(4, dtype=bool)
    self.env_outs = [EOut() for _ in range(4)]

  monkeypatch.setattr(tsr.TorchSyncRollout, "_reset_batch_agents", _fake_reset)

  # Pre-programmed LLM responses: first env phase, then tool phase
  calls = {"i": 0}
  env_responses = ["", "", "", ""]
  tool_responses = [
    "nonsense",  # invalid
    "<function=finish><parameter=result>move</parameter></function>",  # finish
    "<answer>Up||Left</answer>",  # no finish
    "tool"  # ignored due to budget=0
  ]

  class Out:
    def __init__(self, responses):
      self.batch = {"responses": responses}

  def _fake_generate(self, prompts):
    if calls["i"] == 0:
      calls["i"] += 1
      return Out(env_responses)
    else:
      return Out(tool_responses[: len(prompts)])

  monkeypatch.setattr(tsr.TorchSyncRollout, "generate_sequences", _fake_generate)

  # Simplify final batch building to avoid tokenizer plumbing
  def _fake_build(self, states):
    return {"n_agents": len(self.agents), "done_mask": self.done_mask.copy(), "states": states}

  monkeypatch.setattr(tsr.TorchSyncRollout, "build_rollout_batch", _fake_build)

  rollout = TorchSyncRollout(actor_rollout_wg=object(), cfg=cfg, tokenizer=tokenizer, validation=False)

  result = rollout.rollout()

  assert result["n_agents"] == 4
  # Agent 1 done via finish, Agent 3 done via budget; others not forced done in inner loop
  assert result["done_mask"].tolist() == [False, True, False, True]
  # Group layout 0,0,1,1
  assert [a.group_id for a in rollout.agents] == [0, 0, 1, 1]
  # Invalid reply feedback recorded
  assert any("Invalid tool-call format" in m.get("content", "") for m in rollout.agents[0].messages)


def test_reset_batch_agents_initializes_messages_and_prompts(monkeypatch):
  import grl.rollout.torch_sync_rollout as tsr
  from grl.rollout.torch_sync_rollout import TorchSyncRollout

  cfg = _build_mock_cfg()

  # Simple tokenizer stub to build prompts
  class Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
      return " || ".join([str(m.get("role", "")) + ":" + str(m.get("content", "")) for m in messages])

  tokenizer = Tok()

  # Minimal EnvOutput replacement
  class EOut:
    def __init__(self, state="init", truncated=False, terminated=False, reward=0.0):
      self.state = state
      self.truncated = truncated
      self.terminated = terminated
      self.reward = reward

  # Fake agent that sets initial messages in reset
  class FA:
    def __init__(self, idx, group):
      self.agent_id = idx
      self.group_id = group
      self.agent_config = {"max_steps": 1, "max_turns": 1}
      self.messages = []

    def reset(self, seed=None):
      self.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"init_state_seed={seed}"},
      ]
      return EOut(state=f"state_{seed}")

    def get_messages(self):
      return self.messages

    def get_tool_llm_prompts(self):
      return self.get_messages()

  class FakeBuilder:
    def __init__(self, group_id, group_size):
      self.group_id = group_id
      self.group_size = int(group_size)

    async def make_agents(self):
      return [FA(idx=i, group=self.group_id) for i in range(self.group_size)]

  class FakeRLDataset:
    def __init__(self, base_configs, seeds, group_nums, group_sizes, agent_names=None):
      self._builders = []
      gid = 0
      for i, num_groups in enumerate(group_nums):
        for _ in range(int(num_groups)):
          self._builders.append(FakeBuilder(group_id=gid, group_size=group_sizes[i]))
          gid += 1

    def get_batch(self, index=None, agent_name=None, group_num=None):
      return list(self._builders)

  monkeypatch.setattr(tsr, "RLDataset", FakeRLDataset)

  rollout = TorchSyncRollout(actor_rollout_wg=object(), cfg=cfg, tokenizer=tokenizer, validation=False)
  rollout._reset_batch_agents(seed=42)

  # Agents and env_outs created
  assert len(rollout.agents) == rollout.total_agent_num
  assert rollout.env_outs is not None
  assert len(rollout.env_outs) == rollout.total_agent_num

  # Each agent has initial messages (system + user)
  for a in rollout.agents:
    msgs = a.get_messages()
    assert isinstance(msgs, list) and len(msgs) >= 2
    assert msgs[0].get("role") == "system"
    assert msgs[1].get("role") == "user"

  # get_batch_tool_llm_prompts produces non-empty prompts from initial messages
  active_indices = list(range(len(rollout.agents)))
  tool_prompts = rollout.get_batch_tool_llm_prompts(active_indices)
  assert len(tool_prompts) == len(active_indices)
  assert all(isinstance(p, str) and len(p) > 0 for p in tool_prompts)

