import os
import json
import types
import numpy as np
import torch


class DummyTokenizer:
  def __init__(self):
    # Simple fake vocab for special tokens and basic words
    self.pad_token_id = 0
    self.eos_token_id = 2

  def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
    # Very simple concatenation of message contents
    parts = []
    for m in messages:
      if isinstance(m, dict):
        parts.append(str(m.get("content", "")))
      else:
        parts.append(str(m))
    text = "\n".join(parts)
    return text

  def __call__(self, texts, return_tensors=None, padding=True, padding_side="left", truncation=False):
    # Super-minimal tokenizer: map each char to id>0 for non-empty text
    if isinstance(texts, str):
      texts = [texts]
    max_len = max(len(t) for t in texts) if texts else 1
    input_ids = []
    attn = []
    for t in texts:
      ids = [min(255, ord(c)) for c in t]
      pad = [self.pad_token_id] * (max_len - len(ids))
      if padding_side == "left":
        arr = pad + ids
        mask = [0] * len(pad) + [1] * len(ids)
      else:
        arr = ids + pad
        mask = [1] * len(ids) + [0] * len(pad)
      input_ids.append(arr)
      attn.append(mask)
    class R:
      pass
    r = R()
    r.input_ids = torch.tensor(input_ids, dtype=torch.long)
    r.attention_mask = torch.tensor(attn, dtype=torch.long)
    return r

  def batch_decode(self, batch_ids, skip_special_tokens=True):
    # Reverse dummy: map integers back to chars if in printable range
    outs = []
    for row in batch_ids:
      text = "".join(chr(int(x)) if int(x) >= 32 else " " for x in row)
      outs.append(text.strip())
    return outs

  def encode(self, s):
    # Used for special token lookups in masks; return non-zero ids for markers
    return [1]


class DummyDataProto:
  def __init__(self, batch):
    self.batch = batch
    self.meta_info = {}
    self.non_tensor_batch = {}

  @staticmethod
  def from_single_dict(d):
    return DummyDataProto(d)


class DummyActorWG:
  def __init__(self):
    self.world_size = 1

  def generate_sequences(self, dp):
    # Expect dp.batch["input_ids"], produce dp-like with "responses"
    input_ids = dp.batch["input_ids"]
    bsz = input_ids.shape[0]
    # produce a simple response that includes an answer with a valid action string
    # format: "<answer>Right || Right</answer>"
    resp_text = "<answer>Right || Right</answer>"
    responses = []
    for _ in range(bsz):
      responses.append([ord(c) for c in resp_text])
    dp_out = DummyDataProto({"responses": torch.tensor(responses, dtype=torch.long)})
    return dp_out


class StubEnvOut:
  def __init__(self, state="", truncated=False, terminated=False, reward=0.0, info=None):
    self.state = state
    self.truncated = truncated
    self.terminated = terminated
    self.reward = reward
    self.info = info or {}


class StubAgent:
  def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
    self.group_id = group_id
    self.agent_id = agent_id
    self.tag = tag or "stub"
    self.agent_config = config.get("agent_config", {})
    self.env_config = config.get("env_config", {})
    self.max_turns = int(self.agent_config.get("max_turns", 1))
    self.action_separator = self.agent_config.get("action_separator", "||")
    self.messages = [
        {"role": "system", "content": self.agent_config.get("system_prompt", "sys")},
        {"role": "user", "content": self.agent_config.get("prompt", "user")},
    ]
    self.history = []

  def reset(self, seed=None):
    # Reinitialize messages for a fresh turn
    self.messages = [
        {"role": "system", "content": self.agent_config.get("system_prompt", "sys")},
        {"role": "user", "content": self.agent_config.get("prompt", "user")},
    ]
    return StubEnvOut(state="Sokoban(state)", truncated=False, terminated=False, reward=0.0, info={})

  def get_messages(self):
    return self.messages

  def execute_tool_call(self, llm_response: str):
    # No actual tools in this stub; signal no finish
    return False, None

  def get_env_outputs(self, llm_response: str):
    # Treat llm_response as final assistant content
    self.messages.append({"role": "assistant", "content": llm_response})
    # Track simple history row
    self.history.append({"state": "S", "actions_left": 0, "actions": [3, 3], "reward": 1.0, "info": {}, "llm_response": llm_response, "llm_raw_response": llm_response})
    return StubEnvOut(state="Sokoban(next)", truncated=True, terminated=True, reward=1.0, info={})

  def get_final_rollout_states(self):
    return {
        "agent_id": self.agent_id,
        "group_id": self.group_id,
        "tag": self.tag,
        "history": self.history,
        "metrics": {f"{self.tag}/success": 1.0},
        "penalty": 0.0,
    }


def _build_min_cfg():
  class Cfg(dict):
    def __getattr__(self, k):
      return self.get(k)
    def __setattr__(self, k, v):
      self[k] = v

  cfg = Cfg()
  cfg.max_prompt_length = 2048

  class Roll:
    pass
  roll = Roll()
  roll.truncation = False
  roll.show_tqdm = False
  roll.num_prompt_threads = 0
  roll.num_env_threads = 0
  roll.num_init_threads = 0
  roll.agent_group_num = [1]
  roll.agent_group_size = [2]
  # Ensure rollout looks up the correct agent name
  roll.training = ["sokobanCodingAgent"]

  class RN:
    pass
  rn = RN()
  rn.grouping = "batch"
  rn.method = "identity"
  roll.reward_normalization = rn
  roll.use_turn_scores = False
  cfg.rollout = roll

  # agent section (dict under key; accessed via cfg[agent_name])
  cfg.training = ["sokobanCodingAgent"]
  cfg["sokobanCodingAgent"] = {
      "agent_type": "sokobanAgent",
      "agent_config": {
          "tool_use": True,
          "workspace_path": os.path.join(os.path.dirname(__file__), "..", "..", "workspace"),
          "enable_think": False,
          "max_tokens": 256,
          "max_turns": 1,
          "max_actions_per_turn": 10,
          "max_actions_all_turns": 10,
          "max_steps": 2,
          "format_penalty": -0.1,
          "action_separator": "||",
      },
      "env_config": {
          "dim_room": [5, 5],
          "num_boxes": 1,
          "max_steps": 50,
          "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
          "grid_vocab": {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"},
          "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
          "render_mode": "text",
      },
  }
  return cfg


def test_torch_sync_rollout_end_to_end(monkeypatch, tmp_path):
  # Monkeypatch verl.DataProto used by rollout into our dummy
  import grl.rollout.torch_sync_rollout as mod

  # Patch inside module scope
  monkeypatch.setitem(mod.__dict__, "DataProto", DummyDataProto)
  # Patch AgentGroupBuilder.make_agents to produce stub agents
  import grl_agents.agent_group_builder as agb
  async def _stub_make_agents(self):
    agents = []
    for idx in range(self.group_num):
      # Use rollout agent name to align metrics aggregation keys
      agents.append(StubAgent(config=self.config, group_id=0, agent_id=idx, seed=self.seed, tag="sokobanCodingAgent"))
    return agents
  monkeypatch.setattr(agb.AgentGroupBuilder, "make_agents", _stub_make_agents, raising=True)

  # Build rollout
  from grl.rollout.torch_sync_rollout import TorchSyncRollout
  cfg = _build_min_cfg()
  tokenizer = DummyTokenizer()
  actor_wg = DummyActorWG()

  rollout = TorchSyncRollout(actor_wg, cfg, tokenizer, validation=False)

  # Smoke test: get_batch_llm_prompts / env outputs requires agents; run rollout()
  batch = rollout.rollout()

  # Validate RolloutBatch structure
  assert hasattr(batch, "input_ids")
  assert hasattr(batch, "loss_mask")
  assert hasattr(batch, "reward_scores")
  assert hasattr(batch, "agent_raw_data")
  assert hasattr(batch, "meta_info")

  # Shapes
  assert isinstance(batch.input_ids, np.ndarray)
  assert isinstance(batch.loss_mask, np.ndarray)
  assert isinstance(batch.reward_scores, np.ndarray)
  assert isinstance(batch.agent_raw_data, dict)

  # Non-empty
  assert batch.input_ids.shape[0] > 0
  assert batch.loss_mask.shape[0] == batch.input_ids.shape[0]
  assert batch.reward_scores.shape[0] == batch.input_ids.shape[0]

  # meta metrics include response_length
  assert isinstance(batch.meta_info, dict)
  assert "metrics" in batch.meta_info
  assert "response_length" in batch.meta_info["metrics"]

  # Ensure agent_raw_data arrays line up
  n = batch.input_ids.shape[0]
  for k in ["agent_ids", "group_ids", "messages_list"]:
    assert k in batch.agent_raw_data
    assert len(batch.agent_raw_data[k]) == n

  # Test masks and scores compatibility
  import grl.rollout.torch_sync_rollout as rmod
  loss_mask, score_tensor, response_mask = rollout.get_masks_and_scores(
      torch.tensor(batch.input_ids, dtype=torch.long),
      all_scores=[[1.0] for _ in range(n)],
      use_turn_scores=False,
  )
  assert loss_mask.shape[0] == n
  assert score_tensor.shape[0] == n
  assert response_mask.shape[0] == n