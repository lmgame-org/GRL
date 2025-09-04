#!/usr/bin/env python3
"""
Rollout tests for grl/rollout/tunix_sync_multi_turn_rollout.py

We follow the style of tests/rollout_tests/sync_multi_turn_rollout_test.py but
focus on unit-testing the core masking and reward alignment helpers in the
Tunix rollout manager without requiring full agent/env setup.

Tests included:
- test_get_masks_and_scores_alignment: validates loss_mask and score_tensor
  alignment to next-token targets (drop last of loss_mask, drop first of scores).
- test_normalize_score_tensor_batch_grouping: validates reward normalization
  over batch grouping and penalty application.
 - test_tunix_rollout_creation: end-to-end creation with mock rl_cluster/agent/tokenizer.
 - test_tunix_prompt_and_tokenization: validate prompt generation and tokenization outputs.
 - test_tunix_full_rollout_and_batch: run rollout and build rollout batch, verify shapes.
 - test_tunix_final_states_and_filtering: collect final states and test filtering helper.
"""

import numpy as np
import jax.numpy as jnp
import sys
import os
from pathlib import Path
from datetime import datetime

from grl.rollout.tunix_sync_multi_turn_rollout import SyncMultiTurnRollout
from grl.agents import REGISTERED_AGENTS


class _DummyTokenizer:
  """Minimal tokenizer stub with only methods used by the tested helpers."""

  def __init__(self, im_start_id: int = 101, im_end_id: int = 102):
    self._im_start_id = im_start_id
    self._im_end_id = im_end_id
    self.pad_token = "<|pad|>"
    self.pad_token_id = 0
    self.eos_token = "<|eos|>"
    self.eos_token_id = 2

  def encode(self, text: str):
    if text == "<|im_start|>":
      return [self._im_start_id]
    if text == "<|im_end|>":
      return [self._im_end_id]
    # generic mapping for other strings
    return [hash(text) % 10000]

  def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
    # Simple concatenation with special tokens; roles are ignored beyond content
    parts = []
    for msg in messages:
      content = msg.get("content", "")
      parts.append(f"<|im_start|>{content}<|im_end|>")
    if add_generation_prompt:
      parts.append("<|im_start|>")
    return "".join(parts)

  def __call__(self, texts, return_tensors=None, padding=True, padding_side="left", truncation=False):
    if isinstance(texts, str):
      texts = [texts]
    # Very simple tokenizer that maps special tokens and characters to ids
    def tokenize_text(t):
      ids = []
      i = 0
      while i < len(t):
        if t.startswith("<|im_start|>", i):
          ids.append(self._im_start_id)
          i += len("<|im_start|>")
          continue
        if t.startswith("<|im_end|>", i):
          ids.append(self._im_end_id)
          i += len("<|im_end|>")
          continue
        ch = t[i]
        ids.append(100 + (ord(ch) % 100))
        i += 1
      return ids
    token_lists = [tokenize_text(t) for t in texts]
    max_len = max(len(x) for x in token_lists) if token_lists else 0
    def left_pad(arr, L):
      pad = [self.pad_token_id] * (L - len(arr))
      return pad + arr
    padded = [left_pad(x, max_len) for x in token_lists]
    attn = [[0] * (max_len - len(x)) + [1] * len(x) for x in token_lists]
    return {
      "input_ids": padded,
      "attention_mask": attn,
    }

  def batch_decode(self, arr, skip_special_tokens=True):
    # Convert back to strings; special tokens become markers unless skipped
    result = []
    for row in np.array(arr):
      parts = []
      for tid in row.tolist():
        if tid == self.pad_token_id:
          continue
        if tid == self._im_start_id:
          if not skip_special_tokens:
            parts.append("<|im_start|>")
          continue
        if tid == self._im_end_id:
          if not skip_special_tokens:
            parts.append("<|im_end|>")
          continue
        parts.append(chr((tid - 100) % 100 + 27))  # arbitrary visible chars
      result.append("".join(parts))
    return result


class _Cfg:
  class _Rollout:
    def __init__(self):
      # reward normalization config defaults
      class RN:
        def __init__(self):
          self.grouping = "batch"
          self.method = "identity"
      self.reward_normalization = RN()
      self.use_turn_scores = False
      self.training = ["testAgent"]
      self.agent_group_num = [2]
      self.agent_group_size = [2]
      self.rollout_filter_ratio = 1.0
      self.rollout_filter_type = "std"
      self.show_tqdm = False
      self.num_prompt_threads = 0
      self.num_env_threads = 0
      self.num_init_threads = 0
      self.validation_seed = 123

  def __init__(self):
    self.rollout = _Cfg._Rollout()
    # Agent entry as accessed by cfg[agent_name]
    self._agents = {
      "testAgent": {
        "agent_type": "TestAgent",
        "agent_config": {
          "max_turns": 2,
          "max_actions_all_turns": 4,
          "max_actions_per_turn": 2,
          "max_tokens": 16,
          "use_think_answer_token": True,
          "enable_think": True,
          "system_prompt": "You are a helpful assistant.",
          "prompt": "Respond with actions."
        },
        "env_config": {}
      }
    }

  def __getitem__(self, key):
    return self._agents[key]


def _make_dummy_instance(tokenizer: _DummyTokenizer, cfg: _Cfg) -> SyncMultiTurnRollout:
  """Create a SyncMultiTurnRollout instance without running its __init__."""
  inst = object.__new__(SyncMultiTurnRollout)
  # Only attributes needed by the tested helpers
  inst.tokenizer = tokenizer
  inst.cfg = cfg
  return inst


def test_get_masks_and_scores_alignment():
  """Ensure loss_mask drops the last column and score_tensor drops the first.

  We construct an input_ids sequence with three turns marked by <|im_start|>
  and rewards placed at <|im_end|>. Only assistant turns (odd, >1) are trainable.
  """
  tok = _DummyTokenizer(im_start_id=11, im_end_id=12)
  cfg = _Cfg()
  mgr = _make_dummy_instance(tok, cfg)

  # Build a toy batch (B=2) with seq_len=10; mark three turns at positions {0,4,7}
  B, L = 2, 10
  input_ids = np.zeros((B, L), dtype=np.int32)
  input_ids[:, 0] = tok._im_start_id
  input_ids[:, 4] = tok._im_start_id
  input_ids[:, 7] = tok._im_start_id
  # Place <|im_end|> token at final position for a reward
  input_ids[:, -1] = tok._im_end_id
  input_ids = jnp.array(input_ids)

  # All-sample rewards per turn (two samples, one score list each)
  all_scores = [[0.2, 0.3, 0.5], [0.1, 0.1, 0.2]]

  loss_mask, score_tensor, response_mask = mgr.get_masks_and_scores(
      input_ids=input_ids, all_scores=all_scores, use_turn_scores=False
  )

  # Shapes
  assert loss_mask.shape == (B, L - 1)
  assert score_tensor.shape == (B, L - 1)
  assert response_mask.shape == (B, L)

  # Check that loss_mask is a right-truncated version of (turn_indicators > 1)
  # i.e., last column removed
  # and that score_tensor is a left-truncated version (first column removed)
  # i.e., reward on last token aligns to the last column of score_tensor
  assert jnp.all(score_tensor[:, -1] > 0.0)
  assert jnp.all(loss_mask[:, -1] == (input_ids[:, -2] != 0))


def test_normalize_score_tensor_batch_grouping():
  """Validate reward normalization 'batch' grouping and penalty addition."""
  tok = _DummyTokenizer()
  cfg = _Cfg()
  cfg.rollout.reward_normalization.grouping = "batch"
  cfg.rollout.reward_normalization.method = "mean"
  mgr = _make_dummy_instance(tok, cfg)

  # Build a batch with last-column rewards [1.0, 3.0] and penalties [0.1, -0.2]
  B, T = 2, 5
  score_tensor = jnp.zeros((B, T), dtype=jnp.float32)
  score_tensor = score_tensor.at[:, -1].set(jnp.array([1.0, 3.0], dtype=jnp.float32))

  env_outputs = [
      {"group_id": 0, "tag": "A", "penalty": 0.1},
      {"group_id": 0, "tag": "A", "penalty": -0.2},
  ]

  norm_scores = mgr._normalize_score_tensor(score_tensor, env_outputs)

  # After 'mean' normalization across batch: subtract mean 2.0 -> [-1.0, +1.0]
  # Then add penalties [0.1, -0.2] -> [-0.9, 0.8]
  expected_last = jnp.array([-0.9, 0.8], dtype=jnp.float32)
  assert jnp.allclose(norm_scores[:, -1], expected_last, atol=1e-5)



# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive, runnable tests for tunix rollout manager
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleEnvOut:
  def __init__(self, truncated=False, terminated=False, state="", reward=0.0, info=None):
    self.truncated = truncated
    self.terminated = terminated
    self.state = state
    self.reward = reward
    self.info = info or {}


class _TestAgent:
  def __init__(self, config, agent_id, group_id, tag):
    self.agent_id = agent_id
    self.group_id = group_id
    self.tag = tag
    self.agent_config = config["agent_config"]
    self.env_config = config.get("env_config", {})
    self.max_turns = int(self.agent_config.get("max_turns", 2))
    self.enable_think = bool(self.agent_config.get("enable_think", True))
    self.use_think_answer_token = bool(self.agent_config.get("use_think_answer_token", True))
    self.max_actions_all_turns = int(self.agent_config.get("max_actions_all_turns", 4))
    self.max_actions_per_turn = int(self.agent_config.get("max_actions_per_turn", 2))
    self.max_tokens = int(self.agent_config.get("max_tokens", 16))
    self.system_prompt = self.agent_config.get("system_prompt", "You are a helpful assistant.")
    self.prompt = self.agent_config.get("prompt", "Respond with actions.")
    self.cur_turn = 0
    self.total_actions_consumed = 0
    self.penalty = 0.0
    self.messages = [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": self.prompt},
    ]
    self._history = []

  def get_llm_prompts(self, env_out):
    # Ensure there is a user message for this turn
    turn_msg = {
      "role": "user",
      "content": f"Turn {self.cur_turn+1}: State=({env_out.state}) ActionsLeft={self.max_actions_all_turns - self.total_actions_consumed}"
    }
    if self.cur_turn == 0 and len(self.messages) == 2:
      self.messages[1]["content"] += "\n" + turn_msg["content"]
    else:
      self.messages.append(turn_msg)
    return self.messages

  def get_messages(self):
    return self.messages

  def reset(self, seed=None):
    self.cur_turn = 0
    self.total_actions_consumed = 0
    self.penalty = 0.0
    self.messages = [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": self.prompt},
    ]
    self._history = []
    return _SimpleEnvOut(truncated=False, terminated=False, state=f"init(seed={seed})", reward=0.0, info={})

  def _append_turn(self, reply_text):
    # Add assistant response and record step
    self.messages.append({"role": "assistant", "content": reply_text})
    reward = float((len(reply_text) % 5) - 2) * 0.1
    info = {"success": len(reply_text) % 2 == 0, "action_is_valid": True, "action_is_effective": True}
    self._history.append({
      "state": f"state@turn{self.cur_turn}",
      "actions_left": max(0, self.max_actions_all_turns - self.total_actions_consumed),
      "actions": ["act1", "act2"][: self.max_actions_per_turn],
      "reward": reward,
      "info": info,
      "llm_response": reply_text,
      "llm_raw_response": reply_text,
    })
    self.total_actions_consumed += min(2, self.max_actions_per_turn)
    self.cur_turn += 1
    done = self.cur_turn >= self.max_turns
    return _SimpleEnvOut(truncated=False, terminated=done, state=f"state@turn{self.cur_turn}", reward=reward, info=info)

  def get_env_outputs(self, llm_response):
    return self._append_turn(llm_response)

  def get_final_rollout_states(self):
    metrics = {
      f"{self.tag}/success": float(any(step["info"].get("success", False) for step in self._history)),
      f"{self.tag}/num_actions": sum(len(step["actions"]) for step in self._history),
      f"{self.tag}/action_is_effective": 1.0 if self._history else 0.0,
      f"{self.tag}/action_is_valid": 1.0 if self._history else 0.0,
    }
    return {
      "env_id": self.agent_id,
      "history": list(self._history),
      "group_id": self.group_id,
      "tag": self.tag,
      "metrics": metrics,
      "penalty": self.penalty,
    }

  def close(self):
    pass


class _MockRLCluster:
  class _Out:
    def __init__(self, texts):
      self.text = texts
  def generate(self, prompts, mode):
    # Deterministic faux responses that include think/answer for simple parsing
    outs = []
    for p in prompts:
      outs.append("<think>ok</think><answer>act1||act2</answer>")
    return _MockRLCluster._Out(outs)


def _register_test_agent():
  REGISTERED_AGENTS["TestAgent"] = _TestAgent


def _setup_logging():
  log_dir = Path(__file__).resolve().parent / "test_logs"
  os.makedirs(log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  return log_dir / f"tunix_rollout_debug_{ts}.log"


def test_tunix_rollout_creation():
  _register_test_agent()
  cfg = _Cfg()
  tok = _DummyTokenizer(im_start_id=11, im_end_id=12)
  rl = _MockRLCluster()
  rollout = SyncMultiTurnRollout(rl_cluster=rl, cfg=cfg, tokenizer=tok, processor=None, validation=False)
  assert rollout.total_agent_num == cfg.rollout.agent_group_num[0] * cfg.rollout.agent_group_size[0]
  assert len(rollout.agents) == rollout.total_agent_num
  assert rollout.max_turns == cfg["testAgent"]["agent_config"]["max_turns"]
  rollout.close()


def test_tunix_prompt_and_tokenization():
  _register_test_agent()
  cfg = _Cfg()
  tok = _DummyTokenizer(im_start_id=11, im_end_id=12)
  rl = _MockRLCluster()
  rollout = SyncMultiTurnRollout(rl_cluster=rl, cfg=cfg, tokenizer=tok, processor=None, validation=False)
  rollout._reset_batch_agents()
  prompts = rollout.get_batch_llm_prompts(rollout.env_outs)
  assert isinstance(prompts, list) and len(prompts) == rollout.total_agent_num
  # Tokenize these prompts (sanity) and ensure non-empty sequences
  enc = tok(prompts, return_tensors=None, padding=True, padding_side="left")
  assert "input_ids" in enc and "attention_mask" in enc
  assert len(enc["input_ids"]) == len(prompts)
  assert all(len(row) == len(enc["input_ids"][0]) for row in enc["input_ids"])  # equal length
  rollout.close()


def test_tunix_full_rollout_and_batch():
  _register_test_agent()
  cfg = _Cfg()
  tok = _DummyTokenizer(im_start_id=11, im_end_id=12)
  rl = _MockRLCluster()
  rollout = SyncMultiTurnRollout(rl_cluster=rl, cfg=cfg, tokenizer=tok, processor=None, validation=False)
  batch = rollout.rollout()
  # Verify RolloutBatch-like structure
  assert hasattr(batch, "input_ids")
  assert hasattr(batch, "loss_mask")
  assert hasattr(batch, "reward_scores")
  assert hasattr(batch, "agent_raw_data")
  assert hasattr(batch, "meta_info")
  # Shapes are consistent
  N = batch.input_ids.shape[0]
  L = batch.input_ids.shape[1]
  assert batch.loss_mask.shape == (N, L - 1)
  assert batch.reward_scores.shape == (N, L - 1)
  rollout.close()


def test_tunix_final_states_and_filtering():
  _register_test_agent()
  cfg = _Cfg()
  tok = _DummyTokenizer(im_start_id=11, im_end_id=12)
  rl = _MockRLCluster()
  rollout = SyncMultiTurnRollout(rl_cluster=rl, cfg=cfg, tokenizer=tok, processor=None, validation=False)
  batch = rollout.rollout()
  states = rollout._collect_final_rollout_states()
  assert isinstance(states, list) and len(states) == rollout.total_agent_num
  # Test filtering pass-through when ratio == 1
  filtered, metrics = rollout.filter_rollout_batch(batch)
  assert hasattr(filtered, "input_ids") and isinstance(metrics, dict)
  rollout.close()


if __name__ == "__main__":
  log_file = _setup_logging()
  print(f"📝 TUNIX ROLLOUT DEBUG LOG: {log_file}")
  print("🚀 Starting tunix_sync_multi_turn_rollout tests...")

  try:
    print("Test 1: get_masks_and_scores_alignment")
    test_get_masks_and_scores_alignment()
    print("✅ Passed 1")

    print("Test 2: normalize_score_tensor_batch_grouping")
    test_normalize_score_tensor_batch_grouping()
    print("✅ Passed 2")

    print("Test 3: tunix_rollout_creation")
    test_tunix_rollout_creation()
    print("✅ Passed 3")

    print("Test 4: tunix_prompt_and_tokenization")
    test_tunix_prompt_and_tokenization()
    print("✅ Passed 4")

    print("Test 5: tunix_full_rollout_and_batch")
    test_tunix_full_rollout_and_batch()
    print("✅ Passed 5")

    print("Test 6: tunix_final_states_and_filtering")
    test_tunix_final_states_and_filtering()
    print("✅ Passed 6")

    print("🎉 All tests passed for tunix_sync_multi_turn_rollout")
  except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
