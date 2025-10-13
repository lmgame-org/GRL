import os
import json
from pathlib import Path
import torch

from transformers import AutoTokenizer


class DummyDataProto:
  def __init__(self, batch):
    self.batch = batch
    self.meta_info = {}
    self.non_tensor_batch = {}
  @staticmethod
  def from_single_dict(d):
    return DummyDataProto(d)


class MockActorWG:
  def __init__(self, tokenizer, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    self.world_size = 1
    self.model_name = model_name
    self.tokenizer = tokenizer

  def generate_sequences(self, dp):
    # Always emit a finish call with a valid Sokoban action plan
    input_ids = dp.batch["input_ids"]
    bsz = input_ids.shape[0]
    resp_text = (
      "<function=finish>\n"
      "<parameter=command>submit</parameter>\n"
      "<parameter=result>Right || Right</parameter>\n"
      "</function>"
    )
    # Tokenize using the same tokenizer that will be used to decode later
    resp_ids = self.tokenizer(
      resp_text,
      add_special_tokens=False,
      return_tensors="pt",
    ).input_ids.squeeze(0)
    responses = resp_ids.unsqueeze(0).repeat(bsz, 1)
    raw_texts = [resp_text for _ in range(bsz)]
    return DummyDataProto({"responses": responses, "raw_texts": raw_texts})


def _build_cfg(repo_root: Path):
  class Cfg(dict):
    def __getattr__(self, k):
      return self.get(k)
    def __setattr__(self, k, v):
      self[k] = v

  cfg = Cfg()
  # From configs/base.yaml (use smaller prompt length for speed)
  cfg.max_prompt_length = 4096

  class Roll: ...
  roll = Roll()
  roll.truncation = False
  roll.show_tqdm = False
  roll.num_prompt_threads = 0
  roll.num_env_threads = 0
  roll.num_init_threads = 0
  roll.agent_group_num = [2]
  roll.agent_group_size = [2]
  roll.training = ["sokobanCodingAgent"]
  class RN: ...
  rn = RN()
  rn.grouping = "batch"
  rn.method = "identity"
  roll.reward_normalization = rn
  roll.use_turn_scores = False
  cfg.rollout = roll

  # SokobanCodingAgent config
  from grl_agents.puzzle_agents.sokoban_coding_agent.config import get_sokoban_coding_agent_config
  base = get_sokoban_coding_agent_config()
  base["agent_config"]["max_turns"] = 1
  base["agent_config"]["max_steps"] = 10
  base["agent_config"]["enable_think"] = False
  base["agent_config"]["tool_use"] = True
  base["agent_config"]["workspace_path"] = str(repo_root / "workspace")
  cfg.training = ["sokobanCodingAgent"]
  cfg["sokobanCodingAgent"] = base
  return cfg


def main():
  repo_root = Path(__file__).resolve().parents[2]
  cache_dir = repo_root / "cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  # Use real tokenizer
  model_name = "Qwen/Qwen2.5-0.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  # Inject DummyDataProto into rollout module namespace to avoid heavy deps
  import grl.rollout.torch_sync_rollout as rollout_mod
  rollout_mod.DataProto = DummyDataProto

  # Provide the missing builder that TorchSyncRollout.generate_sequences expects
  from grl.rollout.torch_sync_rollout import TorchSyncRollout
  def _build_dataproto_from_prompts(prompts):
    toks = tokenizer(
      prompts,
      return_tensors="pt",
      padding=True,
      truncation=False,
    )
    return DummyDataProto({
      "input_ids": toks.input_ids,
      "attention_mask": toks.attention_mask,
    })
  # Monkeypatch at class level so instances use it
  try:
    TorchSyncRollout._build_dataproto_from_prompts = staticmethod(_build_dataproto_from_prompts)
  except Exception:
    pass

  actor_wg = MockActorWG(tokenizer, model_name=model_name)
  cfg = _build_cfg(repo_root)

  # Patch AgentGroupBuilder to normalize agent.tag to 'sokobanCodingAgent'
  import grl_agents.agent_group_builder as agb
  _orig_make_agents = agb.AgentGroupBuilder.make_agents
  async def _patched_make_agents(self):
    agents = await _orig_make_agents(self)
    for a in agents:
      try:
        a.tag = "sokobanCodingAgent"
      except Exception:
        pass
    return agents
  try:
    agb.AgentGroupBuilder.make_agents = _patched_make_agents
  except Exception:
    pass

  # Build and run rollout
  rollout = TorchSyncRollout(actor_wg, cfg, tokenizer, validation=False)
  batch = rollout.rollout()

  # Save rollout batch summary
  out_batch = cache_dir / "qwen_rollout_batch.json"
  payload = {
    "input_ids_shape": list(batch.input_ids.shape),
    "loss_mask_shape": list(batch.loss_mask.shape),
    "reward_scores_shape": list(batch.reward_scores.shape),
    "meta_info": batch.meta_info,
  }
  out_batch.write_text(json.dumps(payload), encoding="utf-8")

  # Save final rollout states for all agents
  states = rollout._collect_final_rollout_states()
  out_states = cache_dir / "qwen_rollout_states.json"
  try:
    out_states.write_text(json.dumps(states), encoding="utf-8")
  except Exception:
    # Fallback if not JSON-serializable
    out_states.write_text(str(states), encoding="utf-8")

  # Stream-like console and file logging
  log_path = cache_dir / "tool_demo_log.txt"
  def _append_log(txt: str):
    try:
      with log_path.open("a", encoding="utf-8") as f:
        f.write(txt)
        if not txt.endswith("\n"):
          f.write("\n")
    except Exception:
      pass

  _append_log("=== Qwen Rollout Batch Summary ===")
  _append_log(repr(payload))
  _append_log("=== Final Rollout States (repr) ===")
  for i, st in enumerate(states):
    try:
      _append_log(f"[Agent {i}] {repr(st)}")
    except Exception:
      _append_log(f"[Agent {i}] <unprintable state>")

  # Console logs
  print("Saved batch:", out_batch)
  print("Saved states:", out_states)
  print("Saved stream log:", log_path)


if __name__ == "__main__":
  main()
