#!/usr/bin/env python3
import sys, json
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from grl.agents.math500Agent.agent import Math500Agent
import yaml


def setup_logging():
  log_dir = Path(__file__).parent / "test_logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = log_dir / f"math500_agent_test_{ts}.log"

  class Tee:

    def __init__(self, fp):
      self.file = open(fp, "w")
      self.stdout = sys.stdout

    def write(self, x):
      self.file.write(x)
      self.file.flush()
      self.stdout.write(x)

    def flush(self):
      self.file.flush()
      self.stdout.flush()

    def close(self):
      self.file.close()

  tee = Tee(log_path)
  sys.stdout = tee
  print(f"📝 Math500Agent Test log started at {datetime.now()}")
  print(f"📄 Log file: {log_path}")
  print("=" * 60)
  return tee


def load_config():
  cfg_dir = project_root / "configs"
  with open(cfg_dir / "base.yaml") as f:
    base_cfg = yaml.safe_load(f)
  with open(cfg_dir / "agents.yaml") as f:
    agent_cfgs = yaml.safe_load(f)
  cfg = {**base_cfg, **agent_cfgs}
  print(f"✅ Loaded configuration from {cfg_dir}")
  return cfg


def test_agent_creation():
  print("🔍 Testing Math500Agent creation …")
  cfg = load_config()
  ag = Math500Agent(
      cfg["math500Agent_single_turn"],
      group_id=0,
      agent_id=0,
      seed=42,
      tag="TestMath500",
  )
  assert ag.max_turns >= 1
  assert hasattr(ag, "env") and hasattr(ag.env, "reset")
  print("✅ Creation OK — max_turns:", ag.max_turns)
  ag.close()


def test_agent_reset_and_step():
  print("\n🔍 Testing reset and one step …")
  cfg = load_config()
  ag = Math500Agent(cfg["math500Agent_single_turn"], agent_id=0, group_id=0)
  env_out = ag.reset(seed=123)
  assert not (env_out.truncated or env_out.terminated)

  # respond with the correct answer directly
  gold = str(ag.env.correct_answer)
  out = ag.get_env_outputs(f"<answer>{gold}</answer>")
  assert out.reward > 0 and (out.truncated or out.terminated)
  print("✅ Step with correct answer finished")
  ag.close()


def test_single_rollout():
  print("\n🔍 Testing one complete rollout …")
  cfg = load_config()
  ag = Math500Agent(
      cfg["math500Agent_single_turn"], agent_id=0, group_id=0, seed=0
  )
  env_out = ag.reset()

  # always answer correctly once
  gold = str(ag.env.correct_answer)
  step = 0
  while step < ag.max_turns and not (env_out.truncated or env_out.terminated):
    _ = ag.get_llm_prompts(env_out)
    env_out = ag.get_env_outputs(f"<answer>{gold}</answer>")
    step += 1
    print(
        f"   Turn {step}: reward={env_out.reward}, done={env_out.truncated or env_out.terminated}"
    )

  states = ag.get_final_rollout_states()
  print(f"\n📊 Final metrics: {json.dumps(states['metrics'], indent=2)}")
  ag.close()
  print("✅ Rollout test done")


if __name__ == "__main__":
  tee = setup_logging()
  try:
    test_agent_creation()
    test_agent_reset_and_step()
    test_single_rollout()
    print("\n🎉 ALL Math500Agent TESTS PASSED!")
  finally:
    tee.close()
    sys.stdout = tee.stdout
