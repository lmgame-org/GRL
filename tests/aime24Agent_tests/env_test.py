#!/usr/bin/env python3
import sys, re
from datetime import datetime
from pathlib import Path

# project root on path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from grl.agents.aime24Agent.env import AIME24Env


def setup_logging():
  log_dir = Path(__file__).parent / "test_logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = log_dir / f"aime24_env_test_{ts}.log"

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
  print(f"📝 AIME24Env Test log started at {datetime.now()}")
  print(f"📄 Log file: {log_path}")
  print("=" * 60)
  return tee


class AIME24EnvConfig:

  def __init__(self):
    self.dataset_path = "math-ai/aime24"
    self.split = "test"
    self.max_steps = 5


def get_default_config():
  cfg = AIME24EnvConfig()
  print("✅ Using default AIME24 configuration")
  print(f"   Dataset path: {cfg.dataset_path}")
  print(f"   Split: {cfg.split}")
  return cfg


def make_env(cfg_obj):
  return AIME24Env(vars(cfg_obj))


def test_env_creation_and_reset():
  print("🔍 Test 1: environment creation & reset")
  env = make_env(get_default_config())
  obs = env.reset(seed=42)
  assert isinstance(obs, str) and obs
  print(f"   Question len: {len(obs)}")
  print("   Question sample:")
  print("   " + obs[:200].replace("\n", " ") + (" …" if len(obs) > 200 else ""))
  print(f"   Answer sample: {env.correct_answer}")
  assert isinstance(env.correct_answer, (str, int, float))
  env.close()


def test_step_logic_wrappers():
  print("🔍 Test 2: step() correctness with common wrappers")
  env = make_env(get_default_config())
  env.reset(seed=123)
  gold = str(env.correct_answer)
  candidates = [
      gold,
      f"$ {gold} $",
      f"\\boxed{{{gold}}}",
      f"\\text{{{gold}}}",
  ]
  for cand in candidates:
    _, r, done, info = env.step(cand)
    assert info["action_is_valid"] is True
    assert info["success"] is True and r > 0 and done is True
    print(f"   Accepted: {repr(cand)}")
    env.reset(seed=123)
  env.close()


def test_optional_equivalences():
  print("🔍 Test 3: optional equivalence cases (fraction/tuple if present)")
  env = make_env(get_default_config())
  env.reset(seed=321)
  gold = str(env.correct_answer)

  # simple tuple spacing tolerance
  if "," in gold and (gold.startswith("(") and gold.endswith(")")):
    compact = gold.replace(", ", ",")
    _, r, done, info = env.step(compact)
    assert info["success"] is True and done is True and r > 0
    print("   Tuple spacing accepted")

  env.close()


def test_seeding_determinism():
  print("🔍 Test 4: seeding determinism & diversity")
  env = make_env(get_default_config())
  a = env.reset(seed=111)
  b = env.reset(seed=111)
  c = env.reset(seed=222)
  assert a == b and a != c
  print("   Same seed → same question; different seed → different question")
  env.close()


def test_info_structure():
  print("🔍 Test 5: info dict structure")
  env = make_env(get_default_config())
  env.reset(seed=42)
  _, _, _, info = env.step(str(env.correct_answer))
  assert set(info) == {"action_is_effective", "action_is_valid", "success"}
  assert all(isinstance(v, bool) for v in info.values())
  print(f"   Info keys OK → {info}")
  env.close()


if __name__ == "__main__":
  tee = setup_logging()
  try:
    print("🚀 Starting AIME24Env tests\n")
    test_env_creation_and_reset()
    print()
    test_step_logic_wrappers()
    print()
    test_optional_equivalences()
    print()
    test_seeding_determinism()
    print()
    test_info_structure()
    print()
    print("=" * 60)
    print("🎉 All AIME24Env tests passed!")
    print(f"✅ Completed at {datetime.now()}")
  except Exception as e:
    print("❌ Test run failed:", e)
    import traceback

    traceback.print_exc()
  finally:
    tee.close()
    sys.stdout = tee.stdout
