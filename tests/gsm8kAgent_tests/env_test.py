import random, sys, os
from datetime import datetime
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.gsm8kAgent.env import GSM8KEnv

# ── logging helper (unchanged) ───────────────────────────────────────────────
def setup_logging():
    test_logs_dir = Path(__file__).parent / "test_logs"
    test_logs_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = test_logs_dir / f"gsm8k_env_test_{ts}.log"

    class Tee:
        def __init__(self, fp):
            self.file = open(fp, "w")
            self.stdout = sys.stdout
        def write(self, data):
            self.file.write(data); self.file.flush(); self.stdout.write(data)
        def flush(self):
            self.file.flush(); self.stdout.flush()
        def close(self):
            self.file.close()

    tee = Tee(log_file)
    sys.stdout = tee

    print(f"📝 GSM8K Environment Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    return tee

# ── lightweight config object ────────────────────────────────────────────────
class GSM8KEnvConfig:
    def __init__(self):
        self.dataset_path = "openai/gsm8k"
        self.split        = "train"
        self.max_steps    = 10

def get_default_config():
    cfg = GSM8KEnvConfig()
    print("✅ Using default GSM8K configuration")
    print(f"   Dataset path: {cfg.dataset_path}")
    print(f"   Split: {cfg.split}")
    print(f"   Max steps: {cfg.max_steps}")
    return cfg

# Helper: always convert the config object to a dict for GSM8KEnv
def make_env(cfg_obj):
    return GSM8KEnv(vars(cfg_obj))      # <── single-line fix

# ── individual tests ─────────────────────────────────────────────────────────
def test_env_creation_and_reset():
    print("🔍 Test 1: environment creation & reset")
    env = make_env(get_default_config())
    obs = env.reset(seed=42)
    assert isinstance(obs, str) and obs
    print(f"   Question length: {len(obs)} chars")
    env.close()

def test_step_logic():
    print("🔍 Test 2: step() with correct / incorrect answers")
    env = make_env(get_default_config())
    env.reset(seed=123)
    correct = str(env.correct_answer)
    wrong   = str(int(env.correct_answer) + 1) if isinstance(env.correct_answer, int) else "999"

    # correct answer
    _, r, done, info = env.step(correct)
    assert done and r > 0 and info["success"]
    print(f"   Correct path OK → reward {r}")

    # incorrect answer
    env.reset(seed=123)
    _, r, done, info = env.step(wrong)
    assert not done and r < 0 and not info["success"]
    print(f"   Incorrect path OK → reward {r}")
    env.close()

def test_seeding_determinism():
    print("🔍 Test 3: seeding determinism & diversity")
    env = make_env(get_default_config())
    a = env.reset(seed=111)
    b = env.reset(seed=111)
    c = env.reset(seed=222)
    assert a == b and a != c
    print("   Same seed → same question; different seed → different question")
    env.close()

def test_info_structure():
    print("🔍 Test 4: info dict structure")
    env = make_env(get_default_config())
    env.reset(seed=42)
    _, _, _, info = env.step("42")
    assert set(info) == {"action_is_effective", "action_is_valid", "success"}
    assert all(isinstance(v, bool) for v in info.values())
    print(f"   Info keys OK → {info}")
    env.close()

# ── main runner ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        print("🚀 Starting condensed GSM8KEnv tests\n")
        test_env_creation_and_reset();   print()
        test_step_logic();               print()
        test_seeding_determinism();      print()
        test_info_structure();           print()
        print("=" * 60)
        print("🎉 All condensed tests passed!")
        print(f"✅ Completed at {datetime.now()}")
    except Exception as e:
        print("❌ Test run failed:", e)
        import traceback; traceback.print_exc()
    finally:
        tee.close()
        sys.stdout = tee.stdout
