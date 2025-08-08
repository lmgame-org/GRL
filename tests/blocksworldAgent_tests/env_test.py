#!/usr/bin/env python3
"""
BlocksworldEnv Test – condensed sanity-checks
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import sys, json
from datetime import datetime
from pathlib import Path

# ── repo imports ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.blocksworldAgent.env import BlocksworldEnv

# ───────────────────────────── logging helper ────────────────────────────────
def setup_logging():
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp       = log_dir / f"blocksworld_env_test_{ts}.log"

    class Tee:
        def __init__(self, path):
            self.file, self.stdout = open(path, "w", encoding="utf-8"), sys.stdout
        def write(self, x): self.file.write(x); self.file.flush(); self.stdout.write(x)
        def flush(self):    self.file.flush();   self.stdout.flush()
        def close(self):    self.file.close()

    tee = Tee(fp)
    sys.stdout = tee
    print(f"📝 BlocksworldEnv test log  –  {datetime.now()}")
    print(f"📄 Log file: {fp}")
    print("="*60)
    return tee

# ────────────────────────── tiny config helper ───────────────────────────────
class BWEnvCfg:
    def __init__(self, num_blocks=3, render_mode="text"):
        self.num_blocks  = num_blocks
        self.render_mode = render_mode

def make_env(cfg):           # BlocksworldEnv expects a dict-like config
    return BlocksworldEnv(vars(cfg))

# ───────────────────────────────── tests ─────────────────────────────────────
def test_creation_and_reset():
    print("🔍 Test 1 – creation & reset")
    env = make_env(BWEnvCfg())
    obs = env.reset(seed=0)
    assert isinstance(obs, str) and obs
    print(f"   Render lines: {len(obs.splitlines())}")
    env.close()

def test_step_logic():
    print("🔍 Test 2 – basic step() logic")
    env  = make_env(BWEnvCfg())
    _    = env.reset(seed=5)

    # ----- invalid action ----------------------------------------------------
    _, r_bad, done_bad, info_bad = env.step("move X to Y")
    assert not done_bad and r_bad < 0 and not info_bad["action_is_valid"]
    print(f"   Invalid action handled → reward {r_bad}")

    # ----- build a valid, non-goal move --------------------------------------
    state          = env.state          # e.g. [0,1,0]  (length = num_blocks)
    num_blocks     = env.num_blocks
    clear_blocks   = [b for b in range(1, num_blocks+1) if b not in state]
    blk            = clear_blocks[0]               # first block with nothing on it
    if state[blk-1] != 0:                          # already on another block
        dest = 0                                   # move to table
    else:                                          # already on table → place on a clear block
        dest = next(d for d in range(1, num_blocks+1)
                    if d != blk and d not in state)
    valid_action   = f"(move {blk} to {dest})"

    _, r_ok, done_ok, info_ok = env.step(valid_action)
    assert info_ok["action_is_valid"], "Expected action_is_valid == True"
    print(f"   Valid action '{valid_action}' → reward {r_ok}")

    env.close()

def test_goal_reached():
    print("🔍 Test 3 – reaching the goal")
    env = make_env(BWEnvCfg())
    _   = env.reset(seed=3)
    goal = env.goal
    for blk, dst in enumerate(goal, 1):
        if env.state[blk-1] != dst:
            env.step(f"(move {blk} to {dst})")
    _, r, done, info = env.step("(move 1 to 0)")   # final no-op trigger
    assert done and info["success"]
    print("   Goal reached → success flag True")
    env.close()

def test_seeding_determinism():
    print("🔍 Test 4 – seeding determinism")
    env = make_env(BWEnvCfg())
    a = env.reset(seed=11)
    b = env.reset(seed=11)
    c = env.reset(seed=22)
    assert a == b and a != c
    print("   Same seed ⇒ identical render; different seed ⇒ different render")
    env.close()

def test_info_structure():
    print("🔍 Test 5 – info-dict structure")
    env = make_env(BWEnvCfg())
    env.reset(seed=1)
    _, _, _, info = env.step("(move 1 to 0)")
    assert set(info) == {"action_is_effective", "action_is_valid", "success"}
    print(f"   Info keys OK → {info}")
    env.close()

# ───────────────────────────────── runner ────────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        print("🚀 Starting BlocksworldEnv tests\n")
        test_creation_and_reset();   print()
        test_step_logic();           print()
        test_goal_reached();         print()
        test_seeding_determinism();  print()
        test_info_structure();       print()
        print("="*60)
        print("🎉 All BlocksworldEnv tests passed")
    except Exception as exc:
        print("❌ Test run failed:", exc)
        import traceback; traceback.print_exc()
    finally:
        tee.close()
        sys.stdout = tee.stdout  # restore console output
