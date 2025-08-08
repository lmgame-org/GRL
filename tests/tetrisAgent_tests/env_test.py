import random, sys, os
from datetime import datetime
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.tetrisAgent.env import TetrisEnv          # ← adjust if needed

# ── logging helper (same pattern as GSM8K test) ──────────────────────────────
def setup_logging():
    test_logs_dir = Path(__file__).parent / "test_logs"
    test_logs_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = test_logs_dir / f"tetris_env_test_{ts}.log"

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

    print(f"📝 Tetris Environment Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    return tee

# ── lightweight config object ────────────────────────────────────────────────
class TetrisEnvConfig:
    dim_x       = 8
    dim_y       = 8
    render_mode = "text"
    box_type    = 2               # shapes I and —
    grid_lookup = {0: "_", 1: "#", 2: "X"}
    action_lookup = {0: "Left", 1: "Right", 2: "Down"}

def get_default_config():
    cfg = TetrisEnvConfig()
    print("✅ Using default Tetris configuration")
    print(f"   Board: {cfg.dim_x}×{cfg.dim_y}  |  box_type {cfg.box_type}")
    return cfg

# Helper: convert config object to dict for TetrisEnv
def make_env(cfg_obj):
    return TetrisEnv(vars(cfg_obj))

# ── individual tests ─────────────────────────────────────────────────────────
def test_env_creation_and_reset():
    print("🔍 Test 1: environment creation & reset")
    env = make_env(get_default_config())
    obs = env.reset(seed=42)
    assert isinstance(obs, str) and obs
    print(f"   Rendered board has {len(obs.splitlines())} rows")
    env.close()

def test_movement_actions():
    print("🔍 Test 2: horizontal moves (Left / Right)")
    env = make_env(get_default_config())
    env.reset(seed=0)
    for action in (0, 1):                        # 0 = Left, 1 = Right
        x0, y0 = env.anchor
        _, _, _, info = env.step(action)
        assert info["action_is_effective"]
        assert env.anchor != (x0, y0)
        print(f"   Action {action} moved the piece as expected")
    env.close()

def test_soft_drop():
    print("🔍 Test 3: soft-drop moves piece downward")
    env = make_env(get_default_config())
    env.reset(seed=0)
    y0 = env.anchor[1]
    _, _, _, info = env.step(2)                  # 2 = Down (soft-drop)
    assert info["action_is_effective"] and env.anchor[1] > y0
    print("   Soft-drop advanced the piece")
    env.close()

def test_lock_and_line_clear():
    print("🔍 Test 4: piece locks & optional line clear")
    env = make_env(get_default_config())
    env.reset(seed=123)
    for _ in range(env.height * 2):
        _, reward, done, info = env.step(2)      # keep dropping
        if info["dropped"]:
            break
    assert info["dropped"]
    if info["success"]:
        assert reward >= 10 and done
        print(f"   Line cleared → reward {reward}")
    else:
        print("   Piece locked with no line clear")
    env.close()

def test_invalid_action():
    print("🔍 Test 5: invalid action handling")
    env = make_env(get_default_config())
    env.reset(seed=0)
    _, _, done, info = env.step(99)
    assert done and "error" in info
    print("   Invalid action path handled gracefully")
    env.close()

def test_render_contains_X():
    print("🔍 Test 6: render shows current piece")
    env = make_env(get_default_config())
    board = env.reset(seed=0)
    assert "X" in board
    print("   Render output includes 'X' marker")
    env.close()

# ── main runner ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        print("🚀 Starting condensed TetrisEnv tests\n")
        test_env_creation_and_reset();   print()
        test_movement_actions();         print()
        test_soft_drop();                print()
        test_lock_and_line_clear();      print()
        test_invalid_action();           print()
        test_render_contains_X();        print()
        print("=" * 60)
        print("🎉 All condensed tests passed!")
        print(f"✅ Completed at {datetime.now()}")
    except Exception as e:
        print("❌ Test run failed:", e)
        import traceback; traceback.print_exc()
    finally:
        tee.close()
        sys.stdout = tee.stdout
