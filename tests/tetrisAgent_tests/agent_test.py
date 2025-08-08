#!/usr/bin/env python3
"""
TetrisAgent Test – Exercises creation, reset, action parsing, and rollout logic
with mocked LLM responses.  Styled after the SokobanAgent test suite.
"""
# ── stdlib ──────────────────────────────────────────────────────────────
import sys, os, json, yaml, random
from datetime import datetime
from pathlib import Path

# ── project imports ─────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.tetrisAgent.agent import TetrisAgent      # adjust path if needed

# ╭────────────────────────── LOGGING SETUP ───────────────────────────╮
def setup_logging():
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"tetris_agent_test_{ts}.log"

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

    tee = Tee(logfile)
    sys.stdout = tee
    print(f"📝 TetrisAgent Test log started at {datetime.now()}")
    print(f"📄 Log file: {logfile}")
    print("=" * 60)
    return tee

# ╭──────────────────────── CONFIG LOADER ─────────────────────────────╮
def load_config():
    cfg_dir = project_root / "configs"
    with open(cfg_dir / "base.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open(cfg_dir / "agents.yaml") as f:
        agent_cfg = yaml.safe_load(f)
    cfg = {**base_cfg, **agent_cfg}
    print(f"✅ Loaded config from {cfg_dir}")
    return cfg

# ╭──────────────────────── MOCK RESPONSES ────────────────────────────╮
def mock_llm_responses():
    return [
        "<answer>Left || Down</answer>",
        "<answer>Right</answer>",
        "<answer>Down || Down</answer>",
        "<answer>Left</answer>",
        "<answer>Right || Down</answer>",
        "<answer>Left</answer>",
    ]

# ╭──────────────────────── INDIVIDUAL TESTS ──────────────────────────╮
def test_agent_creation():
    print("🔍 Test 1: TetrisAgent creation")
    cfg = load_config()
    agent = TetrisAgent(cfg["tetrisAgent_type_1_dim_4"], group_id=0, agent_id=0, seed=42, tag="TestTetris")
    assert agent.max_turns > 0 and agent.max_actions_all_turns > 0
    assert hasattr(agent, "env") and agent.env.board.shape == (agent.env.width, agent.env.height)
    print("   ✅ creation OK")
    agent.close()

def test_agent_reset():
    print("🔍 Test 2: reset() returns valid EnvOutput")
    cfg = load_config()
    agent = TetrisAgent(cfg["tetrisAgent_type_1_dim_4"], agent_id=0, group_id=0, seed=123)
    env_out = agent.reset(seed=123)
    assert env_out.truncated is False and env_out.reward == 0.0 and isinstance(env_out.state, str)
    print("   ✅ reset OK – state len:", len(env_out.state))
    agent.close()

def test_action_parsing():
    print("🔍 Test 3: action extraction logic")
    cfg = load_config()
    agent = TetrisAgent(cfg["tetrisAgent_type_1_dim_4"], seed=0)
    test_cases = [
        ("<answer>Left || Right</answer>", ["Left", "Right"]),
        ("<answer>Down</answer>", ["Down"]),
        ("<answer>Left</answer>", ["Left"]),
    ]
    agent.reset()
    for llm_resp, expected in test_cases:
        processed, acts = agent.parse_llm_response(llm_resp, enable_think=False)
        print(f"extracted actions: {acts}")
        assert acts == expected
    print("   ✅ parsing OK")
    agent.close()

def test_rollout():
    print("🔍 Test 4: full rollout with mocked LLM")
    cfg = load_config()
    agent = TetrisAgent(cfg["tetrisAgent_type_1_dim_4"], seed=0, tag="RolloutTetris")
    mocks = mock_llm_responses()
    env_out = agent.reset()
    idx = 0
    while not env_out.terminated and idx < len(mocks):
        prompts = agent.get_llm_prompts(env_out)
        env_out = agent.get_env_outputs(mocks[idx]); idx += 1
    print(f"   ✅ rollout finished – total reward {env_out.reward}")
    agent.close()

def test_final_states():
    print("🔍 Test 5: final rollout states structure")
    cfg = load_config()
    agent = TetrisAgent(cfg["tetrisAgent_type_1_dim_4"], seed=7)
    agent.reset()
    agent.get_env_outputs("<answer>Down</answer>")
    final = agent.get_final_rollout_states()
    keys = {"env_id", "group_id", "tag", "history", "metrics"}
    assert keys.issubset(final)
    print("   ✅ structure OK")
    agent.close()

# ╭──────────────────────── MAIN RUNNER ───────────────────────────────╮
if __name__ == "__main__":
    tee = setup_logging()
    try:
        test_agent_creation();   print()
        test_agent_reset();      print()
        test_action_parsing();   print()
        test_rollout();          print()
        test_final_states();     print()
        print("🎉 ALL TetrisAgent tests passed!")
    except Exception as e:
        print("❌ Test run failed:", e)
        import traceback; traceback.print_exc()
    finally:
        tee.close(); sys.stdout = tee.stdout
