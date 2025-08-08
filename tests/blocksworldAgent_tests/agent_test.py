#!/usr/bin/env python3
"""
BlocksworldAgent test-suite
(adapted from the GSM8KAgent tests).
"""

# ── stdlib ──────────────────────────────────────────────────────────────
import sys, os, json, yaml, random
from datetime import datetime
from pathlib import Path

# --- project root ---------------------------------------------------------
# go up TWO levels from tests/blocksworldAgent_tests/agent_test.py
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# ── agent under test ────────────────────────────────────────────────────
from lmgamerl.agents.blocksworldAgent.agent import BlocksworldAgent

# ────────────────────────── logging helper ──────────────────────────────
def setup_logging() -> "Tee":
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"blocksworld_agent_test_{ts}.log"

    class Tee:
        def __init__(self, fp):
            self.file = open(fp, "w")
            self.stdout = sys.stdout
        def write(self, x):
            self.file.write(x); self.file.flush(); self.stdout.write(x)
        def flush(self):
            self.file.flush(); self.stdout.flush()
        def close(self):
            self.file.close()

    tee = Tee(log_path)
    sys.stdout = tee
    print(f"📝 BlocksworldAgent Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_path}")
    print("=" * 60)
    return tee

# --- config loader --------------------------------------------------------
def load_config():
    cfg_dir = project_root / "configs"          # <repo>/configs
    with open(cfg_dir / "base.yaml")   as f: base_cfg   = yaml.safe_load(f)
    with open(cfg_dir / "agents.yaml") as f: agent_cfgs = yaml.safe_load(f)
    cfg = {**base_cfg, **agent_cfgs}
    print(f"✅ Loaded configuration from {cfg_dir}")
    return cfg

# ───────────────────── mock LLM action sequences ────────────────────────
def get_mock_llm_responses():
    """
    Each response should be parseable by BlocksworldAgent.parse_llm_response
    and yield a list like ["move 2 to 0" , "move 1 to 3"].
    """
    samples = [
        "<think>...</think><answer>(move 1 to 0) || (move 2 to 1)</answer>",
        "<answer>(move 3 to 0)</answer>",
        "<answer>(move 2 to 1) || (move 1 to 2) || (move 3 to 0)</answer>",
        "<answer>(move 1 to 0)</answer>",
    ]
    random.shuffle(samples)
    return samples

# ────────────────────────── individual tests ────────────────────────────
def test_agent_creation():
    print("🔍 Testing BlocksworldAgent creation …")
    cfg = load_config()
    ag  = BlocksworldAgent(cfg["blocksworldAgent_text"],
                           group_id=0, agent_id=0, seed=42, tag="TestBW")
    assert ag.max_turns >= 1
    assert hasattr(ag, "env") and hasattr(ag.env, "reset")
    print("✅ Creation OK — max_turns:", ag.max_turns)
    ag.close()

def test_agent_reset():
    print("\n🔍 Testing reset …")
    cfg = load_config()
    ag  = BlocksworldAgent(cfg["blocksworldAgent_text"], seed=123)
    env_out = ag.reset()

    done_flag = env_out.truncated or env_out.terminated
    assert not done_flag and isinstance(env_out.state, str)
    print("✅ Reset OK — initial state preview:", env_out.state[:60], "…")
    ag.close()

def test_action_parsing_and_step():
    print("\n🔍 Testing action parsing & env.step …")
    cfg = load_config()
    ag = BlocksworldAgent(cfg["blocksworldAgent_text"], seed=1)
    ag.reset()
    cases = [
        "<answer>(move 1 to 0)</answer>",
        "<answer>(move 2 to 1) || (move 1 to 2)</answer>",
        "nonsense text",
    ]
    for txt in cases:
        out = ag.get_env_outputs(txt)
        print(f"   '{txt}' → reward={out.reward}, "
              f"valid={out.info.get('action_is_valid')}")
    ag.close()
    print("✅ Action parsing test finished")

def test_single_rollout():
    print("\n🔍 Testing one complete rollout …")
    cfg = load_config()
    ag  = BlocksworldAgent(cfg["blocksworldAgent_text"], seed=0)
    env_out = ag.reset()
    mock = get_mock_llm_responses()

    step = 0
    while step < ag.max_turns and not (env_out.truncated or env_out.terminated):
        _ = ag.get_llm_prompts(env_out)           # prompts not needed for mock
        env_out = ag.get_env_outputs(mock[step % len(mock)])
        step += 1
        print(f"   Turn {step}: reward={env_out.reward}, "
              f"done={env_out.truncated or env_out.terminated}")

    states = ag.get_final_rollout_states()
    print(f"\n📊 Final metrics: {json.dumps(states['metrics'], indent=2)}")
    ag.close()
    print("✅ Rollout test done")

def test_parse_response_unit():
    """
    Unit-test BlocksworldAgent.parse_llm_response.
    """
    print("\n🔍 Unit-test parse_llm_response …")
    dummy_cfg = {
        "agent_config": {
            "max_actions_per_turn": 10,
        },
        "env_config": {"num_blocks": 3}
    }
    ag = BlocksworldAgent(dummy_cfg)

    s = "<think>…</think><answer>(move 1 to 0) || (move 2 to 1)</answer>"
    proc, acts = ag.parse_llm_response(s, enable_think=True)
    assert acts == ["(move 1 to 0)", "(move 2 to 1)"], f"got {acts}"
    print("✅ parse_llm_response returned:", acts)
    ag.close()

def test_final_states_and_messages():
    print("\n🔍 Testing final rollout states & message history …")
    cfg = load_config()
    ag  = BlocksworldAgent(cfg["blocksworldAgent_text"], seed=101)

    env_out = ag.reset()
    mock    = get_mock_llm_responses()

    for turn_idx in range(2):
        if env_out.truncated or env_out.terminated:
            break
        ag.get_llm_prompts(env_out)
        env_out = ag.get_env_outputs(mock[turn_idx])

    # show rollout summary
    rollout = ag.get_final_rollout_states()
    print("\n📜 get_final_rollout_states() =")
    print(json.dumps(rollout, indent=2, default=str))

    # show assembled messages
    print("\n💬 agent.messages:")
    for i, m in enumerate(ag.get_messages(), 1):
        role, snippet = m["role"], m["content"]
        print(f"  {i:02d}. {role.upper():9s} | {repr(snippet)}")

    ag.close()
    print("✅ Final states & messages test finished")

# ──────────────────────────── main runner ───────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        test_agent_creation()
        test_agent_reset()
        test_action_parsing_and_step()
        test_single_rollout()
        test_parse_response_unit()
        test_final_states_and_messages()
        print("\n🎉 ALL BlocksworldAgent TESTS PASSED!")
    finally:
        tee.close()
        sys.stdout = tee.stdout  # restore stdout
