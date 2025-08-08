#!/usr/bin/env python3
"""
GSM8KAgent Test – exercises agent rollout logic with mocked LLM responses
(adapted from the SokobanAgent test suite).
"""

# ── standard imports ──────────────────────────────────────────────────────────
import sys, os, json, yaml, random
from datetime import datetime
from pathlib import Path

# ── project imports ───────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.gsm8kAgent.agent import GSM8KAgent     # the agent under test

# ────────────────────────── logging helper ────────────────────────────────────
def setup_logging() -> "Tee":
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"gsm8k_agent_test_{ts}.log"

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
    print(f"📝 GSM8KAgent Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_path}")
    print("=" * 60)
    return tee

# ─────────────────────────── config loader ────────────────────────────────────
def load_config():
    cfg_dir = project_root / "configs"
    with open(cfg_dir / "base.yaml")   as f: base_cfg   = yaml.safe_load(f)
    with open(cfg_dir / "agents.yaml") as f: agent_cfgs = yaml.safe_load(f)
    cfg = {**base_cfg, **agent_cfgs}
    print(f"✅ Loaded configuration from {cfg_dir}")
    return cfg

# ─────────────────────────── mock LLM answers helper ──────────────────────────
def get_mock_llm_responses():
    # answers vary in phrasing; GSM8KEnv extracts the *last* integer
    answers = [
        "...</think><answer>42</answer>",
        "...</think><answer>The answer is 160</answer>",
        "...</think><answer>  8  </answer>",
        "...</think><answer>It should be 275 dollars</answer>",
        "...</think><answer>99%</answer>",
        "...</think><answer>12</answer>",
    ]
    random.shuffle(answers)
    return answers

# ───────────────────────────── individual tests ───────────────────────────────
def test_agent_creation():
    print("🔍 Testing GSM8KAgent creation …")
    cfg = load_config()
    ag = GSM8KAgent(cfg["gsm8kAgent_single_turn"],
                    group_id=0, agent_id=0, seed=42, tag="TestGSM8K")
    assert ag.max_turns >= 1
    assert hasattr(ag, "env") and hasattr(ag.env, "reset")
    print("✅ Creation OK — max_turns:", ag.max_turns)
    ag.close()

def test_agent_reset():
    print("\n🔍 Testing reset …")
    cfg = load_config()
    ag = GSM8KAgent(cfg["gsm8kAgent_single_turn"], agent_id=0, group_id=0, seed=42)
    env_out = ag.reset(seed=123)

    # new API ⇒ ‘done’ = truncated or terminated
    done_flag = env_out.truncated or env_out.terminated
    assert done_flag is False and isinstance(env_out.state, str)

    print("✅ Reset OK — state preview:", env_out.state[:80], "…")
    ag.close()

def test_answer_processing():
    print("\n🔍 Testing answer parsing / env.step …")
    cfg = load_config()
    ag = GSM8KAgent(cfg["gsm8kAgent_single_turn"], agent_id=0, group_id=0)
    ag.reset(seed=1)
    cases = [
        "<answer>42</answer>",
        "<answer>The result is 17</answer>",
        "random text"
    ]
    for txt in cases:
        out = ag.get_env_outputs(txt)
        print(f"   '{txt}' → reward {out.reward}, "
              f"valid {out.info.get('action_is_valid')}")
    ag.close()
    print("✅ Answer processing test finished")

def test_single_rollout():
    print("\n🔍 Testing one complete rollout …")
    cfg = load_config()
    ag = GSM8KAgent(cfg["gsm8kAgent_single_turn"], agent_id=0, group_id=0, seed=0)
    env_out = ag.reset()
    mock = get_mock_llm_responses()

    step = 0
    while step < ag.max_turns and not (env_out.truncated or env_out.terminated):
        _ = ag.get_llm_prompts(env_out)          # prompts not needed for mock
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
    Unit-test the agent’s internal parse_llm_response.
    """
    print("\n🔍 Unit-test GSM8KAgent.parse_llm_response …")

    # minimal dummy config so constructor doesn't explode
    dummy_cfg = {
        "agent_config": {},
        "env_config": {"dataset_path": "openai/gsm8k", "split": "train"}
    }
    ag = GSM8KAgent(dummy_cfg)

    s = "<think>… reasoning …</think><answer>7</answer>"
    proc, acts = ag.parse_llm_response(s, enable_think=True)
    assert acts == ["7"], f"Expected ['7'], got {acts}"
    print(f"✅ parse_llm_response returned {acts}")
    ag.close()

def test_final_states_and_messages():
    """
    Run two turns and print:
      • full JSON from get_final_rollout_states()
      • the current agent.messages list (produced by get_llm_prompts)
    """
    print("\n🔍 Testing final rollout states & message history …")
    cfg = load_config()
    ag  = GSM8KAgent(cfg["gsm8kAgent_single_turn"], agent_id=0, group_id=0, seed=101)

    # turn-0 reset
    env_out = ag.reset()
    mock    = get_mock_llm_responses()

    # two mock turns (or until done)
    for turn_idx in range(2):
        if env_out.truncated or env_out.terminated:
            break
        ag.get_llm_prompts(env_out)
        env_out = ag.get_env_outputs(mock[turn_idx])

    # print rollout states
    rollout = ag.get_final_rollout_states()
    print("\n📜 get_final_rollout_states():")
    print(json.dumps(rollout, indent=2, default=str))

    # print assembled message history
    print("\n💬 Message list (agent.messages):")
    for i, m in enumerate(ag.get_messages(), 1):
        role = m["role"]; snippet = m["content"]
        print(f"  {i:02d}. {role.upper():9s} | {repr(snippet)}")

    ag.close()
    print("✅ Final states & messages test finished")

# ─────────────────────────────── main runner ──────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        test_agent_creation()
        test_agent_reset()
        test_answer_processing()
        test_single_rollout()
        test_parse_response_unit()
        test_final_states_and_messages()
        print("\n🎉 ALL GSM8KAgent TESTS PASSED!")
    finally:
        tee.close()
        sys.stdout = tee.stdout  # restore original stdout
