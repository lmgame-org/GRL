#!/usr/bin/env python3
"""
Condensed sanity-checks for WebShopEnv
Similar in spirit to gsm8k_env_test.py
"""

import sys, os, random, string
from datetime import datetime
from pathlib import Path
from typing import Dict

# ── project / external path hack ────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))                # local packages
sys.path.insert(0, str(project_root / "external" / "webshop-minimal"))

from lmgamerl.agents.webshopAgent.env import WebShopEnv         # adjust if module path differs

# ── logging helper (same pattern as GSM8K) ──────────────────────────────────
def setup_logging():
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"webshop_env_test_{ts}.log"

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

    print(f"📝 WebShop Environment Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    return tee

# ── lightweight default config ──────────────────────────────────────────────
class WebShopEnvCfg(dict):
    """Plain dict-subclass so we can pass it straight to WebShopEnv"""
    def __init__(self):
        super().__init__(
            observation_mode = "text",
            file_path        = '',
            server           = None,
            filter_goals     = None,
            limit_goals      = -1,
            num_products     = None,
            human_goals      = False,
            show_attrs       = False,
            dataset_size     = "small",
        )

def get_default_config() -> Dict:
    cfg = WebShopEnvCfg()
    print("✅ Using default WebShop configuration")
    for k, v in cfg.items():
        print(f"   {k}: {v}")
    return cfg

def make_env(cfg_dict):
    return WebShopEnv(cfg_dict)

# ── individual tests ────────────────────────────────────────────────────────
def test_env_creation_and_reset():
    print("🔍 Test 1: environment creation & reset")
    env = make_env(get_default_config())
    obs = env.reset(seed=42)
    print("=" * 60)
    print(f"obs: {obs}")
    print("=" * 60)
    assert isinstance(obs, str) and obs
    print(f"   Observation length: {len(obs)} chars")
    env.close()

def test_step_logic():
    print("🔍 Test 2: step() executes with a valid action")
    env = make_env(get_default_config())
    env.reset(seed=0)
    action = env.get_available_actions()[0]          # take first legal action
    _, r, done, info = env.step(action)
    assert "action_is_valid" in info and info["action_is_valid"]
    print(f"   Took action '{action}' → reward {r}, done={done}")
    env.close()

def test_seeding_determinism():
    print("🔍 Test 3: seeding determinism")
    env = make_env(get_default_config())
    a = env.reset(seed=111)
    b = env.reset(seed=111)
    c = env.reset(seed=222)
    assert a == b and a != c
    print("   Same seed → same obs; different seed → different obs")
    env.close()

def test_info_structure():
    print("🔍 Test 4: info dict structure")
    env = make_env(get_default_config())
    env.reset(seed=0)
    action = env.get_available_actions()[0]
    _, _, _, info = env.step(action)
    assert set(info) == {"action_is_effective", "action_is_valid", "success"}
    assert all(isinstance(v, bool) for v in info.values())
    print(f"   Info keys OK → {info}")
    env.close()

# ── main runner ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        print("🚀 Starting condensed WebShopEnv tests\n")
        test_env_creation_and_reset(); print()
        test_step_logic();             print()
        test_seeding_determinism();    print()
        test_info_structure();         print()
        print("=" * 60)
        print("🎉 All condensed tests passed!")
        print(f"✅ Completed at {datetime.now()}")
    except Exception as e:
        print("❌ Test run failed:", e)
        import traceback; traceback.print_exc()
    finally:
        tee.close()
        sys.stdout = tee.stdout
