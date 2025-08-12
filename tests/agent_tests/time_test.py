#!/usr/bin/env python3
"""
Timing test-suite for all available agents.
Measures initialization, reset, prompt building, and environment step times.

Style and logging follow tests in birdAgent_tests/agent_test.py.
"""

# ── stdlib ──────────────────────────────────────────────────────────────
import sys, os, json, yaml, time
from datetime import datetime
from pathlib import Path

# --- project root ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── agent registry ───────────────────────────────────────────────────────
from grl.agents import (
    list_registered_agents,
    list_unavailable_agents,
    get_agent_cls,
)


# ────────────────────────── logging helper ──────────────────────────────
def setup_logging():
    log_dir = PROJECT_ROOT / "tests" / "agent_tests" / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"time_test_{ts}.log"
    json_path = log_dir / f"time_test_{ts}.json"

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
    print(f"📝 Agent timing log started at {datetime.now()}")
    print(f"📄 Log file: {log_path}")
    print(f"📄 JSON:     {json_path}")
    print("=" * 60)
    return tee, json_path


# --- config loader --------------------------------------------------------
def load_config_dicts():
    cfg_dir = PROJECT_ROOT / "configs"
    with open(cfg_dir / "base.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open(cfg_dir / "agents.yaml") as f:
        agent_cfgs = yaml.safe_load(f)
    cfg = {**base_cfg, **agent_cfgs}
    print(f"✅ Loaded configuration from {cfg_dir}")
    return cfg


# ────────────────────────── helpers / defaults ───────────────────────────
# Representative task key per agent type, sourced from configs/agents.yaml
REPRESENTATIVE_TASK_FOR_TYPE = {
    "sokobanAgent": "simpleSokobanAgent",
    "gsm8kAgent": "gsm8kAgent_single_turn",
    "blocksworldAgent": "blocksworldAgent_text",
    "tetrisAgent": "tetrisAgent_type_1_dim_4",
    "webshopAgent": "webshopAgent",
    "birdAgent": "birdAgent",
}

# Minimal mock replies per agent type to exercise env stepping
MOCK_REPLY_FOR_TYPE = {
    "sokobanAgent": "<answer>Right</answer>",
    "gsm8kAgent": "<answer>1</answer>",
    "blocksworldAgent": "<answer>(move 1 to 0)</answer>",
    "tetrisAgent": "<answer>Down</answer>",
    "webshopAgent": "<answer>search[test]</answer>",
    "birdAgent": "<answer>```sql\nSELECT 1;\n```</answer>",
}


def time_one_agent(agent_type: str, cfg: dict, n_steps: int = 100) -> dict:
    """
    Time a representative agent instance over several operations.
    Returns a metrics dict.
    """
    result = {
        "agent_type": agent_type,
        "status": "ok",
        "init_ms": None,
        "reset_ms": None,
        "prompt_ms_avg": None,
        "env_ms_avg": None,
        "total_prompt_ms": None,
        "total_env_ms": None,
        "errors": [],
        "n_steps": 0,
    }

    task_key = REPRESENTATIVE_TASK_FOR_TYPE.get(agent_type)
    if task_key is None or task_key not in cfg:
        result["status"] = "skipped"
        result["errors"].append(f"No representative task found for {agent_type}")
        return result

    agent_cfg = cfg[task_key]
    AgentCls = get_agent_cls(agent_type)

    # Initialize
    t0 = time.perf_counter()
    try:
        ag = AgentCls(agent_cfg, group_id=0, agent_id=0, seed=123, tag=f"TimeTest:{agent_type}")
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"init: {e}")
        return result
    result["init_ms"] = (time.perf_counter() - t0) * 1000.0

    # Reset
    t0 = time.perf_counter()
    try:
        env_out = ag.reset()
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"reset: {e}")
        try:
            ag.close()
        except Exception:
            pass
        return result
    result["reset_ms"] = (time.perf_counter() - t0) * 1000.0

    # Benchmark prompt building and env stepping
    reply = MOCK_REPLY_FOR_TYPE.get(agent_type, "<answer></answer>")
    total_prompt, total_env, steps_done = 0.0, 0.0, 0
    for i in range(n_steps):
        # Prompt creation
        t0 = time.perf_counter()
        try:
            _ = ag.get_llm_prompts(env_out)
        except Exception as e:
            result["errors"].append(f"prompt@{i}: {e}")
            break
        total_prompt += (time.perf_counter() - t0) * 1000.0

        # Env step
        t0 = time.perf_counter()
        try:
            env_out = ag.get_env_outputs(reply)
        except Exception as e:
            result["errors"].append(f"env@{i}: {e}")
            break
        total_env += (time.perf_counter() - t0) * 1000.0
        steps_done += 1

        if env_out.truncated or env_out.terminated:
            # restart a new episode to keep counting steps
            try:
                env_out = ag.reset()
            except Exception as e:
                result["errors"].append(f"reset_after_done@{i}: {e}")
                break

    result["n_steps"] = steps_done
    if steps_done > 0:
        result["total_prompt_ms"] = total_prompt
        result["total_env_ms"] = total_env
        result["prompt_ms_avg"] = total_prompt / steps_done
        result["env_ms_avg"] = total_env / steps_done
    else:
        result["status"] = "error"
        result["errors"].append("no successful steps")

    try:
        ag.close()
    except Exception:
        pass
    return result


def main():
    tee, json_path = setup_logging()
    try:
        cfg = load_config_dicts()

        registered = list_registered_agents()
        unavailable = list_unavailable_agents()
        print(f"🔎 Registered agents: {registered}")
        if unavailable:
            print(f"⚠️  Unavailable agents: {json.dumps(unavailable, indent=2)}")

        # Only test agents with a representative task mapping
        test_types = [a for a in registered if a in REPRESENTATIVE_TASK_FOR_TYPE]
        print(f"🧪 Testing agent types: {test_types}")

        N_STEPS = int(os.environ.get("AGENT_TIME_STEPS", "100"))
        print(f"⏱  Steps per agent: {N_STEPS}")
        print("=" * 60)

        results = []
        for agent_type in test_types:
            print(f"\n—— Timing {agent_type} ———————————————————————————————")
            metrics = time_one_agent(agent_type, cfg, n_steps=N_STEPS)
            results.append(metrics)
            status = metrics["status"]
            pavg = metrics["prompt_ms_avg"]
            eavg = metrics["env_ms_avg"]
            print(f"status={status} | init={metrics['init_ms']:.1f} ms | reset={metrics['reset_ms']:.1f} ms | "
                  f"prompt_avg={pavg:.2f} ms | env_avg={eavg:.2f} ms | steps={metrics['n_steps']}")
            if metrics["errors"]:
                print("errors:")
                for err in metrics["errors"][:3]:
                    print("  -", err)

        # Summary table sorted by env time
        print("\n📊 Summary (sorted by env_ms_avg):")
        ok_results = [r for r in results if r["status"] == "ok" and r["env_ms_avg"] is not None]
        ok_results.sort(key=lambda r: r["env_ms_avg"])  # fastest first
        for r in ok_results:
            print(f"  {r['agent_type']:<18s} env_avg={r['env_ms_avg']:.2f} ms | prompt_avg={r['prompt_ms_avg']:.2f} ms | steps={r['n_steps']}")

        # Dump JSON
        with open(json_path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\n💾 Saved JSON to: {json_path}")

        print("\n🎉 TIME TEST COMPLETED")
    finally:
        tee.close()
        sys.stdout = tee.stdout


if __name__ == "__main__":
    main()


