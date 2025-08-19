#!/usr/bin/env python3
"""
Tunix SyncMultiTurnRollout Tests (unit-style)

Covers key functions except rollout() and generate_sequences():
 - get_batch_llm_prompts
 - get_batch_env_outputs (with mocked LLM responses)
 - decode helpers (tokens_to_text, decode_llm_responses)
 - get_masks_and_scores and _normalize_score_tensor
 - build_rollout_batch
 - filter_rollout_batch

Follows the testing/logging style from sync_multi_turn_rollout_test.py.
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

import numpy as np
import jax.numpy as jnp

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer  # noqa: E402
from grl.rollout.tunix_sync_multi_turn_rollout import SyncMultiTurnRollout  # noqa: E402


def load_config():
    """
    Load configuration from configs/base.yaml and configs/agents.yaml
    (same approach as existing sync tests)
    """
    config_dir = project_root / "configs"
    with open(config_dir / "base.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    with open(config_dir / "agents.yaml", "r") as f:
        agents_config = yaml.safe_load(f)
    config = {**base_config, **agents_config}
    return config


def create_config_object(config_dict):
    """Create attribute-accessible config wrapper (mirrors sync tests)."""
    class Config:
        def __init__(self, d):
            self._original_dict = d
            for k, v in d.items():
                if isinstance(k, str) and k.isidentifier():
                    setattr(self, k, Config(v) if isinstance(v, dict) else v)

        def get(self, key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            return self._original_dict.get(key, default)

        def __contains__(self, key):
            return hasattr(self, key) or key in self._original_dict

        def __getitem__(self, key):
            if hasattr(self, key):
                attr_value = getattr(self, key)
                if isinstance(attr_value, Config):
                    return attr_value._original_dict
                return attr_value
            return self._original_dict[key]

        def __getattr__(self, key):
            if key.startswith('_'):
                raise AttributeError
            return self._original_dict.get(key, None)

    return Config(config_dict)


def create_real_tokenizer():
    """Load a real tokenizer and ensure pad token is set; batch_decode used for decoding."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_logging():
    """Setup logging to stream outputs to test_logs directory (mirrors sync test style)."""
    import os
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"tunix_sync_multi_turn_rollout_debug_{timestamp}.log")

    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout

        def write(self, data):
            self.file.write(data)
            self.file.flush()
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

        def close(self):
            self.file.close()

    tee = Tee(log_file)
    sys.stdout = tee

    print(f"📝 Tunix SyncMultiTurnRollout DEBUG Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 70)

    return tee


def test_get_batch_llm_prompts_and_env_outputs():
    """
    - Initialize rollout and agents
    - Build batch prompts and validate types/shapes (list[str])
    - Mock LLM responses (list[str])
    - Feed to get_batch_env_outputs and validate updated env_outs length and done_mask dtype
    """
    print("🔍 Testing get_batch_llm_prompts and get_batch_env_outputs...")

    config = load_config()
    cfg = create_config_object(config)
    tokenizer = create_real_tokenizer()

    class DummyCluster:
        pass

    rollout = SyncMultiTurnRollout(
        rl_cluster=DummyCluster(), cfg=cfg, tokenizer=tokenizer, processor=None
    )

    # Ensure initial env_outs
    rollout._reset_batch_agents()

    # 1) get_batch_llm_prompts
    llm_prompts = rollout.get_batch_llm_prompts(rollout.env_outs)
    print(f"llm_prompts: type={type(llm_prompts)}, len={len(llm_prompts)}")
    assert isinstance(llm_prompts, list)
    assert all(isinstance(p, str) for p in llm_prompts)
    assert len(llm_prompts) == rollout.total_agent_num

    # 2) get_batch_env_outputs with mocked responses (simple echoes)
    mocked_responses = ["<answer>Right || Down</answer>" for _ in range(len(llm_prompts))]
    updated_env_outs = rollout.get_batch_env_outputs(mocked_responses)
    print(f"updated_env_outs: len={len(updated_env_outs)}")
    print(f"done_mask dtype: {rollout.done_mask.dtype}, shape: {rollout.done_mask.shape}")
    assert isinstance(updated_env_outs, list)
    assert len(updated_env_outs) == rollout.total_agent_num
    assert hasattr(rollout, "done_mask") and rollout.done_mask.shape[0] == rollout.total_agent_num


def test_decode_helpers():
    """
    - tokens_to_text: decode token id arrays → strings
    - decode_llm_responses: supports List[str], object with .text, object with .tokens
    """
    print("🔍 Testing decode helpers (tokens_to_text, decode_llm_responses)...")

    tokenizer = create_real_tokenizer()

    class DummyCluster:
        pass

    cfg = create_config_object(load_config())
    rollout = SyncMultiTurnRollout(rl_cluster=DummyCluster(), cfg=cfg, tokenizer=tokenizer, processor=None)
    rollout._reset_batch_agents()

    # tokens_to_text
    texts = ["hello", "world"]
    enc = tokenizer(texts, return_tensors=None, padding=True)
    token_array = np.array(enc["input_ids"]) if isinstance(enc, dict) else np.array(enc.input_ids)
    decoded = rollout.tokens_to_text(token_array)
    print(f"decoded (tokens_to_text): {decoded}")
    assert isinstance(decoded, list) and len(decoded) == 2

    # decode_llm_responses: raw List[str]
    raw_list = ["a", "b"]
    out1 = rollout.decode_llm_responses(raw_list)
    print(f"decode_llm_responses(raw list): {out1}")
    assert out1 == raw_list

    # decode_llm_responses: object with .text
    class ObjWithText:
        def __init__(self, text):
            self.text = text
    out2 = rollout.decode_llm_responses(ObjWithText(["x", "y"]))
    print(f"decode_llm_responses(.text): {out2}")
    assert out2 == ["x", "y"]

    # decode_llm_responses: object with .tokens
    class ObjWithTokens:
        def __init__(self, tokens):
            self.tokens = tokens
    out3 = rollout.decode_llm_responses(ObjWithTokens(token_array))
    print(f"decode_llm_responses(.tokens): {out3}")
    assert isinstance(out3, list) and len(out3) == 2


def test_build_and_filter_rollout_batch():
    """
    - After initialization, collect final states
    - Build rollout batch (RolloutBatch)
    - Print and validate shapes: input_ids [N,L], loss_mask [N,L-1], reward_scores [N,L-1]
    - Run filter_rollout_batch and validate shapes reduce or stay the same depending on ratio
    """
    print("🔍 Testing build_rollout_batch and filter_rollout_batch...")

    config = load_config()
    cfg = create_config_object(config)
    tokenizer = create_real_tokenizer()

    class DummyCluster:
        pass

    rollout = SyncMultiTurnRollout(
        rl_cluster=DummyCluster(), cfg=cfg, tokenizer=tokenizer, processor=None
    )

    rollout._reset_batch_agents()

    # Collect agent final states (without full rollout, minimal trajectories exist)
    final_states = rollout._collect_final_rollout_states()
    assert len(final_states) == rollout.total_agent_num

    # Build batch
    batch = rollout.build_rollout_batch(final_states)
    print(
        f"Batch types: input_ids={type(batch.input_ids)}, loss_mask={type(batch.loss_mask)}, reward_scores={type(batch.reward_scores)}"
    )
    print(
        f"Shapes: input_ids={np.shape(batch.input_ids)}, loss_mask={np.shape(batch.loss_mask)}, reward_scores={np.shape(batch.reward_scores)}"
    )

    N, L = batch.input_ids.shape
    assert batch.loss_mask.shape == (N, L - 1)
    assert batch.reward_scores.shape == (N, L - 1)

    # Filter batch (support both method names for robustness)
    if hasattr(rollout, "filter_rollout_batch"):
        filtered, metrics = rollout.filter_rollout_batch(batch)
    else:
        filtered, metrics = rollout.filter_rollout(batch)

    print(
        f"Filtered shapes: input_ids={np.shape(filtered.input_ids)}, loss_mask={np.shape(filtered.loss_mask)}, reward_scores={np.shape(filtered.reward_scores)}"
    )
    print(f"Metrics keys: {list(metrics.keys())}")

    Nf, Lf = filtered.input_ids.shape
    assert filtered.loss_mask.shape == (Nf, Lf - 1)
    assert filtered.reward_scores.shape == (Nf, Lf - 1)
    assert isinstance(metrics, dict)

    # If ratio < 1, expect shrink to selected_groups * group_size
    ratio = cfg.rollout.rollout_filter_ratio
    if ratio < 1:
        group_num = cfg.rollout.agent_group_num[0]
        group_size = cfg.rollout.agent_group_size[0]
        selected = int(ratio * group_num)
        expected = selected * group_size
        print(f"Expected filtered N: {expected} (selected_groups={selected} * group_size={group_size})")
        assert Nf == expected


def test_masks_and_scores_and_normalization():
    """
    - Build input_ids from agent messages via chat template
    - Call get_masks_and_scores and verify shapes [N,L-1]
    - Normalize via _normalize_score_tensor (identity by default) and verify last column increases by penalty
    """
    print("🔍 Testing get_masks_and_scores and _normalize_score_tensor...")

    cfg = create_config_object(load_config())
    tokenizer = create_real_tokenizer()

    class DummyCluster:
        pass

    rollout = SyncMultiTurnRollout(rl_cluster=DummyCluster(), cfg=cfg, tokenizer=tokenizer, processor=None)
    rollout._reset_batch_agents()

    # Prepare chat-rendered inputs from each agent's messages
    llm_input_texts = []
    for agent in rollout.agents:
        messages = agent.get_messages()
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            prompt_text = "System error in chat template"
        llm_input_texts.append(prompt_text)

    enc = tokenizer(llm_input_texts, return_tensors=None, padding=True, padding_side="left", truncation=False)
    input_ids = np.array(enc["input_ids"]) if isinstance(enc, dict) else np.array(enc.input_ids)

    # Simulated per-turn rewards from env history
    final_states = rollout._collect_final_rollout_states()
    scores = [[step['reward'] for step in env_output['history']] for env_output in final_states]

    loss_mask, score_tensor, response_mask = rollout.get_masks_and_scores(jnp.array(input_ids), scores, use_turn_scores=cfg.rollout.use_turn_scores)
    print(f"loss_mask shape: {loss_mask.shape}, score_tensor shape: {score_tensor.shape}, response_mask shape: {response_mask.shape}")
    N, L = input_ids.shape
    assert loss_mask.shape == (N, L - 1)
    assert score_tensor.shape == (N, L - 1)

    # Normalize (identity by default): last column += penalty
    score_norm = rollout._normalize_score_tensor(score_tensor, final_states)
    print(f"normalized score_tensor shape: {score_norm.shape}")
    assert score_norm.shape == (N, L - 1)

    # If identity, check difference equals penalty at last column
    if cfg.rollout.reward_normalization.method == "identity" and not cfg.rollout.use_turn_scores:
        before = np.array(score_tensor)[:, -1]
        after = np.array(score_norm)[:, -1]
        penalties = np.array([env_output.get("penalty", 0.0) for env_output in final_states])
        diff = after - before
        print(f"mean |diff - penalty|: {np.mean(np.abs(diff - penalties)):.6f}")
        assert np.allclose(diff, penalties, atol=1e-5)


def test_lifecycle_reset_and_close():
    """Basic lifecycle: reset() and close() should run without errors."""
    print("🔍 Testing lifecycle reset() and close()...")

    cfg = create_config_object(load_config())
    tokenizer = create_real_tokenizer()

    class DummyCluster:
        pass

    rollout = SyncMultiTurnRollout(rl_cluster=DummyCluster(), cfg=cfg, tokenizer=tokenizer, processor=None)
    rollout.reset()
    assert rollout.env_outs is not None and len(rollout.env_outs) == rollout.total_agent_num
    rollout.close()


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    try:
        print("🚀 Starting Tunix SyncMultiTurnRollout DEBUG Tests...")
        print()

        print("Test 1: get_batch_llm_prompts and get_batch_env_outputs")
        test_get_batch_llm_prompts_and_env_outputs()
        print()

        print("Test 2: decode helpers")
        test_decode_helpers()
        print()

        print("Test 3: masks/scores and normalization")
        test_masks_and_scores_and_normalization()
        print()

        print("Test 4: build and filter rollout batch")
        test_build_and_filter_rollout_batch()
        print()

        print("Test 5: lifecycle reset and close")
        test_lifecycle_reset_and_close()
        print()

        print("=" * 70)
        print("🎉 All Tunix SyncMultiTurnRollout tests completed!")
        print(f"✅ Test completed at {datetime.now()}")
    finally:
        try:
            tee.close()
            sys.stdout = tee.stdout
        except Exception:
            pass


