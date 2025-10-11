#!/usr/bin/env python3
"""
Trainer Test – validates convert_multi_rollout_batch in PpoLearnerExp
using a mocked multi-turn rollout batch.
"""

# standard imports
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# project imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from grl.trainer.tunix_agent_trainer_exp import (
    PpoLearnerExp,
    entropy_from_logits_jax,
    agg_loss_jax,
)  # noqa: E402


class DummyBatch:
    def __init__(self, input_ids, loss_mask):
        self.input_ids = input_ids
        self.loss_mask = loss_mask


def test_convert_multi_rollout_batch():
    # Arrange
    # B=1, L=5. loss_mask shape [B, L-1]=[1,4]
    # 0 => prompt prefix, 1 => completion suffix (starting at first 1)
    input_ids = np.array([[100, 101, 102, 103, 104]], dtype=np.int32)
    loss_mask = np.array([[0, 0, 0, 1]], dtype=np.int32)
    batch = DummyBatch(input_ids=input_ids, loss_mask=loss_mask)

    # Create instance without running heavy __init__
    learner = object.__new__(PpoLearnerExp)

    pad_value = 0
    max_prompt_length = 4  # Pmax

    # Act
    prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx = (
        learner.convert_multi_rollout_batch(
            batch,
            pad_value=pad_value,
            max_prompt_length=max_prompt_length,
        )
    )

    # Visualize
    print("\n[Visualization] convert_multi_rollout_batch")
    print("input_ids:", input_ids.tolist())
    print("loss_mask (0=prompt, 1=completion):", loss_mask.tolist())
    print("prompt_ids:", prompt_ids.tolist())
    print("completion_ids:", completion_ids.tolist())
    print("prompt_mask:", prompt_mask.tolist())
    print("completion_mask:", completion_mask.tolist())
    print("eos_idx:", eos_idx.tolist())

    # Assert
    # Expected prompt: left-padded to length 4 with pad_value
    # First 1 in loss_mask at pos=3 (target for token at input index 3),
    # so completion starts at input_ids[3]. Prompt tokens: input_ids[:3]
    expected_prompt = jnp.array([[pad_value, 100, 101, 102]], dtype=jnp.int32)
    # Expected completion: tokens from index 3 onwards => [103,104]
    expected_completion = jnp.array([[103, 104]], dtype=jnp.int32)
    # Masks
    expected_prompt_mask = (expected_prompt != pad_value).astype(jnp.int32)
    expected_completion_mask = jnp.array([[1, 1]], dtype=jnp.int32)
    # EOS idx is end of completion sequence
    expected_eos_idx = jnp.array([1], dtype=jnp.int32)

    assert jnp.array_equal(prompt_ids, expected_prompt)
    assert jnp.array_equal(completion_ids, expected_completion)
    assert jnp.array_equal(prompt_mask, expected_prompt_mask)
    assert jnp.array_equal(completion_mask, expected_completion_mask)
    assert jnp.array_equal(eos_idx, expected_eos_idx)


def test_convert_multi_rollout_batch_varied():
    # B=2 with different split points, rectangular inputs (no ragged rows)
    input_ids = np.array([
        [5, 6, 7, 8, 9],        # L=5
        [9, 10, 11, 12, 13],    # L=5
    ], dtype=np.int32)
    loss_mask = np.array([
        [0, 1, 1, 1],  # completion starts at index 1 -> [6,7,8,9]
        [0, 0, 0, 1],  # completion starts at index 3 -> [12,13]
    ], dtype=np.int32)

    batch = DummyBatch(input_ids=input_ids, loss_mask=loss_mask)
    learner = object.__new__(PpoLearnerExp)
    pad_value = 0
    max_prompt_length = 3

    prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx = (
        learner.convert_multi_rollout_batch(
            batch,
            pad_value=pad_value,
            max_prompt_length=max_prompt_length,
        )
    )

    # Expected prompts (left-padded to length 3)
    expected_prompt = jnp.array([
        [0, 0, 5],       # first row prompt tokens [:1]
        [9, 10, 11],     # second row prompt tokens [:3]
    ], dtype=jnp.int32)
    # Expected completions (ragged -> keep as Python lists per row)
    expected_completion_row0 = [6, 7, 8, 9]
    expected_completion_row1 = [12, 13]
    # Masks
    expected_prompt_mask = (expected_prompt != pad_value).astype(jnp.int32)
    expected_completion_mask_row0 = [1, 1, 1, 1]
    expected_completion_mask_row1 = [1, 1]
    expected_eos_idx = jnp.array([3, 1], dtype=jnp.int32)

    # Shape for ragged arrays differs; compare per-row
    assert jnp.array_equal(prompt_ids, expected_prompt)
    assert jnp.array_equal(prompt_mask, expected_prompt_mask)
    assert jnp.array_equal(eos_idx, expected_eos_idx)
    # completion_ids/completion_mask are rectangular in our converter (padded to row max)
    # Check prefix matches and mask correctness per row
    assert jnp.array_equal(completion_ids[0, :4], jnp.array(expected_completion_row0, dtype=jnp.int32))
    assert jnp.array_equal(completion_ids[1, :2], jnp.array(expected_completion_row1, dtype=jnp.int32))
    assert jnp.array_equal(completion_mask[0, :4], jnp.array(expected_completion_mask_row0, dtype=jnp.int32))
    assert jnp.array_equal(completion_mask[1, :2], jnp.array(expected_completion_mask_row1, dtype=jnp.int32))

    # Visualization
    print("\n[Visualization] convert_multi_rollout_batch (varied)")
    print("input_ids:", input_ids.tolist())
    print("loss_mask:", loss_mask.tolist())
    print("prompt_ids:\n", prompt_ids.tolist())
    print("completion_ids:\n", completion_ids.tolist())
    print("prompt_mask:\n", prompt_mask.tolist())
    print("completion_mask:\n", completion_mask.tolist())
    print("eos_idx:", eos_idx.tolist())
    print("✅ convert_multi_rollout_batch varied test passed")


def test_entropy_from_logits_uniform_token_mean():
    # Arrange: uniform logits over vocab -> entropy per token = log(V)
    V = 4
    B, T = 2, 3
    logits0 = jnp.zeros((B, T, V), dtype=jnp.float32)
    logits1 = jnp.ones((B, T, V), dtype=jnp.float32)
    completion_mask = jnp.array([[1, 1, 1], [1, 0, 0]], dtype=jnp.int32)

    # Act
    token_entropy0 = entropy_from_logits_jax(logits0)  # shape [B, T]
    ent_token_mean0 = agg_loss_jax(token_entropy0, completion_mask, "token-mean")
    token_entropy1 = entropy_from_logits_jax(logits1)
    ent_token_mean1 = agg_loss_jax(token_entropy1, completion_mask, "token-mean")

    # Assert
    expected = jnp.log(jnp.array(V, dtype=jnp.float32))
    # Visualization
    print("\n[Visualization] entropy_from_logits_jax")
    print("Vocab size:", int(V))
    print("logits0.shape:", tuple(logits0.shape))
    print("completion_mask:\n", completion_mask.tolist())
    print("token_entropy (zeros):\n", token_entropy0.tolist())
    print("token_entropy (ones):\n", token_entropy1.tolist())
    print("expected token_entropy (scalar log(V)):", float(expected))
    print("aggregated/token-mean (zeros):", float(ent_token_mean0))
    print("aggregated/token-mean (ones):", float(ent_token_mean1))
    print("expected aggregated/token-mean:", float(expected))
    # token entropies are all log(V) where masked
    assert jnp.allclose(token_entropy0[completion_mask == 1], expected, atol=1e-6)
    assert jnp.allclose(token_entropy1[completion_mask == 1], expected, atol=1e-6)
    # aggregated token-mean equals log(V)
    assert jnp.allclose(ent_token_mean0, expected, atol=1e-6)
    assert jnp.allclose(ent_token_mean1, expected, atol=1e-6)
    # invariance to additive constant per token position
    assert jnp.allclose(token_entropy0, token_entropy1, atol=1e-6)
    print("✅ entropy_from_logits_jax token-mean test passed")


def test_agg_loss_jax_modes():
    # Arrange: simple loss matrix and mask
    loss_mat = jnp.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=jnp.float32)
    loss_mask = jnp.array([[1, 0, 0], [0, 1, 0]], dtype=jnp.int32)

    # Act
    token_mean = agg_loss_jax(loss_mat, loss_mask, "token-mean")
    seq_mean_token_sum = agg_loss_jax(loss_mat, loss_mask, "seq-mean-token-sum")
    seq_mean_token_mean = agg_loss_jax(loss_mat, loss_mask, "seq-mean-token-mean")
    seq_mean_token_sum_norm = agg_loss_jax(loss_mat, loss_mask, "seq-mean-token-sum-norm")

    # Assert
    # Masked values: [10, 5]
    # token-mean: (10+5)/2 = 7.5
    # seq-mean-token-sum: mean([10, 5]) = 7.5
    # seq-mean-token-mean: mean([10/1, 5/1]) = 7.5
    # seq-mean-token-sum-norm: (10+5)/T with T=3 => 15/3 = 5.0
    print("\n[Visualization] agg_loss_jax")
    print("loss_mat:\n", loss_mat.tolist())
    print("loss_mask:\n", loss_mask.tolist())
    print("token_mean:", float(token_mean))
    print("seq_mean_token_sum:", float(seq_mean_token_sum))
    print("seq_mean_token_mean:", float(seq_mean_token_mean))
    print("seq_mean_token_sum_norm:", float(seq_mean_token_sum_norm))
    print("expected token_mean:", 7.5)
    print("expected seq_mean_token_sum:", 7.5)
    print("expected seq_mean_token_mean:", 7.5)
    print("expected seq_mean_token_sum_norm:", 5.0)
    assert jnp.allclose(token_mean, 7.5, atol=1e-6)
    assert jnp.allclose(seq_mean_token_sum, 7.5, atol=1e-6)
    assert jnp.allclose(seq_mean_token_mean, 7.5, atol=1e-6)
    assert jnp.allclose(seq_mean_token_sum_norm, 5.0, atol=1e-6)
    print("✅ agg_loss_jax modes test passed")


def test_entropy_from_logits_various_distributions():
    # Arrange: test several non-uniform probability distributions
    V = 4
    B, T = 2, 3
    completion_mask = jnp.array([[1, 1, 1], [1, 0, 0]], dtype=jnp.int32)
    distributions = [
        jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32),
        jnp.array([0.7, 0.1, 0.1, 0.1], dtype=jnp.float32),
        jnp.array([0.01, 0.01, 0.01, 0.97], dtype=jnp.float32),
    ]

    for idx, p in enumerate(distributions):
        p = p / jnp.sum(p)
        logits = jnp.log(p)
        logits_bt = jnp.tile(logits, (B, T, 1))

        # Act
        token_entropy = entropy_from_logits_jax(logits_bt)
        ent_token_mean = agg_loss_jax(token_entropy, completion_mask, "token-mean")

        # Expected entropy H(p) = -sum p log p
        expected = -jnp.sum(p * jnp.log(p))

        # Visualization
        print("\n[Visualization] entropy_from_logits_jax (distribution)")
        print(f"case {idx}:")
        print("p:", p.tolist())
        print("logits (log p):", logits.tolist())
        print("token_entropy:\n", token_entropy.tolist())
        print("expected H(p):", float(expected))
        print("aggregated/token-mean:", float(ent_token_mean))

        # Assert both token-wise (on masked positions) and aggregated equality
        masked_vals = token_entropy[completion_mask == 1]
        max_abs_diff = jnp.max(jnp.abs(masked_vals - expected))
        print("max_abs_diff token-wise:", float(max_abs_diff))
        # Allow small numerical tolerance
        assert jnp.allclose(masked_vals, expected, atol=3e-5)
        assert jnp.allclose(ent_token_mean, expected, atol=3e-5)

    print("✅ entropy_from_logits_jax various distributions test passed")


def run_all_tests():
    test_convert_multi_rollout_batch()
    print("✅ convert_multi_rollout_batch basic test passed")
    test_convert_multi_rollout_batch_varied()
    print("✅ convert_multi_rollout_batch varied test passed")
    test_entropy_from_logits_uniform_token_mean()
    test_agg_loss_jax_modes()
    test_entropy_from_logits_various_distributions()
    print("✅ entropy and aggregation tests passed")
    print("\n🎉 All trainer tests passed")


if __name__ == "__main__":
    run_all_tests()


