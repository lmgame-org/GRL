import os
import jax.numpy as jnp

def test_tokenizer_and_masks_imports():
    # Basic smoke test to import the example script without executing training
    # Ensures tokenizer assets and pad/eos setup are available
    from grl.train_ppo_multi_turn_example_exp import tokenizer
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

def test_split_alignment_example():
    # Minimal alignment check for convert_multi_rollout_batch invariants
    import numpy as np
    from grl.trainer.tunix_agent_trainer_exp import PpoLearnerExp
    # Fake batch: loss_mask has a single run of ones at the end
    input_ids = np.array([[1,2,3,4,5]], dtype=np.int32)
    loss_mask = np.array([[0,0,1,1]], dtype=np.int32)
    class _B: pass
    b = _B()
    b.input_ids = input_ids
    b.loss_mask = loss_mask
    # Stub trainer object with only convert method usage
    class _Stub:
        def __init__(self):
            self.rl_cluster = type("_RC", (), {"rollout": type("_R", (), {"pad_id": staticmethod(lambda: 0)})})
        convert_multi_rollout_batch = PpoLearnerExp.convert_multi_rollout_batch
    stub = _Stub()
    prompt_ids, completion_ids, prompt_mask, completion_mask, eos_idx = stub.convert_multi_rollout_batch(
        stub, b, pad_value=0, max_prompt_length=0
    )
    # completion should be 2 tokens (ids 4,5), eos_idx=1
    assert int(completion_mask.sum()) == 2
    assert int(eos_idx[0]) == 1

