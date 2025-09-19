import numpy as np
import jax.numpy as jnp
import torch

from grl.trainer.tunix_agent_trainer_exp import entropy_from_logits_jax


def _torch_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
  pd = torch.nn.functional.softmax(logits, dim=-1)
  return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


def test_entropy_from_logits_matches_torch():
  rng = np.random.default_rng(123)
  # Batch, Time, Vocab sizes to test
  shapes = [
    (1, 1, 4),
    (2, 3, 5),
    (3, 5, 7),
  ]
  for (B, T, V) in shapes:
    logits_np = rng.normal(loc=0.0, scale=2.0, size=(B, T, V)).astype(np.float32)

    # JAX path
    jax_ent = entropy_from_logits_jax(jnp.array(logits_np))
    jax_ent_np = np.array(jax_ent)

    # Torch reference
    torch_logits = torch.tensor(logits_np, dtype=torch.float32)
    torch_ent = _torch_entropy_from_logits(torch_logits).detach().cpu().numpy()

    # Shape checks
    assert jax_ent_np.shape == (B, T)
    assert torch_ent.shape == (B, T)
    # Numerical closeness
    assert np.allclose(jax_ent_np, torch_ent, atol=1e-5, rtol=1e-5)
    # Finite
    assert np.isfinite(jax_ent_np).all()


def test_entropy_uniform_logits_is_log_vocab():
  B, T, V = 2, 4, 11
  # Uniform logits → uniform probs → entropy = log(V)
  logits_np = np.zeros((B, T, V), dtype=np.float32)
  jax_ent = entropy_from_logits_jax(jnp.array(logits_np))
  jax_ent_np = np.array(jax_ent)
  expected = np.log(V).astype(np.float32)
  assert np.allclose(jax_ent_np, expected, atol=1e-6)

