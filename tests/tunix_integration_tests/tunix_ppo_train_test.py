#!/usr/bin/env python3
"""
Integration-style tests for grl/tunix_ppo_train.py

Modeled after tests/tunix_integration_tests/rollout_test.py: lightweight, runnable
tests with minimal mocks to validate wiring before full training runs.
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------------
# Helpers: logging and dummy classes
# ---------------------------------------------------------------------------------


def _setup_logging():
  log_dir = Path(__file__).resolve().parent / "test_logs"
  os.makedirs(log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  return log_dir / f"tunix_ppo_train_debug_{ts}.log"


class _DummyMesh:

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False


class _DummyTokenizer:

  def __init__(self):
    self.pad_token_id = 0
    self.eos_token_id = 2
    self.pad_token = "<|pad|>"
    self.eos_token = "<|eos|>"


class _DummyLearner:

  def __init__(
      self,
      rl_cluster,
      ppo_config,
      reward_fns,
      multi_turn_cfg,
      multi_turn_processor,
      multi_turn_validation,
  ):
    self.rl_cluster = rl_cluster
    self.ppo_config = ppo_config
    self.reward_fns = reward_fns
    self.multi_turn_cfg = multi_turn_cfg
    self.multi_turn_processor = multi_turn_processor
    self.multi_turn_validation = multi_turn_validation
    self._trained_steps = 0

  def train(self, dataset):
    # Walk the dataset once to simulate training steps
    for _ in dataset:
      self._trained_steps += 1


class _DummyRLCluster:

  def __init__(self, actor, critic, reference, tokenizer, cluster_config):
    self.actor = actor
    self.critic = critic
    self.reference = reference
    self.tokenizer = tokenizer
    self.cluster_config = cluster_config

  def close(self):
    pass


# ---------------------------------------------------------------------------------
# Import target module
# ---------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"

sys.path.insert(0, str(ROOT))
import grl.tunix_ppo_train as ttp  # noqa: E402


# ---------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------


def test_derive_hparams_from_yaml():
  cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
  derived = ttp.derive_hparams(cfg)
  # basic shape and keys
  for k in [
      "num_ppo_epochs",
      "mini_batch_size",
      "gamma",
      "gae_lambda",
      "beta",
      "epsilon",
      "vf_coef",
      "clip_range_value",
      "entropy_coeff",
      "epsilon_low",
      "epsilon_high",
      "epsilon_c",
      "kl_penalty_method",
      "num_batches",
      "eval_every_n_steps",
      "max_steps",
      "actor_lr",
      "critic_lr",
  ]:
    assert k in derived
  # mesh format [shape, axes]
  assert isinstance(derived["mesh"], list) and len(derived["mesh"]) == 2


def test_get_dataset_behaviour():
  N = 5
  ds = ttp.get_dataset(N, "train")
  assert len(ds) == N
  count = 0
  for row in ds:
    assert isinstance(row, dict)
    count += 1
  assert count == N


def test_dummy_reward_fn_length():
  prompts = ["p1", "p2", "p3"]
  completions = ["c1", "c2", "c3"]
  scores = ttp._dummy_reward_fn(prompts, completions)
  assert isinstance(scores, list) and len(scores) == len(completions)
  assert all(abs(s) < 1e-8 for s in scores)


def test_download_model_weights_mock():
  with patch.object(ttp, "snapshot_download", return_value="/tmp/fake") as sd:
    out = ttp.download_model_weights("some/repo", "/tmp/local")
    sd.assert_called_once()
    assert out == "/tmp/fake"


def test_load_qwen2_from_safetensors():
  with tempfile.TemporaryDirectory() as d:
    # create a sentinel safetensors file
    open(os.path.join(d, "weights.safetensors"), "wb").close()
    with patch.object(
        ttp.params,
        "create_model_from_safe_tensors",
        return_value=SimpleNamespace(name="model"),
    ) as cm:
      m = ttp.load_qwen2_from_safetensors(d, model_config={})
      cm.assert_called_once()
      assert getattr(m, "name", None) == "model"


def test_save_intermediate_state_mock():
  with tempfile.TemporaryDirectory() as d:
    # module split -> (_, state)
    with (
        patch.object(ttp.nnx, "split", return_value=("gdef", {"a": 1})) as sp,
        patch.object(
            ttp.ocp,
            "StandardCheckpointer",
            return_value=SimpleNamespace(save=lambda path, state: None),
        ) as sc,
        patch.object(ttp.time, "sleep", lambda *_args, **_kw: None),
    ):
      # call
      ttp.save_intermediate_state(module=object(), save_dir=d)
      sp.assert_called_once()
      # after first save, the file exists and subsequent call should be a no-op
      ttp.save_intermediate_state(module=object(), save_dir=d)


def test_build_reference_model_from_ckpt_mock():
  dummy_mesh = _DummyMesh()

  def _split(x):
    if x == "abs_qwen2":
      return ("graph_def", None)
    return ("gdef", {"state": 1})

  with (
      patch.object(ttp.jax, "make_mesh", return_value=dummy_mesh),
      patch.object(ttp.nnx, "eval_shape", return_value="abs_qwen2"),
      patch.object(ttp.nnx, "state", return_value={"state": 1}),
      patch.object(ttp.nnx, "get_named_sharding", return_value="shard"),
      patch.object(ttp.jax.tree, "map", return_value={"placed": 1}),
      patch.object(
          ttp.ocp,
          "StandardCheckpointer",
          return_value=SimpleNamespace(
              restore=lambda ckpt_path, target: target
          ),
      ),
      patch.object(ttp.nnx, "split", side_effect=_split),
      patch.object(ttp.nnx, "merge", return_value="merged_model"),
  ):
    qref, mesh, mcfg = ttp.build_reference_model_from_ckpt(
        "/tmp/ckpt", dummy_mesh
    )
    assert qref == "merged_model"
    assert mesh is dummy_mesh
    assert mcfg is not None


def test_clone_module_like_mock():
  dummy_mesh = _DummyMesh()
  with (
      patch.object(ttp.nnx, "eval_shape", return_value="abs_mod"),
      patch.object(ttp.nnx, "split", return_value=("gdef", None)),
      patch.object(ttp.nnx, "state", return_value={"x": 1}),
      patch.object(
          ttp.nnx, "get_named_sharding", return_value="target_sharding"
      ),
      patch.object(ttp.jax.tree, "map", return_value={"x": 1}),
      patch.object(ttp.nnx, "merge", return_value="merged"),
  ):
    out = ttp.clone_module_like("src", model_config={}, mesh=dummy_mesh)
    assert out == "merged"


def test_build_models_and_tokenizer_pipeline_mock():
  dummy_mesh = _DummyMesh()
  with tempfile.TemporaryDirectory() as d:
    # patch internals used by the pipeline
    with (
        patch.object(ttp, "download_model_weights", return_value=d),
        patch.object(
            ttp,
            "load_qwen2_from_safetensors",
            return_value=SimpleNamespace(name="qwen2"),
        ),
        patch.object(ttp, "save_intermediate_state", return_value=None),
        patch.object(
            ttp,
            "build_reference_model_from_ckpt",
            return_value=("qref", dummy_mesh, SimpleNamespace(name="mcfg")),
        ),
        patch.object(ttp, "clone_module_like", return_value="policy"),
        patch.object(ttp, "get_critic_model", return_value="critic"),
        patch.object(ttp, "AutoTokenizer") as AT,
    ):
      AT.from_pretrained.return_value = _DummyTokenizer()
      cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
      derived = ttp.derive_hparams(cfg)
      pol, crit, ref, tok, mesh, mcfg = ttp.build_models_and_tokenizer(
          cfg, derived
      )
      assert pol == "policy" and crit == "critic" and ref == "qref"
      assert isinstance(tok, _DummyTokenizer)
      assert isinstance(mesh, _DummyMesh)
      assert hasattr(mcfg, "name")


def test_build_optimizers_sanity():
  cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
  derived = ttp.derive_hparams(cfg)
  aopt, copt = ttp.build_optimizers(derived)
  assert aopt is not None and copt is not None


def test_build_cluster_config_basic():
  cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
  derived = ttp.derive_hparams(cfg)
  mesh = _DummyMesh()
  tok = _DummyTokenizer()
  cc = ttp.build_cluster_config(mesh, tok, derived, cfg)
  assert hasattr(cc, "training_config")
  assert isinstance(cc.rollout_config, dict)
  assert (
      "TRAIN"
      in {k.name if hasattr(k, "name") else k for k in cc.rollout_config.keys()}
      or len(cc.rollout_config) > 0
  )


def test_build_trainer_wiring():
  cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
  derived = ttp.derive_hparams(cfg)
  rl = _DummyRLCluster(
      actor="p",
      critic="c",
      reference="r",
      tokenizer=_DummyTokenizer(),
      cluster_config=None,
  )
  with patch.object(ttp, "PpoLearnerExp", _DummyLearner):
    trainer = ttp.build_trainer(rl, cfg, derived, cfg)
    assert isinstance(trainer, _DummyLearner)
    assert trainer.ppo_config.num_ppo_epochs == derived["num_ppo_epochs"]


def test_orchestration_smoke_path():
  cfg = OmegaConf.load(str(CONFIGS_DIR / "tunix_base.yaml"))
  derived = ttp.derive_hparams(cfg)
  dataset = ttp.get_dataset(derived["num_batches"], "train")

  dummy_mesh = _DummyMesh()
  with (
      patch.object(
          ttp,
          "build_models_and_tokenizer",
          return_value=("p", "c", "r", _DummyTokenizer(), dummy_mesh, None),
      ),
      patch.object(ttp.rl_cluster_lib, "RLCluster", _DummyRLCluster),
      patch.object(ttp, "PpoLearnerExp", _DummyLearner),
  ):
    pol, crit, ref, tok, mesh, mcfg = ttp.build_models_and_tokenizer(
        cfg, derived
    )
    cc = ttp.build_cluster_config(mesh, tok, derived, cfg)
    with mesh:
      rl = ttp.rl_cluster_lib.RLCluster(
          actor=pol,
          critic=crit,
          reference=ref,
          tokenizer=tok,
          cluster_config=cc,
      )
      trainer = ttp.build_trainer(rl, cfg, derived, cfg)
      trainer.train(dataset)


if __name__ == "__main__":
  log_file = _setup_logging()
  print(f"📝 TUNIX PPO TRAIN DEBUG LOG: {log_file}")
  print("🚀 Starting tunix_ppo_train tests...")

  try:
    print("Test 1: derive_hparams_from_yaml")
    test_derive_hparams_from_yaml()
    print("✅ Passed 1")

    print("Test 2: get_dataset_behaviour")
    test_get_dataset_behaviour()
    print("✅ Passed 2")

    print("Test 3: dummy_reward_fn_length")
    test_dummy_reward_fn_length()
    print("✅ Passed 3")

    print("Test 4: download_model_weights_mock")
    test_download_model_weights_mock()
    print("✅ Passed 4")

    print("Test 5: load_qwen2_from_safetensors")
    test_load_qwen2_from_safetensors()
    print("✅ Passed 5")

    print("Test 6: save_intermediate_state_mock")
    test_save_intermediate_state_mock()
    print("✅ Passed 6")

    print("Test 7: build_reference_model_from_ckpt_mock")
    test_build_reference_model_from_ckpt_mock()
    print("✅ Passed 7")

    print("Test 8: clone_module_like_mock")
    test_clone_module_like_mock()
    print("✅ Passed 8")

    print("Test 9: build_models_and_tokenizer_pipeline_mock")
    test_build_models_and_tokenizer_pipeline_mock()
    print("✅ Passed 9")

    print("Test 10: build_optimizers_sanity")
    test_build_optimizers_sanity()
    print("✅ Passed 10")

    print("Test 11: build_cluster_config_basic")
    test_build_cluster_config_basic()
    print("✅ Passed 11")

    print("Test 12: build_trainer_wiring")
    test_build_trainer_wiring()
    print("✅ Passed 12")

    print("Test 13: orchestration_smoke_path")
    test_orchestration_smoke_path()
    print("✅ Passed 13")

    print("🎉 All tests passed for tunix_ppo_train")
  except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
