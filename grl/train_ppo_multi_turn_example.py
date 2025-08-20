"""
Script version of the Jupyter notebook (tunix_ppo_gsm8k_example.ipynb),
with cells preserved in order. Jupyter magics and shell commands are
commented out for Python execution.
"""

# ============================== Cell 0 ==============================
from tunix.models.qwen2 import params
from tunix.models.qwen2 import model
from flax import nnx
from huggingface_hub import snapshot_download
from etils import epath

MODEL_CP_PATH = "/home/vanitas/lmgame_projects/GRL/qwen_models"
repo_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Download safetensors locally (to MODEL_CP_PATH)
downloaded_path = snapshot_download(
    repo_id=repo_id,
    local_dir=MODEL_CP_PATH,
    allow_patterns=["*.safetensors", "*.json"],
)
print("Files downloaded to:", downloaded_path)

# Prefer the actual directory where files landed
model_dir = str(downloaded_path)

config = model.ModelConfig.qwen2_5_0_5_b()
# Only attempt load if .safetensors exist
if list(epath.Path(model_dir).expanduser().glob("*.safetensors")):
  qwen2 = params.create_model_from_safe_tensors(model_dir, config)
else:
  raise ValueError(f"No safetensors found in {model_dir}")
# nnx.display(qwen2)


# ============================== Cell 1 ==============================
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)


# ============================== Cell 2 ==============================
# The following were Jupyter shell/magic commands; commented for script use.
# !pip install -q tensorflow
# !pip install -q tensorboardX
# !pip install -q grain
# !pip install -q git+https://github.com/google/tunix
# !pip install -q git+https://github.com/google/qwix

# !pip uninstall -q -y flax
# !pip install -q git+https://github.com/google/flax.git

# !pip install -q datasets
# !pip install -q tensorflow_datasets


# ============================== Cell 3 ==============================
import functools
import gc
import os
from pprint import pprint
import re
import time

from flax import nnx as _nnx  # avoid shadowing, though nnx already imported
import grain
import humanize
import jax
import jax.numpy as jnp
# import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.ppo.ppo_learner import PpoConfig, PpoLearner
from tunix.sft import metrics_logger
from pathlib import Path
from jax_smi import initialise_tracking
initialise_tracking()


# ============================== Cell 4 ==============================
# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 1.0

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH = [(1, 2), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50

# ====== PPO ======
# PPO hyperparameters (defaults from tunix.tunix.rl.ppo.ppo_learner.PpoConfig)
NUM_PPO_EPOCHS = 4
MINI_BATCH_SIZE = 1
GAMMA = 1.0
GAE_LAMBDA = 0.95
BETA = 0.04
EPSILON = 0.2
VF_COEF = 0.1
CLIP_RANGE_VALUE = 0.2
# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 100

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES  * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/intermediate_ckpt/"
CKPT_DIR = "/home/vanitas/lmgame_projects/GRL/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


# ============================== Cell 5 ==============================
# ----- Utility functions -----
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

# ----- Data preprocessing -----

reasoning_start = ""
reasoning_end = ""
solution_start = ""
solution_end = ""


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """user
{system_prompt}

{question}
model"""

def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=x["question"].decode("utf-8"),
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return dataset

dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[
    :NUM_TEST_BATCHES
]

len(train_dataset), len(val_dataset) if val_dataset is not None else 0, len(
    test_dataset
)

for ele in train_dataset[:1]:
  pprint(ele)


# ============================== Cell 6 ==============================
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(qwen2)
checkpoint_path = os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state")
if not os.path.exists(checkpoint_path):
  checkpointer.save(os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state"), state)
time.sleep(60)
del qwen2
del state
gc.collect()


# ============================== Cell 7 ==============================
# ----- load models -----
def get_ref_model(ckpt_path):
  mesh = jax.make_mesh(*MESH)
  model_config = model.ModelConfig.qwen2_5_0_5_b()
  abs_qwen2: nnx.Module = nnx.eval_shape(
      lambda: model.Qwen2(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_qwen2)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_qwen2)
  qwen2_ref = nnx.merge(graph_def, restored_params)
  return qwen2_ref, mesh, model_config



# Reference model
qwen2_ref, mesh, model_config = get_ref_model(
    ckpt_path=os.path.join(Path(INTERMEDIATE_CKPT_DIR), "state")
)
# nnx.display(qwen2_ref)



# Policy model (use original, no LoRA)
policy_qwen2 = qwen2_ref
# nnx.display(policy_qwen2)


# Critic model (required for PPO): initialize as a fresh Qwen2 and load
# the same weights as the reference to start
class Qwen2CriticWithScoreHead(nnx.Module):
  def __init__(self, backbone: nnx.Module, rngs: nnx.Rngs):
    self.backbone = backbone
    hidden_dim = getattr(self.backbone.config, 'embed_dim')
    self.score = nnx.Linear(in_features=hidden_dim, out_features=1, use_bias=False, rngs=rngs)

  def __call__(self, input_tokens, positions, cache, attention_mask, **kwargs):
    _ = self.backbone(
        input_tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden = nnx.pop(self.backbone, nnx.Intermediate)['all_hidden_states'].value
    if isinstance(hidden, (list, tuple)):
      hidden = hidden[-1]
    score = self.score(hidden)
    return score

def _make_critic_from_reference(_model_config, _ref_model: nnx.Module) -> nnx.Module:
  abs_mod: nnx.Module = nnx.eval_shape(
      lambda: model.Qwen2(_model_config, rngs=nnx.Rngs(params=0))
  )
  graph_def, _ = nnx.split(abs_mod)
  ref_state = nnx.state(_ref_model)
  backbone = nnx.merge(graph_def, ref_state)
  return Qwen2CriticWithScoreHead(backbone, rngs=nnx.Rngs(params=0))

critic_qwen2 = _make_critic_from_reference(model_config, qwen2_ref)


# ============================== Cell 8 ==============================
# ----- Define reward functions -----
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)

def match_format_exactly(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores


def match_format_approximately(prompts, completions, **kargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, **kargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")

def check_numbers(prompts, completions, answer, **kargs):
  question = kargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores


# ============================== Cell 9 ==============================
# ----- Evaluate -----

def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      total_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output


def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return


# Use HF tokenizer already loaded earlier (AutoTokenizer.from_pretrained)
# and the original Qwen2 policy model (no LoRA)
qwen_tokenizer = tokenizer
# rollout_sampler = sampler_lib.Sampler(
#     transformer=policy_qwen2,
#     tokenizer=qwen_tokenizer,
#     cache_config=sampler_lib.CacheConfig(
#         cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
#         num_layers=model_config.num_layers,
#         num_kv_heads=model_config.num_kv_heads,
#         head_dim=model_config.head_dim,
#     ),
# )


# (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
#     test_dataset,
#     rollout_sampler,
#     **GENERATION_CONFIGS["greedy"],
# )
# print(
#     f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
#     f" {format_accuracy=}%"
# )


# ============================== Cell 10 ==============================
# ----- Training -----

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo", flush_every_n_steps=20
)


# Logs (Jupyter tensorboard magics commented out)
# %load_ext tensorboard
# %tensorboard --logdir /home/vanitas/lmgame_projects/GRL/content/tmp/tensorboard/ppo --port=0

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.CRITIC: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=1,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

ppo_config = PpoConfig(
    num_ppo_epochs=NUM_PPO_EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    beta=BETA,
    epsilon=EPSILON,
    vf_coef=VF_COEF,
    clip_range_value=CLIP_RANGE_VALUE,
)

# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=policy_qwen2,
    critic=critic_qwen2,
    reference=qwen2_ref,
    tokenizer=qwen_tokenizer,
    cluster_config=cluster_config,
)

# PPO Trainer
ppo_trainer = PpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    ppo_config=ppo_config,
)

with mesh:
    ppo_trainer.train(dataset)


# ============================== Cell 11 ==============================
# ----- Final Evaluation -----
trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(policy_qwen2, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    policy_qwen2,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(policy_qwen2, nnx.LoRAParam),
        trained_lora_params,
    ),
)

sampler = sampler_lib.Sampler(
    transformer=policy_qwen2,
    tokenizer=qwen_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)


(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)

