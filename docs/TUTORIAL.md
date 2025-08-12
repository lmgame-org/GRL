## GRL — Customizing Training (Quick Start & Configuration)

This guide shows how to quickly run training and how to customize everything from basic runtime knobs to fine‑grained agent, environment, and PPO settings.

### Quick start

- Make sure dependencies are installed and you can access your target model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`).
- From the project root, run the training script: `./quick_train_qwen_halfb.sh`.

That’s it for the first run. Next, tailor the run by editing a few variables directly inside the script.

## Basic knobs (edit inside `quick_train_qwen_halfb.sh`)

Open `quick_train_qwen_halfb.sh` and adjust the variables at the top of the file. These are the most commonly tuned parameters:

- **CUDA_VISIBLE_DEVICES**: Which GPUs to use (e.g., `"0"`, `"0,1"`).
- **AGENT_GROUP_NUM**: Number of training groups. Groups run in parallel and each group shares a single random seed.
- **AGENT_GROUP_SIZE**: Number of agents per group. Total training instances per rollout step = `AGENT_GROUP_NUM × AGENT_GROUP_SIZE`.
- **TRAINING_TASKS**: Comma‑separated list of task names defined in `configs/agents.yaml`.
- **VALIDATION_TASKS**: Comma‑separated list of validation task names (must match the lengths of validation group lists below).
- **VALIDATION_AGENT_GROUP_NUM**: Comma‑separated per‑validation‑task group counts.
- **VALIDATION_AGENT_GROUP_SIZE**: Comma‑separated per‑validation‑task group sizes.
- **N_GPUS_PER_NODE**: Number of GPUs per node used by the trainer.
- **MODEL_PATH**: Model identifier or local path (e.g., `Qwen/Qwen2.5-0.5B-Instruct`).
- Optional labels: **PROJECT_NAME**, **EXPERIMENT_NAME**.

Examples (edit values directly inside the script):

- **Train with 8×16**: set `AGENT_GROUP_NUM=8`, `AGENT_GROUP_SIZE=16`.
- **Validate with 64×1**: set `VALIDATION_AGENT_GROUP_NUM="64"`, `VALIDATION_AGENT_GROUP_SIZE="1"`.
- **Multiple validation tasks**: e.g., `VALIDATION_TASKS="simpleSokobanAgent,largeSokobanAgent"`, then set `VALIDATION_AGENT_GROUP_NUM="64,64"` and `VALIDATION_AGENT_GROUP_SIZE="1,1"` (same length and order as tasks).
- **Switch model**: set `MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"` (or any compatible model you have access to).
- **Choose GPUs**: set `CUDA_VISIBLE_DEVICES="0,1"` (ensure `N_GPUS_PER_NODE` is consistent with how many GPUs you intend to use).

Notes:

- **Random seeds**: the same group shares the same seed; different groups get different seeds.
- Validation group arrays must align with `VALIDATION_TASKS` one‑to‑one.

## High‑freedom task design in `configs/agents.yaml`

Tasks are defined by a pair of sections:

- **agent_config**: Defines the task lifecycle and agent behavior (system/prompt, turn limits, formatting, action separators, etc.).
- **env_config**: Defines environment hyperparameters (difficulty, sizes, time limits, vocab lookups, dataset paths, render mode, etc.).

Create your own task by adding a new entry keyed by a unique name. Example (trimmed):

```yaml
mySokobanTiny:
  agent_type: "sokobanAgent"
  agent_config:
    system_prompt: "You are a helpful AI assistant that solves Sokoban puzzles step by step."
    prompt: "You are solving the Sokoban puzzle... <answer>Right || Up</answer>"
    enable_think: true
    max_turns: 6
    max_actions_per_turn: 5
    max_actions_all_turns: 10
    action_separator: "||"

  env_config:
    dim_room: [6, 6]
    num_boxes: 1
    max_steps: 100
    render_mode: "text"
```

Then reference `mySokobanTiny` inside `TRAINING_TASKS` or `VALIDATION_TASKS` in your script. You can mix different agents (e.g., `sokobanAgent`, `tetrisAgent`, `blocksworldAgent`, `gsm8kAgent`, `webshopAgent`, `birdAgent`) and provide distinct `env_config` for each.

### Integrate your own agent or environment

To add a new agent/env implementation:

1. Implement your logic following the base interfaces in `grl/agents/base_agent.py` and `grl/agents/base_env.py`.
2. Add a new keyed entry in `configs/agents.yaml` with your `agent_type`, `agent_config`, and `env_config`.
3. Use that key in `TRAINING_TASKS` / `VALIDATION_TASKS` inside the script.

This design gives you high freedom to customize both the agent’s lifecycle and the environment’s dynamics without changing the training loop.

## Fine‑grained training hyperparameters in `configs/base.yaml`

For deeper control beyond the script’s convenience knobs, edit `configs/base.yaml`:

- **rollout**: truncation method, rollout filtering (`rollout_filter_ratio`, `rollout_filter_type`), reward normalization (`grouping`, `method`), validation settings.
- **data**: `max_prompt_length`, `max_response_length`, `train_batch_size`.
- **algorithm**: PPO/GAE settings (`gamma`, `lam`, `adv_estimator`, KL penalty and coefficient).
- **actor_rollout_ref**: model path, whether to use reference model, entropy coefficient, KL settings, clipping ranges, micro‑batch sizes, rollout engine options.
- **critic**: critic model path and optimizer settings.
- **trainer**: total steps, validation cadence (`validation_steps`, `val_before_train`), logging, test frequency, save frequency, resume behavior, GPUs per node.
- **gpu_memory_utilization** and **max_model_len** under rollout engine if needed.
- **model_path** here acts as a project default; the script can override it at runtime.

Tip: Keep defaults in `base.yaml` for reproducibility, and only override per‑run values in the script when experimenting.

Shortcuts for visualization and CPU parallelism:

- Set `rollout.show_tqdm: True` to see progress bars for concurrent agent steps (prompts/tokenize/env/agents).
- Optional knobs (0 = auto): `rollout.num_prompt_threads`, `rollout.num_env_threads`, `rollout.num_init_threads`.
- Example:
  ```yaml
  rollout:
    show_tqdm: True
    num_prompt_threads: 0
    num_env_threads: 0
    num_init_threads: 0
  ```

## Minimal workflow recap

1. Edit `quick_train_qwen_halfb.sh` variables for tasks, model, GPUs, and group sizes.
2. Optionally add/modify tasks in `configs/agents.yaml` to change agent behavior or environment difficulty.
3. Optionally tune training/rollout details in `configs/base.yaml` for PPO and system behavior.
4. Run `./quick_train_qwen_halfb.sh`.

You now have a sharp, easy workflow: quick edits in the script for day‑to‑day runs, and high‑freedom customization via `agents.yaml` and `base.yaml` when you need finer control.


