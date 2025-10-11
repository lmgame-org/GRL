## GRL v2 Development Track

### Purpose & Impact

- Elegant, lite, modular training API
  - Inspired by Tinker’s elegant design; positioned between Tinker (very elegant) and VERL (less elegant).
  - Disaggregate the LLM‑RL training flow into clear blocks with stable interfaces:
    - env API
    - agent API
    - group/multi‑agent API
    - rollout & trajectory sampling
    - training loop (weight update, advantage calculation, loss calculation, etc.)
- Game‑first training recipes
  - Provide robust, reproducible recipes from text RL to vision RL with strong game support.

### Concrete Sub‑Goals

- [ ] GRPO training support for coding‑tool agent
- [ ] SFT (Supervised Fine‑Tuning) training support


### Implementation Plan

#### 1.1 GRPO
- [✅] Extend VERL `PPO_trainer.yaml` to support GRPO training.
    - [✅] Understand GRPO workflow in a theory 
    - [ ] Test GRPO Training
- [ ] Integrate Sokoban coding agent into rollout
   - [ ] Draft a minimal training recipe.
    - [ ] write a universal rollout part by inputing rl dataset
    - [ ] share the same agent_trainer.py 
- [ ] Validate end‑to‑end with a small‑scale run and basic metrics.
- [ ] Add SQLgym enviornment for coding evaluation.

#### 1.2 System Design of Agent and Envs
- [ ] Refactor env API and agent API to support GRPO‑style training.
    - System Design: Env Class, Agent Class, AgentGroupBuilder Class, RLDataset Class

Training Variables:
1) sequence length
2) reward patterns ? 
3) multi step and multi turn formation
4) staged reinforcement learning: exploration -> converge 
5) other rl aglrotihm: grpo, srpo, ppo, dapo and relevant hyperparameters
6) model sizes
7) sft effect
8) single game rl vs multi game rl



### Development & Contribution Style

- Testing
  - Write PyTests alongside each change; keep PRs test‑backed.
  - Directory layout reference: `tests/agent_tests/`, `tests/rollout_tests/`, `tests/sokobanAgent_tests/`, `tests/tetrisAgent_tests/`, `tests/webshopAgent_tests/`, `tests/gsm8kAgent_tests/`, `tests/blocksworldAgent_tests/`, `tests/birdAgent_tests/`, `tests/tunix_integration_tests/`, `tests/tunix_train_examples/`.
  - Quick check: `pytest -q`. Full suite: `bash tests/run_tests.sh`.
  - Conventions: test files as `test_*.py`, test functions as `test_*`.
  - Make tests deterministic and focused; prefer small units and helpful fixtures.

- Code style & formatting
  - Run `./code_style.sh` before committing; use `./code_style.sh --check` to verify no changes are needed.
  - Default mode formats changed/new `*.py`/`*.ipynb` vs a base ref; use `./code_style.sh --all-grl` to format everything under `grl/`.
  - Uses `pyink` with 2‑space indentation; line length is 80 or `pylintrc` `max-line-length` if present.

- Commit/PR checklist
  - Tests added/updated and pass locally (`pytest -q`) or via `bash tests/run_tests.sh`.
  - `./code_style.sh --check` passes.
  - Update docs when changing public APIs or interfaces.

