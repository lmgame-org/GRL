## GRL v2 Development Track

### Purpose & Impact

- Elegant, modular training API
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
- [ ] Extend VERL `PPO_trainer.yaml` to support GRPO training.
- [ ] Integrate Sokoban coding agent into rollout
   - [ ] Draft a minimal training recipe.
   - [ ] Refactor env API and agent API to support GRPO‑style training.
- [ ] Validate end‑to‑end with a small‑scale run and basic metrics.


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
