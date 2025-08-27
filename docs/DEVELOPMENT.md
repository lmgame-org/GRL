# Development Track & Guide

## Current Development Status

### 1. Previous PPO Training Sanity Check
- **Qwen 2.5-0.5B Sokoban PPO Training** ✅ **Completed** (yuxuan)
- **Qwen 2.5-7B Sokoban PPO Training** ✅ **Completed** (mingjia)

## Development Roadmap

### 2. Roadmap (yuxuan, mingjia)

#### 2.1 Core Agents
Location: `agents/*`
- [✅] Handle ad‑hoc message format fixes in `get_llm_prompts()`
- [✅] Abstract base agent class for reusability
- [ ] Move common parts to `base_agent.py` and simplify specific agents

#### 2.2 Rollout
Location: `rollout/sync_multi_turn_rollout.py`
- [✅] Debug early stop logic in multi‑turn rollout
- [✅] Optimize reward computation (loss_mask, reward_mask)
- [ ] Replace `tokenizer.encode()` with `verl_F.tokenize_and_postprocess_data()`

#### 2.3 Training
Location: `trainer/agent_trainer.py`
- [✅] Add hyperparameter for validation agent number
- [✅] Debug `_validate()` vs. mingjia’s ragen implementation
- [✅] Checkpoint saving frequency settings
- [✅] Fix `is_action_valid` metric issue
- [ ] Integrate turn‑based loss mask
- [ ] Add extra metrics and LLM generation logging to Weights & Biases

#### 2.4 Benchmarks & Alignment with Paper
- [✅] Correct unstable validation curve
- [✅] Test general ability from simple Sokoban to large Sokoban
- [✅] Integrate more envs
  - [✅] gsm8k & blocksworld
  - [✅] Tetris
  - [✅] Align env parameters and message printout
  - [✅] Agentic WebShop and BIRD
- [✅] Test general ability across all envs

#### 2.5 Trainers & Extensibility
- [ ] JAX PPO trainer integration ([Tunix Integration Plan](TUNIX_INTEGRATION.md))
  - [✅] write tunix_sync_multi_turn_rollout.py
    - [✅] finish tunix multi turn rollout part
    - [✅] verify the final results ids
  - [✅] integrate it with tunix_agent_trianer.py
  - [ ] test the training workflow in tunix_train.py
    - [✅] draft a runnable tunix multi-turn rl training
    - [✅] wandb metric visualization
    - [ ] validation implementation
      - [✅] draft validation rollout
      - [✅] understand tunix trianing and validtion logic for better integration
      - [✅] solve metric logging problem
    - [ ] try critic model automated surgery
    - [ ] align with hyperparameters
    - [ ] wrap up tunix training code 
      - [✅] critic model building + critic tpu allocation
      - [✅] reward score allocation
      - [✅] prompt ids and completions ids from input ids (pattern analysis)
      - [ ] cpu_offload; fsdp + tp to reduce memory
      - [ ] calculate memory consumption
      - [ ] abstract a uniform yaml config file
- [ ] Abstract the framework to integrate different trainers

#### 2.6 Tooling & Build
- [ ] Implement `uv` installation for faster package management
- [✅] Package as `grl` via `pyproject.toml`
- [ ] Convert `env_setup.sh` into an open and useful script
- [ ] Remove submodule and wrap VERL as a monkey patch

#### 2.7 Advanced Features
- [ ] Vision modality support for multi‑turn RL PPO training
- [ ] SFT (Supervised Fine‑Tuning) trainer
- [ ] Asynchronous multi‑turn rollout system

## Contributing

When working on any of these items:
1. Create a feature branch from main
2. Follow the existing code style and patterns
3. Add appropriate tests and documentation
4. Submit a pull request with clear description of changes

## Notes

- Priority should be given to completing the 7B model performance reproduction
- Codebase improvements should focus on maintainability and performance
- New features should be developed incrementally with proper testing
