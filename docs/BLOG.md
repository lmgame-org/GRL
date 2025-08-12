# LMGame Reinforcement Learning 🚀



> **Quick Links**

> - **GitHub Repository:** https://github.com/lmgame-org/LMGameRL

> - **Tutorial:** https://github.com/lmgame-org/LMGameRL/blob/main/docs/TUTORIAL.md

> - **Paper (arXiv):** https://arxiv.org/abs/2505.15146



**TL;DR** 🧪  

LMGame RL is a framework for multi-turn reinforcement learning training of LLMs, designed to study generalization. Not limited to game-based tasks, it can be applied to train and/or evaluate diverse tasks with verifiable rewards, such as math and coding.



Our experiments show that training on board games such as Sokoban and Tetris can drive cross-game transfer and enhance planning and agentic task performance.



---



## LMGameRL: Lightweight, scalable RL for LLM ability gains



This repo provides a light, scalable workflow to study how RL training improves LLM general ability. To evaluate transferability, run a simple experiment using one script that trains on small 6×6 Sokoban levels and then tests performance on other domains such as Tetris, Blocksworld or GSM8K. Common training and validation hyperparameters (GPUs, model name, training/validation tasks, group numbers/sizes) are edited directly inside the provided script for a fast, frictionless iteration loop.



```bash

source quick_train_qwen_halfb.sh

```



---



## Agent–Environment design for high customization



Agents and environments are defined declaratively in `configs/agents.yaml`. The `agent_config` controls the full LLM interaction lifecycle—system/prompt design, thinking, turn budgets, action formatting—and orchestrates how the model proposes actions and parses answers.



The `env_config` controls the task dynamics and difficulty—grid sizes, steps, datasets, render modes, and vocab/lookups—mapping to the corresponding implementation under `lmgamerl/agents/*`. You can add your own environment or agent by following the base interfaces in `lmgamerl/agents/base_agent.py` and `lmgamerl/agents/base_env.py`, then registering a new keyed entry in `agents.yaml`.



To customize a workflow: create or tweak a task entry (choose an `agent_type`, set `agent_config` and `env_config`), then select it in the script via `TRAINING_TASKS` and `VALIDATION_TASKS`. Typical sizing like training 8×16 and validation 64×1 offers fast, stable signal; groups share seeds within a group and differ across groups, enabling reproducible yet diverse rollouts. Mix multiple validation tasks to probe scaling and cross-domain generalization with minimal overhead.



(Please check [TUTORIAL.md](https://github.com/lmgame-org/LMGameRL/blob/main/docs/TUTORIAL.md) for further details)



---



## Training Results



Our experiments are primarily conducted on the Qwen2.5-7B-Instruct model. We trained on one board game (Sokoban or Tetris) using PPO, and evaluated on cross-game, planning, math, code and agentic tasks. Detailed settings are in our paper ([arXiv:2505.15146](https://arxiv.org/abs/2505.15146)).



Each game is trained twice. Reported values correspond to the step at which the training task (Sokoban or Tetris) achieves its peak performance on the validation set, averaged across the two runs.



---



**Part 1**



|                  | Sokoban 6×6 | Sokoban 8×8 | Tetris (1 type) | Tetris (2 types) | Blocksworld (text) |
|------------------|-------------|-------------|-----------------|------------------|--------------------|
| Qwen2.5-7B-Instruct | 12.7        | 5.5         | 2.2             | 9.9              | 67.3               |
| Train on Sokoban | **26.6**     | **7.4**     | 4.5             | 13.1             | **72.2**           |
| Train on Tetris  | 15.1         | 7.2         | **58.4**        | **23.1**         | 64.7               |



---



**Part 2**



|                  | Blocksworld (1d) | Blocksworld (2d) | GSM8K (1 turn) | GSM8K (5 turns) | WebShop |
|------------------|------------------|------------------|----------------|-----------------|---------|
| Qwen2.5-7B-Instruct | 17.3             | 13.5             | **88.3**       | 94.1            | 9.0     |
| Train on Sokoban | **24.3**          | 17.9             | 87.3           | 93.8            | 15.0    |
| Train on Tetris  | 20.8              | **20.6**         | 89.1           | **94.5**        | **15.8** |


![Examples of observed validation success rate curves](assets/example_validation_success_curves.png)


---



## Insights 🎯



- Improvements from board-game training are more apparent in tasks like cross-game (Sokoban and Tetris), Blocksworld and WebShop that require symbolic planning or multi-turn reasoning.

- In contrast, tasks like GSM8K see minimal gains, likely because the model already performs strongly in math due to extensive pretraining, making it hard to benefit from out-of-domain training.

- The training curves have fluctuations, and in some cases performance collapsed after reaching its peak while training continued. This phenomenon needs further investigation.



In summary, we find training on games can generalize to out-of-domain tasks when:

1. Existing pretraining is insufficient, and/or

2. The task shares structural properties with the games (e.g., symbolic format, multi-turn planning).



---



## Reproduce Training Results



**Sokoban Training Results:**

```bash

Source examples/sokoban_ppo/qwen_7b.sh

```



**Tetris Training Results:**

```bash

Source examples/tetris_ppo/qwen_7b.sh

```
