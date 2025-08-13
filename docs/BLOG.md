# GRL (Game Reinforcement Learning) 🚀



> **Quick Links**

> - **GitHub Repository:** https://github.com/lmgame-org/GRL

> - **Tutorial:** https://github.com/lmgame-org/GRL/blob/main/docs/TUTORIAL.md

> - **Paper (arXiv):** https://arxiv.org/abs/2505.15146



**TL;DR** 🧪  

GRL (Game Reinforcement Learning)  is an **agent-centric** framework for multi-turn reinforcement learning of LLMs, designed to study **generalization**. While well-suited for game-based tasks, it extends naturally to training and evaluating diverse domains with verifiable rewards—including math, coding, and beyond.  

Experiments show that training on board games such as Sokoban and Tetris can drive cross-game transfer improving planning ability and overall agentic task performance.





---



##GRL is an agent centric design for 


### 1. **Agent-Centric Reinforcement Learning (GRL)**
Our framework treats each *agent unit* as a self-contained rollout manager—controlling the entire lifecycle from task assignment to execution and feedback. This encapsulation is driven by two declarative configs:
- **`agent_config`**: Governs the LLM interaction—defines prompts, reasoning structure, token and turn budgets, action formatting, etc.
- **`env_config`**: Dictates environment behavior—task dynamics, grid sizes, render modes, vocabularies, datasets, and gym-style dynamics.

This separation ensures each agent is **completely modular and self-contained**, which:
- Makes debugging straightforward and localized.
- Enables clean extensibility across diverse environment types.
- Enhances scalability by reducing cross-agent interference and simplifying configuration management.

## GRL vs. verl-agent vs. RAGEN

| Feature / Aspect          | **GRL (Ours)** – Advantage | verl-agent | RAGEN |
|---------------------------|---------------------------|------------|-------|
| **Design Focus**          | **Agent-centric**: each agent unit controls full rollout lifecycle | Gym-style multi-turn rollouts, less explicit agent isolation | Trajectory-level RL, less agent identity focus |
| **Config Structure**      | **Clear split**: `agent_config` (LLM behavior) + `env_config` (environment) | Mixed configs, memory modules, less separation | Unified config, environment-focused |
| **Scalability**           | **High** – modular agents scale cleanly across diverse envs | High throughput, grouped rollouts | Modular but less per-agent isolation |
| **Debugging Ease**        | **Easy** – localized to single agent unit | Possible, but configs less explicit | More global-level tuning required |
| **Cross-Domain Transfer** | **Built-in** – train/validate within isolated agent units | Possible with custom envs | Focused on stochastic env optimization |
| **Customization**         | **High** – plug-and-play new agents/envs | Flexible, but less structured | Flexible, but environment-centric |

(Please check [TUTORIAL.md](https://github.com/lmgame-org/GRL/blob/main/docs/TUTORIAL.md) for further details)


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
