# GRL (Game Reinforcement Learning) 🚀



> **Quick Links**

> - **GitHub Repository:** https://github.com/lmgame-org/GRL

> - **Tutorial:** https://github.com/lmgame-org/GRL/blob/main/docs/TUTORIAL.md

> - **Paper (arXiv):** https://arxiv.org/abs/2505.15146



**TL;DR** 🧪  

GRL (Game Reinforcement Learning)  is an **agent-centric** framework for multi-turn reinforcement learning of LLMs, designed to study **generalization**. While well-suited for game-based tasks, it extends naturally to training and evaluating diverse domains with verifiable rewards—including math, coding, and beyond.  

Experiments show that training on board games such as Sokoban and Tetris can drive cross-game transfer improving planning ability and overall agentic task performance.




---



## 1. **Agent-Centric Reinforcement Learning Framework**
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



---



Observations from Sokoban and Tetris Training:

1. Same-Game Generalization (Robust within Domain)
We observe improvements in same-game but harder settings, indicating strong in-domain generalization.
Training on Sokoban (6×6) leads to higher performance on the more complex Sokoban (8×8) task, with accuracy improving from 5.5% → 7.4%.
Training on Tetris (1 block type) improves performance on Tetris (2 block types), from 9.9% → 23.1%.
These results suggest that models trained on structured symbolic games can adapt to more challenging variants within the same domain, highlighting their capacity for scaling generalization to increased complexity.

2. Cross-Game Generalization (Symbolic Transfer)
Interestingly, we also find evidence of cross-game transfer. When trained on Sokoban, the model improves on Tetris tasks:
Tetris (1 type): 2.2% → 4.5%
Tetris (2 types): 9.9% → 13.1%
Similarly, when trained on Tetris, the model improves on Sokoban tasks:
Sokoban (6x6): 12.7% -> 15.1%
Sokoban (8x8): 5.5% -> 7.2%
Both Sokoban and Tetris share symbolic 2D ASCII representations and sequential reasoning requirements. The observed transfer suggests that training on one symbolic environment can boost performance on structurally similar symbolic tasks, even across different domains.

3. Blocksworld: Sensitivity to Representation Format
Blocksworld performance demonstrates that representation format matters.
With symbolic formats (2D ASCII tables or 1D lists), training on Sokoban or Tetris consistently improves Blocksworld outcomes.
With natural language text representation, however, gains are inconsistent.
This indicates that game-based symbolic reasoning primarily benefits other symbolic tasks but may not directly transfer to natural language reasoning, underscoring a representational gap between symbolic structures and linguistic abstractions.

4. GSM8K: Neutral Effect on Math Reasoning
For the GSM8K benchmark, training on Sokoban or Tetris yields neutral performance impact.
There can be two explanations:
Domain misalignment – Sokoban and Tetris emphasize spatial reasoning rather than arithmetic or logical deduction.
Well-covered domain – Math is already a well-covered domain in LLM pretraining, so additional training on unrelated symbolic domains provides little incremental value.
This reinforces the view that out-of-domain training benefits most when pretraining is insufficient or when structural overlap exists.

5. WebShop: Multi-Turn Gains but with Instability
For WebShop, which requires multi-turn reasoning and planning, we observe overall improvements after game-based training. This aligns well with the structural properties of board games. However, results are not fully stable:
After Tetris training, run 1 shows a clear improvement, while run 2 shows a decline.
This variability suggests that multi-turn reasoning tasks can benefit from symbolic training, but stability remains a challenge and need further investigation.

Insights:
Game-based training generalizes most effectively when tasks share symbolic or multi-turn structure.
Representation format plays a critical role, with symbolic-to-symbolic transfer being strong, but symbolic-to-text being weaker. 
Domains already well-covered in pretraining (like math) see little gain.
Training instability is a recurring issue, calling for more systematic exploration of curricula and stopping criteria.

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
