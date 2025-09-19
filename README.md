<div align="center">

# GRL: Game Reinforcement Learning for Post‑training LLMs

<em>Game Reinforcement Learning (GRL) for post‑training large language models</em>

</div>

<div>
<br>

<div align="center">


[![Github](https://img.shields.io/badge/GRL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lmgame-org/GRL)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://lmgame.org) 
[![Blog](https://img.shields.io/badge/Blog-1DA1F2?style=for-the-badge&logo=rss&logoColor=white)](https://lmgame.org/#/blog/grl)
[![arXiv](https://img.shields.io/badge/arXiv-2505.15146-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.15146)
[![X (Twitter)](https://img.shields.io/badge/Follow-HaoAILab-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/haoailab)
[![Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/x3dTHfyRZE)

</div>

</div>

GRL (Game Reinforcement Learning) is an open‑source framework that post‑trains LLMs via multi‑turn reinforcement learning on games, yielding general gains across diverse benchmarks.

## News
<strong>[2025/09/29]</strong> 🚀 Tunix integration: PPO multi‑turn training now runs on TPU via JAX, with a Sokoban PPO training example. For more details, see [Tunix](https://github.com/google/tunix), a JAX‑native LLM post‑training library with TPU support.

<strong>[2025/08/27]</strong> 📢 We release GRL to reproduce the paper’s results and to demonstrate general gains across benchmarks by post‑training LLMs via reinforcement learning. Read the blog post [here](https://lmgame.org/#/blog/grl).

## 📖 Table of Contents

- [News](#news)
- [Installation](#installation)
  - [Submodule Installation](#submodule-installation)
    - [Tunix Installation](#tunix-installation-only)
    - [Verl Installation](#verl-installation-only)
    - [WebShop Installation](#webshop-installation-only)
  - [Optional: Install Datasets](#optional-install-datasets)
- [Training Examples](#training-examples)
  - [Tunix Quick Test](#tunix-quick-test)
  - [Reproduce Training Results (Verl)](#reproduce-training-results-verl)
- [Supported Games and Agents](#supported-games-and-agents)
- [Hardware Configuration](#hardware-configuration)
  - [GPU Configurations (Torch + VERL)](#gpu-configurations-torch--verl)
  - [TPU Configurations (JAX + Tunix)](#tpu-configurations-jax--tunix)
- [Documentation](#documentation)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [License](#license)

## Installation

   ```bash
   # clone the repo
   git clone --recurse-submodules https://github.com/lmgame-org/GRL.git
   cd GRL

   # create a conda environment
   conda create --name grl python=3.11
   conda activate grl

   # Submodule installation
   bash scripts/install_submodules.sh --all
   # Or only: --verl (GPU/VERL), --tunix (TPU/JAX), or --webshop

   # Torch + FlashAttention (Linux + CUDA)
   # Required for VERL (GPU) workflows; skip for TPU-only Tunix runs
   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   pip install flash-attn==2.8.0.post2 --no-build-isolation

   # install GRL
   pip install -e .

   # export environment variables
   export WANDB_API_KEY=your_wandb_api_key
   export WANDB_ENTITY=your_wandb_entity
   export HF_TOKEN=your_huggingface_token
   ```

### Submodule Installation

#### Tunix Installation (only)
Use this if you plan to run TPU/JAX with Tunix only:
```bash
bash scripts/install_submodules.sh --tunix
```

#### Verl Installation (only)
Use this if you plan to run GPU/PyTorch with VERL only. Torch and FlashAttention are required on Linux + CUDA:
```bash
bash scripts/install_submodules.sh --verl
# Torch + FlashAttention (required for VERL GPU workflows)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

#### WebShop Installation (only)
Install WebShop tooling and prerequisites only:
```bash
bash scripts/install_submodules.sh --webshop
```


### Optional: Install Datasets
If you want to reproduce paper results and validate BIRD SQL performance or WebShop full dataset performance:
```bash
bash scripts/install_dataset.sh --all
```

## Training Examples

### Tunix Quick Test

Quickly run an end‑to‑end multi‑turn PPO rollout + training loop with Tunix (Qwen2.5‑0.5B‑Instruct on Sokoban). This uses minimal defaults and logs metrics to W&B.

#### Run the quick test (defaults to Qwen2.5‑0.5B; supports 4 TPU v4 with mesh (2,2))
```bash
bash tunix_quick_training_example.sh
```

#### Adjust training hyperparameters (tunix_base.yaml)

Edit `configs/tunix_base.yaml` to freely tune training without touching code. Key sections:

- rollout: agent grouping, validation set, filtering, reward normalization
- ppo: PPO knobs (epochs, minibatch, gamma, lambda, entropy, clip ratios, kl method)
- training: optimizer (lr, betas, weight_decay), grad_accum, eval cadence, checkpointing
- rollout_runtime: generation length and sampling (temperature, top_p/top_k)
- model.repo_id: base model to download

Notes:
- Set `training.max_steps` or `training.eval_every_n_steps` to positive integers to force values; use `-1` to let the script compute defaults.
- The script composes `tunix_base.yaml` with `configs/agents.yaml` via defaults and prints the merged configuration at startup.

### Reproduce Training Results (Verl)

Uses Verl (PyTorch) on GPU.

Train on 6×6 (1‑box) Sokoban and evaluate transferability to Tetris, Blocksworld, and GSM8K.

```bash
bash verl_quick_training_example.sh
```

### General gains of LLM ability from game RL training (paper‑reported results)


![Table 4: Model performance on diverse tasks](docs/assets/table4.png)

### Expected Observed validation success rate curves (examples)


![Examples of observed validation success rate curves](docs/assets/example_validation_success_curves.png)


> **Note:** RL training results may fluctuate relative to reported results, but the overall trend and gains remain consistent.

**Sokoban Agent Training:**
```bash
bash examples/sokoban_ppo/qwen_7b.sh
```

**Tetris Agent Training:**
```bash
bash examples/tetris_ppo/qwen_7b.sh
```

> **Note:** BirdAgent may wait on SQLite file readiness or locks; heavy SQL can stall rollouts and prolong validation. 

### Hardware Configuration

GRL supports both GPU and TPU training backends:

- GPU (Torch + VERL): PyTorch-based training on NVIDIA GPUs via VERL.
- TPU (JAX + Tunix): JAX-based training on Google TPU via Tunix.

### GPU Configurations (Torch + VERL)

| GPU Type | GPUs | Agent Groups | Group Size | Total Agents | Default Model | Task |
|---|---:|---:|---:|---:|---|---|
| A100 | 1 | 8 | 16 | 128 | Qwen/Qwen2.5-0.5B-Instruct | Sokoban |
| L40 | 1 | 4 | 8 | 32 | Qwen/Qwen2.5-0.5B-Instruct | Sokoban |
| A100 | 8 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Sokoban |
| H200 | 4 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Sokoban |
| A100 | 8 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Tetris |



### TPU Configurations (JAX + Tunix)

| TPU Type | Chips | Mesh | Agent Groups | Group Size | Total Agents | Default Model | Task |
|---|---:|---|---:|---:|---:|---|---|
| TPU v4 | 4 | (2,2) | 8 | 16 | 128 | Qwen/Qwen2.5-0.5B-Instruct | Sokoban |


> **Note:** The framework automatically scales based on available hardware. Adjust parameters in the training scripts for best performance on your setup.

## Supported Games and Agents

- **Sokoban**: Puzzle‑solving requiring spatial reasoning (agent: `sokobanAgent`)
- **Tetris**: Decision‑making and planning (agent: `tetrisAgent`)
- **GSM8K**: Grade‑school math reasoning (agent: `gsm8kAgent`)
- **Blocksworld**: Logical planning and manipulation (agent: `blocksworldAgent`)
- **WebShop**: E‑commerce navigation and decision‑making (agent: `webshopAgent`)
- **BIRD (SQL)**: SQL query generation and database reasoning (agent: `birdAgent`)
- **AMC 2023**: Competition math problems from AMC 2023 (agent: `amc23Agent`)
- **AIME 2024**: Competition math problems from AIME 2024 (agent: `aime24Agent`)
- **AIME 2025**: Competition math problems from AIME 2025 (agent: `aime25Agent`)
- **Minerva Math**: Advanced math reasoning dataset (agent: `minervamathAgent`)
- **Math500**: Math word‑problem benchmark (agent: `math500Agent`)

## Documentation
- **[Tutorial](docs/TUTORIAL.md)** - Contributing and development workflow
- **[System Design Overview](docs/SYSTEMDESIGN.md)** - Architecture and design principles
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development workflow

## Acknowledgments

We gratefully acknowledge [Tunix](https://github.com/google/tunix), a JAX‑native LLM post‑training library whose TPU support and JAX‑first techniques enabled us to achieve scalable multi‑turn PPO training on TPU with JAX.

Our work is also powered by [VERL](https://github.com/volcengine/verl), and we draw valuable insights from [RAGEN](https://github.com/RAGEN-AI/RAGEN) that informed how we train multi‑turn PPO in our experiments.

## Citation
If you find this repository helpful, please kindly cite:
```
@article{hu2025lmgame,
  title={lmgame-Bench: How Good are LLMs at Playing Games?},
  author={Hu, Lanxiang and Huo, Mingjia and Zhang, Yuxuan and Yu, Haoyang and Xing, Eric P and Stoica, Ion and Rosing, Tajana and Jin, Haojian and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.15146},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
