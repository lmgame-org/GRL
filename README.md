<div align="center">

# LMGame Reinforcement Learning

</div>

<div>
<br>

<div align="center">


[![Github](https://img.shields.io/badge/LMGameRL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lmgame-org/LMGameRL)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://lmgame.org) 
[![arXiv](https://img.shields.io/badge/arXiv-2505.15146-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.15146)
[![X (Twitter)](https://img.shields.io/badge/Follow-HaoAILab-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/haoailab)
[![Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/x3dTHfyRZE)

</div>

</div>

LMGameRL is an open‑source framework that post‑trains LLMs via multi-turn reinforcement learning on games, yielding general gains across diverse benchmarks.

## Release
<strong>[2025/08/13]</strong> We release LMGameRL to reproduce the paper’s results and to demonstrate general gains across benchmarks by post‑training LLMs via reinforcement learning.


## Installation

   ```bash
   # clone the repo
   git clone --recurse-submodules https://github.com/lmgame-org/LMGameRL.git
   cd LMGameRL

   # create a conda environment
   conda create --name lmgamerl python=3.10
   conda activate lmgamerl

   # install all dependencies
   source scripts/install_submodules.sh
   # avoid compiling flash-attn from source
   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   pip install flash-attn==2.8.0.post2 --no-build-isolation
   pip install -e .

   # export environment variables
   export WANDB_API_KEY=your_wandb_api_key
   export WANDB_ENTITY=your_wandb_entity
   export HF_TOKEN=your_huggingface_token
   ```


### Optional: Install Datasets
If you want to reproduce paper results and validate BIRD SQL performance or WebShop full dataset performance:
```bash
source scripts/install_dataset.sh --all
```

## Quick Run

For quick experimentation:
Trains on 6×6 (1‑box) Sokoban and transfers to 8×8 (1-box).

```bash
source quick_train_qwen_halfb.sh
```

## Training Examples

### General gains of LLM ability from game RL training (paper‑reported results)


![Table 4: Model performance on diverse tasks](docs/assets/table4.png)

> **Note:** RL training results may fluctuate relative to reported results, but the overall trend and gains remain consistent.

**Sokoban Agent Training:**
```bash
source examples/sokoban_ppo/qwen_7b.sh
```

**Tetris Agent Training:**
```bash
source examples/tetris_ppo/qwen_7b.sh
```


### Hardware Configuration

The framework is pre‑configured for different GPU setups:

| GPU Type | GPUs | Agent Groups | Group Size | Total Agents | Default Model | Task |
|---|---:|---:|---:|---:|---|---|
| A100 | 1 | 8 | 16 | 128 | Qwen/Qwen2.5-0.5B-Instruct | Sokoban |
| L40 | 1 | 4 | 8 | 32 | Qwen/Qwen2.5-0.5B-Instruct | Sokoban |
| A100 | 8 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Sokoban |
| H200 | 4 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Sokoban |
| A100 | 8 | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct | Tetris |



> **Note:** The framework automatically scales based on available hardware. Adjust parameters in the training scripts for best performance on your setup.

## Supported Games and Agents

- **Sokoban**: Puzzle-solving game requiring spatial reasoning
- **Tetris**: decision‑making and planning
- **GSM8K**: Mathematical reasoning tasks
- **BlocksWorld**: Logical planning and manipulation
- **WebShop**: E‑commerce navigation and decision‑making
- **BIRD**: SQL query generation and database reasoning

## Documentation
- **[Tutorial](docs/TUTORIAL.md)** - Contributing and development workflow
- **[System Design Overview](docs/SYSTEMDESIGN.md)** - Architecture and design principles
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development workflow

## Acknowledgments

Our work is powered by [VERL](https://github.com/volcengine/verl), an open‑source RLHF library, and draws insights from [Ragen](https://github.com/RAGEN-AI/RAGEN).

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
