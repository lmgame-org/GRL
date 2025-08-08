"""LMGameRL - Multi-Turn PPO Training System with AgentTrainer for Language Model Game Reinforcement Learning."""

__version__ = "0.1.0"

# Import main components
try:
    from .train import main, run_ppo
except ImportError:
    # Fallback for development/testing
    pass

try:
    from .trainer.agent_trainer import AgentTrainer
except ImportError:
    # Fallback for development/testing
    pass

__all__ = [
    "main",
    "run_ppo", 
    "AgentTrainer",
]