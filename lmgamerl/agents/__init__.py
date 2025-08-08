# Agent Registry System
from typing import Dict, Type, Any

# Global agent registry
REGISTERED_AGENTS: Dict[str, Type] = {}

def register_agent(name: str):
    """
    Decorator to register an agent class with a given name.
    
    Args:
        name: The name to register the agent under
        
    Returns:
        Decorator function that registers the agent class
    """
    def decorator(cls):
        if name in REGISTERED_AGENTS and REGISTERED_AGENTS[name] != cls:
            raise ValueError(f"Agent {name} has already been registered: {REGISTERED_AGENTS[name]} vs {cls}")
        REGISTERED_AGENTS[name] = cls
        return cls
    return decorator

def get_agent_cls(name: str) -> Type:
    """
    Get agent class by name from registry.
    
    Args:
        name: Name of the registered agent
        
    Returns:
        Agent class
        
    Raises:
        KeyError: If agent name is not registered
    """
    if name not in REGISTERED_AGENTS:
        raise KeyError(f"Agent '{name}' not found in registry. Available agents: {list(REGISTERED_AGENTS.keys())}")
    return REGISTERED_AGENTS[name]

def list_registered_agents() -> list:
    """
    List all registered agent names.
    
    Returns:
        List of registered agent names
    """
    return list(REGISTERED_AGENTS.keys())

# Import agents to trigger registration
from lmgamerl.agents.sokobanAgent.agent import SokobanAgent 
from lmgamerl.agents.gsm8kAgent.agent import GSM8KAgent
from lmgamerl.agents.blocksworldAgent.agent import BlocksworldAgent
from lmgamerl.agents.tetrisAgent.agent import TetrisAgent
from lmgamerl.agents.webshopAgent.agent import WebShopAgent
from lmgamerl.agents.birdAgent.agent import BirdAgent