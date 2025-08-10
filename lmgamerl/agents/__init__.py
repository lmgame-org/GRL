# Agent Registry System
from typing import Dict, Type, Any
import warnings

# Global agent registry
REGISTERED_AGENTS: Dict[str, Type] = {}
# Keep track of agents that failed to import (missing deps, etc.)
UNAVAILABLE_AGENTS: Dict[str, str] = {}

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
        extra = ""
        if name in UNAVAILABLE_AGENTS:
            extra = f" (agent is unavailable due to import error: {UNAVAILABLE_AGENTS[name]})"
        raise KeyError(
            f"Agent '{name}' not found in registry{extra}. Available agents: {list(REGISTERED_AGENTS.keys())}"
        )
    return REGISTERED_AGENTS[name]

def list_registered_agents() -> list:
    """
    List all registered agent names.
    
    Returns:
        List of registered agent names
    """
    return list(REGISTERED_AGENTS.keys())

def list_unavailable_agents() -> Dict[str, str]:
    """
    Return a dict of agents that failed to import mapped to their error messages.
    """
    return dict(UNAVAILABLE_AGENTS)

# Import agents to trigger registration (fault-tolerant)
def _safe_import(import_fn, agent_key: str):
    try:
        import_fn()
    except Exception as e:
        UNAVAILABLE_AGENTS[agent_key] = str(e)
        warnings.warn(f"Skipping agent '{agent_key}' due to import error: {e}")

_safe_import(lambda: __import__('lmgamerl.agents.sokobanAgent.agent', fromlist=['SokobanAgent']), 'sokobanAgent')
_safe_import(lambda: __import__('lmgamerl.agents.gsm8kAgent.agent', fromlist=['GSM8KAgent']), 'gsm8kAgent')
_safe_import(lambda: __import__('lmgamerl.agents.blocksworldAgent.agent', fromlist=['BlocksworldAgent']), 'blocksworldAgent')
_safe_import(lambda: __import__('lmgamerl.agents.tetrisAgent.agent', fromlist=['TetrisAgent']), 'tetrisAgent')
_safe_import(lambda: __import__('lmgamerl.agents.webshopAgent.agent', fromlist=['WebShopAgent']), 'webshopAgent')
_safe_import(lambda: __import__('lmgamerl.agents.birdAgent.agent', fromlist=['BirdAgent']), 'birdAgent')