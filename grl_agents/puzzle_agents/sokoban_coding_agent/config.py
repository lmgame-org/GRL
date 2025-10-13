import os
import sys

# Resolve absolute path to the repository root: up four levels from this file
REPO_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
# Place per-agent coding work directories under <repo_root>/workspace
# The per-agent subfolder pattern is handled in SokobanCodingAgent:
#   workspace/group_{group_id}/agent_{agent_id}_{seed}
workspace_absolute_path = os.path.join(REPO_ROOT_DIR, "workspace")

# Inline configuration for sokobanCodingAgent_6_6_dim_1_box
# Replaces YAML-based config and pulls prompts from prompts module
try:
    # Prefer external prompts if available
    from external.SokobanCodingAgent.agent.prompts import (
        system_prompt as sokoban_system_prompt,
        prompt as sokoban_user_prompt,
    )
except Exception:
    # Fallback: attempt relative import if external package layout differs
    try:
        from .prompts import (
            system_prompt as sokoban_system_prompt,
            prompt as sokoban_user_prompt,
        )
    except Exception as _e:
        # Last resort: minimal defaults to keep tests runnable
        sokoban_system_prompt = "You are a Sokoban coding agent."
        sokoban_user_prompt = "Plan and produce actions using the configured separator."


def get_sokoban_coding_agent_config():
    """
    Return the configuration dict equivalent to `sokobanCodingAgent_6_6_dim_1_box` in config.yaml,
    but with system_prompt and prompt replaced by the ones in agent.prompts.
    """
    return {
        "agent_type": "sokobanCodingAgent",
        "agent_config": {
            "system_prompt": sokoban_system_prompt,
            "prompt": sokoban_user_prompt,
            # Enable/disable tool-use protocol (function-call tools)
            "tool_use": True,
            "workspace_path": workspace_absolute_path,
            "enable_think": False,
            "max_tokens": 10000,
            "max_turns": 1,
            "max_actions_per_turn": 100,
            "max_actions_all_turns": 100,
            "max_steps": 10,
            "format_penalty": -0.1,
            "action_separator": "||",
        },
        "env_config": {
            "dim_room": [8, 8],
            "num_boxes": 2,
            "max_steps": 100,
            "search_depth": 100,
            "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            "grid_vocab": {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"},
            "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
            "render_mode": "text",
        },
    }


