# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from lmgamerl.agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput, debug_printout_in_env_output
from lmgamerl.agents.base_agent import BaseAgent
from lmgamerl.agents.sokobanAgent.env import SokobanEnv
from lmgamerl.agents import register_agent

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("sokobanAgent")
class SokobanAgent(BaseAgent):
    """
    Sokoban agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        super().__init__(config, group_id, agent_id, seed, tag)
        self.prompt = self._build_enhanced_prompt(self.prompt)
        self.initialize_env()

    def _build_enhanced_prompt(self, base_prompt):
        """Build enhanced prompt with environment info and emphatic format instructions."""
        enhanced_prompt = base_prompt
        
        if self.env_config.get("grid_vocab"):
            symbols = [f"{k}: {v}" for k, v in self.env_config["grid_vocab"].items()]
            grid_vocab = f"\nThe meaning of each symbol in the state is:\n {', '.join(symbols)}"
            enhanced_prompt += grid_vocab
        
        if self.env_config.get("action_lookup"):
            actions = list(self.env_config["action_lookup"].values())
            action_lookup_str = "\nYour available actions are:\n" + ", ".join(actions)
            enhanced_prompt += action_lookup_str

        enhanced_prompt += f"\nYou can make up to {self.max_actions_all_turns} actions, and each action is separated by '{self.action_separator}'."
        return enhanced_prompt

    def initialize_env(self):
        """Initialize the Sokoban environment."""
        self.env = SokobanEnv(self.env_config)


    def get_env_outputs(self, llm_response):
        """Process LLM outputs and get environment outputs."""
        llm_raw_response = llm_response

        # Store raw response for debugging
        self.raw_response_list.append(llm_raw_response)
       
        self.cur_turn += 1

        processed_llm_response, actions = self.parse_llm_response(llm_raw_response, enable_think=self.enable_think)

        self.messages.append({"role": "assistant", "content": processed_llm_response})

        obs = self.env.render()
        total_reward = 0
        done = False
        executed_actions = []
        info = {}  # Initialize info dictionary
        
        action_lookup_reverse = {v: k for k, v in self.env_config['action_lookup'].items()}
        # Create case-insensitive lookup for better fault tolerance
        action_lookup_reverse_lower = {v.lower(): k for k, v in self.env_config['action_lookup'].items()}
        
        valid_actions = []
        invalid_actions = []
        
        for action_str in actions:
            try:
                action_str_clean = action_str.strip()
                
                # First try exact match
                if action_str_clean in action_lookup_reverse:
                    action = action_lookup_reverse[action_str_clean]
                    # Validate action is in expected range
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
                # Then try case-insensitive match
                elif action_str_clean.lower() in action_lookup_reverse_lower:
                    action = action_lookup_reverse_lower[action_str_clean.lower()]
                    # Validate action is in expected range
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
                else:
                    # Try parsing as integer
                    action = int(action_str_clean)
                    # Validate numeric action is in expected range  
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
            except (ValueError, KeyError, TypeError) as e:
                invalid_actions.append(action_str)
                continue
        
        # Apply penalty for invalid actions OR no actions extracted
        if len(actions) == 0 or invalid_actions or len(valid_actions) != len(actions):
            self.penalty += self.format_penalty
        
        # Execute valid actions with fault tolerance
        for action in valid_actions:
            try:
                obs, reward, done, step_info = self.env.step(action)
                total_reward += reward
                executed_actions.append(action)
                info.update(step_info)  # Update info with step info
                if done:
                    break
            except Exception as e:
                # Handle any environment step errors
                print(f"Warning: Agent {self.agent_id} step failed for action {action}: {e}")
                # Continue with next action instead of crashing
                continue
        
        # Update total actions consumed
        self.total_actions_consumed += len(executed_actions)
        
        # Calculate actions left based on max_actions_all_turns
        actions_left = max(0, self.max_actions_all_turns - self.total_actions_consumed)
        
        # Check if done due to max turns or max actions
        if self.cur_turn >= self.max_turns or self.total_actions_consumed >= self.max_actions_all_turns:
            done = True
        
        self.trajectory_history.add(SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions=executed_actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response
        ))
        
        return EnvOutput(
            truncated=done,
            terminated=done,
            state=obs,
            reward=total_reward,
            info=info
        )
