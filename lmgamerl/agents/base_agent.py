from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random
from lmgamerl.agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput
class BaseAgent:
    """
    Abstract base class for agents. Provides high-level method signatures for agent lifecycle, environment interaction, LLM interface, trajectory management, and rollout collection.
    """
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        """Initialize the agent with configuration and identifiers."""
        # initialize base agent
        self.group_id = group_id
        self.agent_id = agent_id
        self.tag = tag
        self.cur_turn = 0
        if seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        else:
            self.seed = seed
        self.agent_config = config['agent_config']
        self.env_config = config['env_config']

        # handle config hyperparameters
        self.max_turns = self.agent_config.get('max_turns', 1)
        self.max_actions_all_turns = self.agent_config.get('max_actions_all_turns', 1)
        self.max_actions_per_turn = self.agent_config.get('max_actions_per_turn', 1)
        self.max_tokens = self.agent_config.get('max_tokens', 100)
        self.format_penalty = self.agent_config.get('format_penalty', -0.1)
        self.enable_think = self.agent_config.get('enable_think', True)
        self.system_prompt = self.agent_config.get('system_prompt', "You are a helpful AI assistant.")
        self.prompt = self.agent_config.get('prompt', "Please respond appropriately.")
        self.action_separator = self.agent_config.get('action_separator', "||")

        # Define turn prompt template based on enable_think
        if self.enable_think:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: {max_tokens} tokens.\n"""
        else:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: {max_tokens} tokens.\n"""
            
        self.trajectory_history = MultiTurnTrajectory(max_length=self.max_turns)
        self.raw_response_list = []  # Store all raw LLM responses for debugging
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        self.total_actions_consumed = 0
        self.penalty = 0.0  # Track accumulated penalty

    # ─────────────────── LLM INTERFACE ───────────────────
    def get_llm_prompts(self, env_out):
        """Convert environment outputs to LLM prompts following SyncMultiTurnRollout interface."""
        
        # Ensure messages are initialized
        if not hasattr(self, 'messages') or not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt}
            ]
        
        # Calculate actions remaining based on max_actions_all_turns
        actions_remaining = max(0, self.max_actions_all_turns - self.total_actions_consumed)
        
        turn_content = self.turn_prompt_template.format(
            turn_number=self.cur_turn + 1,
            state=env_out.state,
            actions_remaining=actions_remaining,
            max_tokens=self.max_tokens
        )

        turn_msg = {"role": "user", "content": turn_content}

        # In the first turn, we merge the turn_content into the prompt
        if self.cur_turn == 0 and len(self.messages) == 2 and self.messages[1]["role"] == "user":
            self.messages[1]["content"] = self.messages[1]["content"] + "\n" + turn_content
        else:
            reward_msg = f"Reward: \n{env_out.reward}\n"
            turn_msg["content"] =  reward_msg + " " + turn_msg["content"]
            self.messages.append(turn_msg)
        
        # Validate final messages before returning
        if not self.messages:
            # Emergency fallback
            self.messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Please respond appropriately."}
            ]
        
        return self.messages
    
    def parse_llm_response(self, llm_response, enable_think=True):
        """
        Parse model response into processed llm_response and action list.
        Simple parsing that handles enable_think cases and limits actions to max_actions_per_turn.
        
        Args:
            llm_response: Raw LLM response string
            enable_think: Whether to expect <think> tags
            
        Returns:
            Tuple[str, List[str]]: (processed_llm_response, actions_list)
        """
        import re

        if self.agent_config.get('use_think_answer_token', True):
            if enable_think:
                llm_response = '<think>' + llm_response
            else:
                llm_response = '<answer>' + llm_response
        else:
            llm_response = llm_response
        
        # Define pattern based on enable_think
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, llm_response, re.DOTALL)
        
        if not match:
            # No valid pattern found, return original response with empty actions
            processed_response, actions = llm_response, []
        else:
            if enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)
            
            # Clean up special tokens
            special_tokens = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
            for special_token in special_tokens:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            # Parse actions using || separator
            actions = [action.strip() for action in action_content.split(self.action_separator) if action.strip()]
            
            # Limit actions to max_actions_per_turn
            if len(actions) > self.max_actions_per_turn:
                actions = actions[:self.max_actions_per_turn]  # Only the first MAX_ACTIONS actions are kept
                action_content = self.action_separator.join(actions)
            
            # Reconstruct properly formatted response
            if enable_think:
                processed_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
            else:
                processed_response = f"<answer>{action_content}</answer>"
                
        return processed_response, actions

    # ─────────────────── ROLLOUT STATE COLLECTION ───────────────────
    def get_final_rollout_states(self):
        """Get final rollout states for PPO training."""
        history = []
        trajectory_deque = self.trajectory_history.get()
        for traj in trajectory_deque:
            history_entry = {
                'state': traj.state,
                'actions_left': traj.actions_left,
                'actions': traj.actions,
                'reward': traj.reward,
                'info': traj.info,
                'llm_response': traj.llm_response,
                'llm_raw_response': traj.llm_raw_response
            }
            history.append(history_entry)
        
        metrics = {}
        
        success_values = [traj.info.get('success', False) for traj in trajectory_deque]
        metrics[f'{self.tag or "baseAgent"}/success'] = float(any(success_values))
        
        total_actions = sum(len(traj.actions) for traj in trajectory_deque)
        metrics[f'{self.tag or "baseAgent"}/num_actions'] = total_actions
        
        action_is_effective_values = [traj.info.get('action_is_effective', False) for traj in trajectory_deque]
        if action_is_effective_values:
            metrics[f'{self.tag or "baseAgent"}/action_is_effective'] = sum(action_is_effective_values) / len(action_is_effective_values)
        else:
            metrics[f'{self.tag or "baseAgent"}/action_is_effective'] = 0.0
        
        action_is_valid_values = [traj.info.get('action_is_valid', False) for traj in trajectory_deque]
        if action_is_valid_values:
            metrics[f'{self.tag or "baseAgent"}/action_is_valid'] = sum(action_is_valid_values) / len(action_is_valid_values)
        else:
            metrics[f'{self.tag or "baseAgent"}/action_is_valid'] = 1.0
        
        if trajectory_deque:
            last_traj = trajectory_deque[-1]
            if 'metrics' in last_traj.info:
                for key, value in last_traj.info['metrics'].items():
                    metrics[key] = value
        
        row_dict = {
            'env_id': self.agent_id,
            'history': history,
            'group_id': self.group_id,
            'tag': self.tag or 'baseAgent',
            'metrics': metrics,
            'penalty': self.penalty
        }
        
        return row_dict


    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────
    def reset(self, seed=None):
        """Reset agent state for new episode and return initial environment outputs."""
        # Implement group-based seeding following reference implementation
        # Agents within the same group should have the same environment (same seed)
        # Different groups should have different environments (different seeds)
        if seed is None:
            # Generate a unique seed only if no seed provided
            reset_seed = random.randint(0, 1000000)
        else:
            # Use the provided group seed directly - all agents in same group get same seed
            reset_seed = seed
            
        obs = self.env.reset(seed=reset_seed)
        if not obs:
            obs = self.env.render()
        
        self.cur_turn = 0
        
        self.trajectory_history.clear()
        self.raw_response_list = []
        self.total_actions_consumed = 0
        self.penalty = 0.0  # Reset penalty for new episode

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        
        # Return initial environment outputs for the rollout loop
        return EnvOutput(
            truncated=False,
            terminated=False,
            state=obs,
            reward=0.0,
            info={}
        )

    def close(self):
        """Clean up agent resources."""
        if hasattr(self, 'env') and hasattr(self.env, 'close'):
            self.env.close()
    
    def get_messages(self):
        """Get messages for debugging."""
        return self.messages

    # ─────────────────── ENVIRONMENT INTERFACE ───────────────────
    def initialize_env(self):
        """Initialize the environment for the agent."""
        pass


    def get_env_outputs(self, llm_response):
        """Process LLM outputs and get environment outputs."""
        pass
