# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from lmgamerl.agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput, debug_printout_in_env_output
from lmgamerl.agents.base_agent import BaseAgent
from lmgamerl.agents.webshopAgent.env import WebShopEnv
from lmgamerl.agents import register_agent

# ─────────────────── WEBSHOP AGENT ───────────────────
@register_agent("webshopAgent")
class WebShopAgent(BaseAgent):
    """
    WebShop agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        super().__init__(config, group_id, agent_id, seed, tag)
        self.initialize_env()

    def initialize_env(self):
        """Initialize the WebShop environment."""
        self.env = WebShopEnv(self.env_config)

    # ─────────────────── ENV INTERACTION ───────────────────
    def get_env_outputs(self, llm_response: str):
        """
        Parse the model’s reply, send the (single) numeric answer to WebShopEnv,
        and package the resulting EnvOutput + trajectory bookkeeping.
        """
        llm_raw_response = llm_response

        # Store raw response for debugging
        self.raw_response_list.append(llm_raw_response)
       
        self.cur_turn += 1

        processed_llm_response, actions = self.parse_llm_response(llm_raw_response, enable_think=self.enable_think)

        self.messages.append({"role": "assistant", "content": processed_llm_response})

        obs = self.env.render()
        total_reward = 0
        done = False
        info = {}  # Initialize info dictionary
        valid_actions = []
        invalid_actions = []

        for action_str in actions:
            action_str_clean = action_str.strip()
            obs, reward, done, step_info = self.env.step(action_str_clean)
            if step_info['action_is_valid']:
                valid_actions.append(action_str)
            else:
                invalid_actions.append(action_str)
            total_reward += reward
            info.update(step_info)
            if done:
                break
        
        if len(valid_actions) == 0 or len(actions) == 0 or len(valid_actions) != len(actions):
            self.penalty += self.format_penalty

        self.total_actions_consumed += len(actions)

        actions_left = max(0, self.max_actions_all_turns - self.total_actions_consumed)

        if self.cur_turn >= self.max_turns or self.total_actions_consumed >= self.max_actions_all_turns:
            done = True

        self.trajectory_history.add(SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions= actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response
        ))

        # debug_printout_in_env_output(self.messages, actions, self.tag)


        return EnvOutput(
            truncated=done,
            terminated=done,
            state=obs,
            reward=total_reward,
            info=info,
        )