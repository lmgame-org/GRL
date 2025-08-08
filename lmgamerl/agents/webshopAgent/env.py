import sys
import os
from pathlib import Path
import sys, subprocess, importlib.util

# Add the external webshop-minimal package to the Python path
project_root = Path(__file__).parent.parent.parent
webshop_path = project_root / "external" / "webshop-minimal"
sys.path.insert(0, str(webshop_path))

from webshop_minimal import WebAgentTextEnv
from typing import Optional, Union
from lmgamerl.agents.agent_utils import all_seed
import random
import string
from webshop_minimal.utils import (
    init_basedir,
    DEFAULT_FILE_PATH,
)
from lmgamerl.agents.base_env import BaseEnv

class WebShopEnv(WebAgentTextEnv, BaseEnv):
    def __init__(self, config, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
        self.config = config
        self.observation_mode = self.config.get('observation_mode', 'text')
        self.file_path = DEFAULT_FILE_PATH
        self.server = self.config.get('server', None)
        self.filter_goals = self.config.get('filter_goals', None)
        self.limit_goals = self.config.get('limit_goals', -1)
        self.num_products = self.config.get('num_products', None)
        self.human_goals = self.config.get('human_goals', False)
        self.show_attrs = self.config.get('show_attrs', False)
        self.dataset_size = self.config.get('dataset_size', 'small')
        self.render_cache = None

        if self.server == "None":
            self.server = None
        if self.filter_goals == "None":
            self.filter_goals = None
        if self.num_products == "None":
            self.num_products = None

        init_basedir(self.dataset_size)

        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            dataset_size=self.dataset_size,
            **kwargs
        )

    def reset(self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None) -> any:
        """
        Reset the environment and return the initial observation.

        Args:
            session (str|int|None): The new session ID.
            instruction_text (str|None): Optional new instruction text.

        Returns:
            The initial observation.
        """
        if session is None:
            with all_seed(seed):
                session = ''.join(random.choices(string.ascii_lowercase, k=10))
        # print(f"Resetting with session: {session}")
        obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
        self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
        # print(f"observation: {self.observation}")
        return obs

    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, done, and info.
        """
        state, reward, done, info = WebAgentTextEnv.step(self, action)
        self.prepare_render_cache(self.observation)
        info = {"action_is_effective": tuple(self.get_available_actions()) == ('click[back to search]', 'click[< prev]', 'click[next >]'), "action_is_valid": True, "success": done}
        return self.observation, reward, done, info

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        available_actions = self.get_available_actions()
        self.render_cache = observation + "\n" + "Available actions: " + ", ".join(available_actions)

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        """
        orig_available_actions = WebAgentTextEnv.get_available_actions(self)
        available_actions = []

        if orig_available_actions['has_search_bar']:
            available_actions.append('search[<content>]')

        for clickable in orig_available_actions['clickables']:
            if clickable != 'search':
                available_actions.append(f'click[{clickable}]')
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        return available_actions