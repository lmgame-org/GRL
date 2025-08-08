import gymnasium as gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from lmgamerl.agents.agent_utils import all_seed
from .utils import generate_room
from lmgamerl.agents.base_env import BaseEnv

class SokobanEnv(GymSokobanEnv, BaseEnv):
    def __init__(self, config, **kwargs):
        self.config = config
        self.GRID_LOOKUP = self.config.get('grid_lookup', {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"})
        self.ACTION_LOOKUP = self.config.get('action_lookup', {1: "Up", 2: "Down", 3: "Left", 4: "Right"})
        self.search_depth = self.config.get('search_depth', 300)
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)  # Our config uses actions 1-4
        self.render_mode = self.config.get('render_mode', 'text')
        
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.get('dim_room', (6, 6)), 
            max_steps=self.config.get('max_steps', 100),
            num_boxes=self.config.get('num_boxes', 1),
            **kwargs
        )

    def reset(self, seed=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        """
        Take an action in the environment.
        Actions: 1=Up, 2=Down, 3=Left, 4=Right (from our config)
        """
        previous_pos = self.player_position.copy()
        _, reward, done, _ = GymSokobanEnv.step(self, action)
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        info = {"action_is_effective": action_effective, "action_is_valid": True, "success": self.boxes_on_target == self.num_boxes}  
        return next_obs, reward, done, info
        

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        elif render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()
