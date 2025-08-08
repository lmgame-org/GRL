import gymnasium as gym
import numpy as np
import copy
import json
import subprocess
from io import StringIO
import re
import os
from collections import defaultdict
from pathlib import Path   
from lmgamerl.agents.base_env import BaseEnv

class BlocksworldEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, config, **kwargs):
        self.config = config
        self.num_blocks = self.config.get('num_blocks', 3)
        self.render_mode = self.config.get('render_mode', 'text')
        self.all_states = json.load(open(Path(__file__).with_suffix("").parent / "states" / f"blocksworld-{self.num_blocks}.json"))
        self.blocks_ids = [i+1 for i in range(self.num_blocks)]
        
        # Define action space: [block_to_move, destination]
        # block_to_move: 1 to num_blocks
        # destination: 0 (table) or 1 to num_blocks (on top of another block)
        self.ACTION_SPACE = gym.spaces.Tuple([
            gym.spaces.Discrete(self.num_blocks + 1, start=1),  # 1 to num_blocks
            gym.spaces.Discrete(self.num_blocks + 1)   # 0 to num_blocks
        ])

    def integer_to_action(self,int_action):
    # From an integer returns the encoded format for an action
    # [block to move, destination]
        ret = []
        ret.append(int(int_action/(self.num_blocks+1)))
        ret.append(int_action%(self.num_blocks+1))
        #print ('\nDecoding action' + str(ret))
        return ret
        
    def action_to_integer(self,action):
    # From an encoded action [block to move, destination] 
    # returns a simple integer
        ret = action[0]*(self.num_blocks-1) + action[1]
        #print ('\Encoding action' + str(ret))
        return ret        

    def reset(self, seed=None):
        """Resets the environment to an initial state and returns the initial state and goal."""
        self.state, self.goal = self.all_states[seed%len(self.all_states)]
        self.steps = 0
        self.episode_total_reward = 0
        return self.render()

    def extract_action(self, action):
        # now action is a string like "(move 2 to 0)"
        # we need to extract the action from the string
        try:
            action = action.strip()
            action = action.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("move", "").replace("to", "").replace("#", "0").strip()
            action = action.split()
            return int(action[0]), int(action[1])
        except Exception as e:
            return None, None
            
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        
        Args:
            action: The action to take [block_to_move, destination]
            block_to_move: 1 to numBlocks
            destination: 0 (table) or 1 to numBlocks (on top of another block)
            
        Returns:
            tuple:
                - observation (object): Agent's observation of the current environment
                - reward (float): Amount of reward returned after previous action
                - done (bool): Whether the episode has ended
                - info (dict): Contains auxiliary diagnostic information
        """
        block_to_move, destination = self.extract_action(action)
        # self.extract_numbers(action)

        info = {"action_is_effective": False, "action_is_valid": False, "success": False}

        if block_to_move is None or destination is None:
            return self.render(), -5, False, info
    
        info['action_is_effective'] = True

        # Check if the action is valid
        if block_to_move not in range(1, self.num_blocks + 1) or \
           destination not in range(0, self.num_blocks + 1):
            return self.render(), -1, False, info
        
        # Check if block is the same as the destination
        if block_to_move == destination:
            return self.render(), -1, False, info
        
        # Check if the block to move has any blocks on top of it
        if any(pos == block_to_move for pos in self.state):
            return self.render(), -1, False, info
            
        # Check if the destination has any blocks on top of it
        if destination != 0 and any(pos == destination for pos in self.state):
            return self.render(), -1, False, info
        
        info['action_is_valid'] = True
        
        # Apply the move
        new_state = self.state.copy()
        new_state[block_to_move - 1] = destination
        
        # Calculate reward
        reward = -1  # Default reward for valid action
        done = False
        
        # Check if goal state is reached
        if np.array_equal(new_state, self.goal):
            reward = 10
            done = True
            info['success'] = True
            
        # Update state
        self.state = new_state
        self.steps += 1
            
        self.episode_total_reward += reward
        return self.render(), reward, done, info
        
    def describe_state(self, state):
        desc = []
        for i, v in enumerate(state):
            block = f"Block {i+1}"
            if v == 0:
                desc.append(f"{block} is on the table.")
            else:
                desc.append(f"{block} is on top of Block {v}.")
        return ' '.join(desc)

    def get_state_graphic(self, state):
        """Draws a simple blocksworld graphic from the list state."""
        from collections import defaultdict

        n = len(state)
        # Build stacks: each block points to the one below it
        above = defaultdict(list)  # key: block it's on -> list of blocks on it
        for block, on in enumerate(state, 1):
            above[on].append(block)
        
        # Build table stacks
        stacks = []
        for block in above[0]:
            stack = [block]
            while stack[-1] in above:
                next_blocks = above[stack[-1]]
                if next_blocks:
                    stack.append(next_blocks[0])
                else:
                    break
            stacks.append(stack)

        # Find max height to align printing
        max_height = max(len(stack) for stack in stacks)

        # Generate lines from top to bottom
        lines = []
        for level in reversed(range(max_height)):
            line = ""
            for stack in stacks:
                if level < len(stack):
                    line += " _  "
                else:
                    line += "    "
            lines.append(line.rstrip())

            line = ""
            for stack in stacks:
                if level < len(stack):
                    line += f"|{stack[level]}| "
                else:
                    line += "    "
            lines.append(line.rstrip())

            line = ""
            for stack in stacks:
                if level < len(stack):
                    line += " ¯  "
                else:
                    line += "    "
            lines.append(line.rstrip())

        return lines
    
    def get_state_graphic_simple(self, state):
        n = len(state)
        above = defaultdict(list)
        for block, on in enumerate(state, 1):
            above[on].append(block)
    
        # Build table stacks
        stacks = []
        for block in above[0]:
            stack = [block]
            while above.get(stack[-1]):
                next_block = above[stack[-1]][0]  # Always pick first block
                stack.append(next_block)
            stacks.append(stack)

        # Find max height
        max_height = max(len(stack) for stack in stacks)

        # Build the graphic top-down
        lines = []
        for level in range(max_height - 1, -1, -1):
            line = ""
            for stack in stacks:
                if level < len(stack):
                    line += str(stack[level])
                else:
                    line += "_"
            lines.append(line)

        # Add the table line
        # lines.append("#" * len(stacks))

        return lines

    def get_state_line(self, state):
        return f"[{', '.join(str(block) for block in state)}]"

    def render(self):
        """Get the current state as a string representation"""
        state_str = []
        if self.render_mode == '1d':
            # Add current state graphical representation
            state_str.append("\nCurrent representation:")
            state_str.append(self.get_state_line(self.state))
            
            # Add goal state graphical representation
            state_str.append("\nGoal representation:")
            state_str.append(self.get_state_line(self.goal))
            
        elif '2d' in self.render_mode:
            if 'sparse' in self.render_mode:
                graphic_func = self.get_state_graphic
            elif 'compact' in self.render_mode:
                graphic_func = self.get_state_graphic_simple
            
            # Add current state graphical representation
            state_str.append("\nGraphical representation (Current):")
            for line in graphic_func(self.state):
                state_str.append(line)
            
            # Add goal state graphical representation
            state_str.append("\nGraphical representation (Goal):")
            for line in graphic_func(self.goal):
                state_str.append(line)
        elif self.render_mode == 'text':
            state_str.append(f"Current state: {self.describe_state(self.state)}")
            state_str.append(f"Goal state: {self.describe_state(self.goal)}")

        return "\n".join(state_str)

    def render_list(self, mode=None):
        """Render the environment state."""
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return f"{self.state}\nGoal: {self.goal}"
        else:
            raise ValueError(f"Invalid mode: {render_mode}")
            
    def get_all_actions(self):
        """Get all possible actions in the environment."""
        actions = []
        for block in range(1, self.num_blocks + 1):
            for dest in range(0, self.num_blocks + 1):
                if block != dest:
                    actions.append((block, dest))
        return actions
        
    def close(self):
        """Clean up any resources."""
        pass

    def plot_row(self,blocksList):
        lines = []
        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + " ¯¯¯ "
        lines.append(oneLine)
        
        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + "| " + str(block+1) + " |"                                            
        lines.append(oneLine)

        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + " ___ "
        lines.append(oneLine)
        return lines
        
    def _render_v2(self, mode='human', close=False):
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("************** New Step ***************\n") 
        # List the files in the table
        on_table = sorted([i for i, e in enumerate(self.state) if e == 0])
        lines = []
        lines.extend(self.plot_row(on_table))
        newrow = []
        on_table = [x+1 for x in on_table]
        
        for block in on_table:                
            #For every block, search if is in the list
            if (block in self.state):
                newrow.append(self.state.index(block))

        while (len(newrow)>0):            
            print('newrow')
            print (newrow)
            lines.extend(self.plot_row(newrow))
            newrow = [x+1 for x in newrow]
            newrow2 = newrow
            newrow = []
            for block in newrow2:                
            #For every block, search if is in the list
                if (block in self.state):
                    newrow.append(self.state.index(block))
            
        # Plot everything in reverse order as it was added

        l = len(lines)-1
        while (l>=0):
            #print (l)
            outfile.write(lines[l])
            outfile.write("\n")
            l = l-1
        return outfile

    def seed(self, seed=None):
        """Set the seed for this env's random number generator."""
        return self._seed(seed)
