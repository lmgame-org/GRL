import numpy as np
import random
from typing import Tuple, Dict, Any, List
import gymnasium as gym
import copy
from lmgamerl.agents.agent_utils import all_seed
from lmgamerl.agents.base_env import BaseEnv

def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False

class TetrisEnv(BaseEnv):
    """
    A Tetris environment that follows the BaseDiscreteActionEnv interface.
    The environment has a 8x8 grid and supports basic Tetris mechanics.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        self.width = self.config.get('dim_x', 4)
        self.height = self.config.get('dim_y', 4)
        self.board = np.zeros(shape=(self.width, self.height), dtype=np.bool_)
        self.GRID_LOOKUP = self.config.get('grid_lookup', {0: "_", 1: "#", 2: "X"})
        self.ACTION_LOOKUP = self.config.get('action_lookup', {0: "Left", 1: "Right", 2: "Down"})
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(3, start=0)
        self.render_mode = self.config.get('render_mode', 'text')

        if self.config.get('box_type', 1) == 2:
            self.shapes = {
                'I': [(0, 0), (0, -1)],
                '—': [(0, 0), (-1, 0)],
            }
            self.shape_names = ['I', '—'] 
        elif self.config.get('box_type', 1) == 3:
            self.shapes = {
                'I': [(0, 0), (0, -1)],
                '—': [(0, 0), (-1, 0)],
                'O': [(0, 0), (-1, 0), (0, -1), (-1, -1)],
            }
            self.shape_names = ['I', '—', 'O'] 
        elif self.config.get('box_type', 1) == 1:
            self.shapes = {
                'O': [(0, 0)],
            }
            self.shape_names = ['O'] 

        # Action space: left, right, down
        self.actions = {
            0: self._left,
            1: self._right,
            2: self._soft_drop
        }
        
        # Game state
        self.time = 0
        self.score = 0
        self.anchor = None
        self.shape = None
        self.n_deaths = 0
        self._shape_counts = [0] * len(self.shapes)
        
        # Pre-generated pieces
        self.pre_generated_pieces = []
        self.current_piece_index = 0
        
        # Initialize game
        self.reset()

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return self.shapes[self.shape_names[i]]

    def _generate_piece(self):
        """Generate a new piece with random position and shape"""
        shape = self._choose_shape()
        if (-1, 0) in shape:
            anchor = (random.randint(1, self.width - 1), 0)
        else:
            anchor = (random.randint(0, self.width - 1), 0)
        return (anchor, shape)

    def _new_piece(self):
        """Get the next pre-generated piece"""
        if self.current_piece_index < len(self.pre_generated_pieces):
            self.anchor, self.shape = self.pre_generated_pieces[self.current_piece_index]
            self.current_piece_index += 1
        else:
            # Fallback if we somehow run out of pre-generated pieces
            self.anchor, self.shape = self._generate_piece()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        lines_cleared = sum(can_clear)
        self.score += lines_cleared
        self.board = new_board
        return lines_cleared

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def _left(self):
        new_anchor = (self.anchor[0] - 1, self.anchor[1])
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _right(self):
        new_anchor = (self.anchor[0] + 1, self.anchor[1])
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _soft_drop(self):
        new_anchor = (self.anchor[0], self.anchor[1] + 1)
        if not is_occupied(self.shape, new_anchor, self.board):
            self.anchor = new_anchor

    def _hard_drop(self):
        while True:
            new_anchor = (self.anchor[0], self.anchor[1] + 1)
            if is_occupied(self.shape, new_anchor, self.board):
                break
            self.anchor = new_anchor

    def _idle(self):
        pass

    def reset(self, seed=None, **kwargs) -> Any:
        """Reset the environment to initial state."""
        try:
            with all_seed(seed):
                self.time = 0
                self.score = 0
                self.board = np.zeros_like(self.board)
                
                # Pre-generate a batch of pieces
                self.pre_generated_pieces = []
                self.current_piece_index = 0
                num_pieces_to_generate = self.width * self.height + 1
                for _ in range(num_pieces_to_generate):
                    self.pre_generated_pieces.append(self._generate_piece())
                
                # Get the first piece
                self._new_piece()
                return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take (0-2)
        Returns:
            observation, reward, done, info
        """
        if action not in self.actions:
            return self.render(), 0, True, {"error": "Invalid action"}

        # Store previous position
        previous_pos = copy.deepcopy(self.anchor) if self.anchor is not None else None

        # Execute action
        self.actions[action]()
        
        # If action was not drop, automatically move down one step
        # if action != 2:  # Not a drop action

        # MAY 3TH: COMMENT OUT SOFT DROP LOGIC FOR TRAINING
        #new_anchor = (self.anchor[0], self.anchor[1] + 1)
        #if not is_occupied(self.shape, new_anchor, self.board):
        #    self.anchor = new_anchor
        
        # Update time and compute reward
        self.time += 1
        reward = -0.1
        
        done = False
        dropped = False
        info = {}
        
        lines_cleared = 0
        if self._has_dropped():
            dropped = True
            # Lock piece in place
            self._set_piece(True)
            
            # Clear lines and add reward
            lines_cleared = self._clear_lines()
            reward += lines_cleared * 10
            
            # Check for game over
            if np.any(self.board[:, 0]):
                done = True
                # print("DEBUG: current_piece_index", self.current_piece_index)
                # info["game_over"] = True
            else:
                self._new_piece()
        
        # Update board state
        self._set_piece(True)
        state = self.render()
        self._set_piece(False)
        
        # Compute if action was effective (piece moved)
        action_effective = previous_pos is not None and previous_pos != self.anchor

        info["action_is_effective"] = action_effective
        info["action_is_valid"] = True
        if lines_cleared > 0:
            done = True
            info["success"] = True
        else:
            info["success"] = False
        info['dropped'] = dropped
        
        return state, reward, done, info

    def render(self, mode: str = 'text') -> Any:
        """Render the current state of the environment."""
        if mode == 'text':
            # First render the board with placed blocks as '#'
            board_str = '\n'.join([''.join(['#' if j else '_' for j in i]) for i in self.board.T])
            
            # Then add the current piece as 'X'
            self._set_piece(True)
            current_piece_positions = [(self.anchor[0] + x, self.anchor[1] + y) for x, y in self.shape]
            self._set_piece(False)
            
            # Convert board to list of lists for easier manipulation
            board_lines = board_str.split('\n')
            
            # Replace '_' with 'X' at current piece positions
            for x, y in current_piece_positions:
                if 0 <= y < len(board_lines) and 0 <= x < len(board_lines[0]):
                    line = list(board_lines[y])
                    line[x] = 'X'
                    board_lines[y] = ''.join(line)
            
            # Join back to string
            s = '\n'.join(board_lines)
        else:
            s = self.board.copy()
        return s

    def get_all_actions(self) -> List[int]:
        """Get list of all valid actions."""
        return list(self.actions.keys())

    # ────────────────── NEW: graceful cleanup ──────────────
    def close(self) -> None:
        """
        Placeholder for interface compatibility.
        (Nothing to free right now, but keeps test suites happy.)
        """
        self.board   = None
        self.anchor  = None
        self.shape   = None