# baseline heuristic agent for snake

import numpy as np
from typing import Tuple, Optional, Union


class BaselineAgent:
    # rule-based snake agent

    # environment constants
    HEAD = 4
    BODY = 3
    FRUIT = 2
    EMPTY = 1
    WALL = 0
    
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, verbose: int = 0):
        # initializes agent
        self.verbose = verbose
        self.action_names = {}
        self.action_names[self.UP] = "UP"
        self.action_names[self.RIGHT] = "RIGHT"
        self.action_names[self.DOWN] = "DOWN"
        self.action_names[self.LEFT] = "LEFT"
        
    def predict(
        self,
        state: np.ndarray
    ) -> Tuple[Union[int, np.ndarray], Optional[np.ndarray]]:
        # returns action for given state
        if len(state.shape) == 3:
            # single state - return scalar action
            action = self._get_action(state)
            return int(action), None
        else:
            # batch of states - return 1d array of actions
            actions = []
            for s in state:
                action = self._get_action(s)
                actions.append(action)
            return np.array(actions, dtype=np.int32), None
    
    def _get_action(self, state: np.ndarray) -> int:
        # gets action for single state
        board = self._decode_observation(state)
        
        # find positions
        head_pos = self._find_position(board, self.HEAD)
        fruit_pos = self._find_position(board, self.FRUIT)
        
        if head_pos is None:
            if self.verbose >= 1:
                print("Warning: No head found, returning random action")
            random_action = np.random.randint(0, 4)
            return random_action
        
        if fruit_pos is None:
            if self.verbose >= 1:
                print("Warning: No fruit found, returning safe action")
            return self._get_safe_action(board, head_pos)
        
        # get action towards fruit
        action = self._move_towards_fruit(board, head_pos, fruit_pos)
        
        if self.verbose >= 2:
            print(f"Head: {head_pos}, Fruit: {fruit_pos}, Action: {self.action_names[action]}")
        
        return action
    
    def _decode_observation(self, state: np.ndarray) -> np.ndarray:
        # converts one-hot to board
        # the encoding typically excludes the first channel (WALL=0), so we add 1
        board = np.argmax(state, axis=-1) + 1
        
        # handle edges that might be walls (assuming walls are at borders)
        # this is a heuristic reconstruction
        if board.shape[0] > 2 and board.shape[1] > 2:
            # check if borders should be walls
            board[0, :] = self.WALL
            board[-1, :] = self.WALL
            board[:, 0] = self.WALL
            board[:, -1] = self.WALL
        
        return board
    
    def _find_position(self, board: np.ndarray, cell_type: int) -> Optional[Tuple[int, int]]:
        # finds position of cell type on board
        positions = np.argwhere(board == cell_type)
        if len(positions) > 0:
            first_pos = positions[0]
            row = first_pos[0]
            col = first_pos[1]
            position = (row, col)
            return position
        return None
    
    def _move_towards_fruit(
        self,
        board: np.ndarray,
        head_pos: Tuple[int, int],
        fruit_pos: Tuple[int, int]
    ) -> int:
        # determines action to move towards fruit while avoiding collisions
        head_row, head_col = head_pos
        fruit_row, fruit_col = fruit_pos
        
        # calculate direction preferences
        row_diff = fruit_row - head_row
        col_diff = fruit_col - head_col
        
        # determine primary and secondary directions
        preferred_actions = []
        
        # prioritize largest difference first
        if abs(row_diff) >= abs(col_diff):
            # vertical movement more important
            if row_diff > 0:
                preferred_actions.append(self.UP)
            elif row_diff < 0:
                preferred_actions.append(self.DOWN)
            
            if col_diff > 0:
                preferred_actions.append(self.RIGHT)
            elif col_diff < 0:
                preferred_actions.append(self.LEFT)
        else:
            # horizontal movement more important
            if col_diff > 0:
                preferred_actions.append(self.RIGHT)
            elif col_diff < 0:
                preferred_actions.append(self.LEFT)
            
            if row_diff > 0:
                preferred_actions.append(self.UP)
            elif row_diff < 0:
                preferred_actions.append(self.DOWN)
        
        # try preferred actions in order
        for action in preferred_actions:
            if self._is_safe_action(board, head_pos, action):
                if self.verbose >= 2:
                    print(f"  Chose preferred action: {self.action_names[action]}")
                return action
        
        # if preferred actions are blocked, try any safe action
        all_actions = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        for action in all_actions:
            if self._is_safe_action(board, head_pos, action):
                if self.verbose >= 1:
                    print(f"  Warning: Preferred blocked, chose safe: {self.action_names[action]}")
                return action
        
        # if no safe action, return first preferred (will likely die, but it's inevitable)
        # check if we have any preferred actions
        if len(preferred_actions) > 0:
            fallback = preferred_actions[0]
        else:
            fallback = self.UP

        if self.verbose >= 1:
            print(f"  Warning: No safe moves! Choosing: {self.action_names[fallback]}")
        return fallback
    
    def _is_safe_action(
        self,
        board: np.ndarray,
        head_pos: Tuple[int, int],
        action: int
    ) -> bool:
        # checks if action is safe (no immediate collision)
        new_pos = self._get_new_position(head_pos, action)

        # check if position is out of bounds
        new_row = new_pos[0]
        new_col = new_pos[1]
        max_row = board.shape[0]
        max_col = board.shape[1]

        if new_row < 0:
            return False
        if new_row >= max_row:
            return False
        if new_col < 0:
            return False
        if new_col >= max_col:
            return False

        # check what's at the new position
        cell_value = board[new_row, new_col]

        # safe if empty
        if cell_value == self.EMPTY:
            return True

        # safe if fruit
        if cell_value == self.FRUIT:
            return True

        # unsafe if wall
        if cell_value == self.WALL:
            return False

        # unsafe if body
        if cell_value == self.BODY:
            return False

        return False
    
    def _get_new_position(
        self,
        pos: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        # calculates new position after action
        row, col = pos
        
        if action == self.UP:
            return (row + 1, col)
        elif action == self.DOWN:
            return (row - 1, col)
        elif action == self.RIGHT:
            return (row, col + 1)
        elif action == self.LEFT:
            return (row, col - 1)
        
        return pos
    
    def _get_safe_action(self, board: np.ndarray, head_pos: Tuple[int, int]) -> int:
        # gets any safe action when fruit position is unknown
        all_actions = [self.UP, self.RIGHT, self.DOWN, self.LEFT]

        for action in all_actions:
            if self._is_safe_action(board, head_pos, action):
                return action

        # no safe action found, return random
        random_action = np.random.randint(0, 4)
        return random_action

