# gymnasium wrapper for snake environment

from typing import Optional, Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from source.environments import OriginalSnakeEnvironment, BaseEnvironment


class gym_SnakeEnvironment(gym.Env):
    # gym wrapper for snake game

    def __init__(
        self,
        n_boards: int,
        board_size: int,
        mask_size: Optional[int] = None,
        rewards: Optional[Dict[str, float]] = None,
    ):
        # initializes the environment
        super().__init__()

        # create the base snake environment
        self.base_env = OriginalSnakeEnvironment(
            n_boards=n_boards,
            board_size=board_size,
            mask_size=mask_size,
            rewards=rewards
        )

        self.n_boards = n_boards
        self.board_size = board_size

        # set mask size
        if mask_size is not None:
            self.mask_size = mask_size
        else:
            self.mask_size = 0

        # check if fully observable
        if mask_size is None:
            self.isFullyObservable = True
        elif mask_size == 0:
            self.isFullyObservable = True
        else:
            self.isFullyObservable = False

        # track termination state
        self.terminated = False

        # store last observation for cases where board resets
        self.last_observation = None

        # define observation space based on to_state() output format
        if self.isFullyObservable:
            # full board observation: (board_size, board_size, 4) one-hot encoded
            # channels: EMPTY, FRUIT, BODY, HEAD (excludes WALL which is channel 0)
            obs_shape = (board_size, board_size, 4)
        else:
            # partial observation: ((2*mask_size+1), (2*mask_size+1), 4)
            obs_shape = (2 * self.mask_size + 1, 2 * self.mask_size + 1, 4)
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # define action space: UP=0, RIGHT=1, DOWN=2, LEFT=3 (excluding NONE=4)
        self.action_space = spaces.Discrete(4)
    
    def _get_obs(self) -> np.ndarray:
        # gets current observation
        # this returns one-hot encoded boards
        state = self.base_env.to_state()
        
        # extract single board state (first board in the batch)
        # to_state() returns shape (n_boards, height, width, channels)
        # the state is already properly one-hot encoded by to_state()
        observation = state[0].astype(np.float32)
        
        # store this observation as last_observation
        self.last_observation = observation.copy()
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        # returns info dict
        board = self.base_env.boards[0]
        
        # find head position
        head_positions = np.argwhere(board == BaseEnvironment.HEAD)
        if len(head_positions) > 0:
            # some safeguards for extreme cases
            head_pos = tuple(head_positions[0])
        else:
            # if no head found, return default values
            default_info = {}
            default_info['manhattan_distance_to_fruit'] = -1
            default_info['manhattan_distance_to_wall'] = -1
            default_info['tail_length'] = 0
            return default_info
        
        # find fruit position
        fruit_positions = np.argwhere(board == BaseEnvironment.FRUIT)
        if len(fruit_positions) > 0:#safeguards again
            fruit_pos = tuple(fruit_positions[0])
            manhattan_distance_to_fruit = abs(head_pos[0] - fruit_pos[0]) + abs(head_pos[1] - fruit_pos[1])
        else:
            manhattan_distance_to_fruit = -1
        
        # calculate minimum manhattan distance to any wall
        # walls are at borders: row 0, row board_size-1, col 0, col board_size-1
        head_row = head_pos[0]
        head_col = head_pos[1]

        dist_to_top = self.board_size - 1 - head_row
        dist_to_bottom = head_row
        dist_to_right = self.board_size - 1 - head_col
        dist_to_left = head_col

        # find the minimum distance to any wall
        manhattan_distance_to_wall = dist_to_top
        if dist_to_bottom < manhattan_distance_to_wall:
            manhattan_distance_to_wall = dist_to_bottom
        if dist_to_right < manhattan_distance_to_wall:
            manhattan_distance_to_wall = dist_to_right
        if dist_to_left < manhattan_distance_to_wall:
            manhattan_distance_to_wall = dist_to_left
        
        # get tail length (number of body segments)
        tail_length = len(self.base_env.bodies[0])

        info = {}
        info['manhattan_distance_to_fruit'] = manhattan_distance_to_fruit
        info['manhattan_distance_to_wall'] = manhattan_distance_to_wall
        info['tail_length'] = tail_length
        return info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # resets environment for new episode
        super().reset(seed=seed)
        
        # set numpy random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # reset the base environment by creating a new board
        # get_board() returns a fresh initialized board
        new_board = self.base_env.get_board()
        self.base_env.boards[0] = new_board
        self.base_env.bodies[0] = []
        
        # place a fruit on the board (get_board() only places the head)
        available = np.argwhere(self.base_env.boards[0] == BaseEnvironment.EMPTY)
        if len(available) > 0:
            fruit_pos = available[np.random.choice(range(len(available)))]
            self.base_env.boards[0][fruit_pos[0], fruit_pos[1]] = BaseEnvironment.FRUIT
        
        # reset termination flag
        self.terminated = False

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # takes one step in environment
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # apply action to base environment using move() method
        # move() expects actions in format that creates 2D array when np.array() is applied
        # pass as [[action]] so np.array([[action]]) creates shape (1, 1)
        # move() returns rewards as tensorflow float32 tensor of shape (n_boards, 1)
        reward_tensor = self.base_env.move([[action]])
        
        # extract scalar reward value (get the latest/only reward for this step)
        reward = float(reward_tensor.numpy()[0, 0])
        
        # check if reward is negative (termination condition)
        if reward < 0:
            self.terminated = True
        
        # get observation after the action
        # note: if the snake won/died, the board may have been reset
        # in that case, we return the last stored observation
        observation = self._get_obs()
        
        # check terminated state
        terminated = self.is_terminated()
        
        # no truncation in this environment
        truncated = False
        
        # get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info 
    
    def is_terminated(self) -> bool:
        # checks if episode ended
        is_done = self.terminated
        return is_done

    def close(self):
        # cleanup (does nothing)
        pass
