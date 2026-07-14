import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class BaseEnvironment:
    # base snake game environment for multiple parallel boards

    # cell types
    HEAD = 4
    BODY = 3
    FRUIT = 2
    EMPTY = 1
    WALL = 0

    # action directions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4

    def __init__(self, n_boards, board_size, mask_size=None, rewards=None):
        # reward values (default)
        self.WIN_REWARD = 1.0
        self.FRUIT_REWARD = 0.5
        self.STEP_REWARD = 0.0
        self.ATE_HIMSELF_REWARD = -0.2
        self.HIT_WALL_REWARD = -0.1
        
        
        # override with custom rewards if provided
        if rewards is not None:
            self.set_rewards(rewards)

        # board configuration
        self.board_size = board_size
        self.n_boards = n_boards

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

        # initialize boards and snake bodies
        self.boards = np.ones((self.n_boards, self.board_size, self.board_size)) * self.EMPTY

        # create empty body list for each board
        self.bodies = []
        for i in range(self.n_boards):
            self.bodies.append([])

    def set_rewards(self, rewards):
        # sets custom reward values from dict
        self.WIN_REWARD = rewards['win']
        self.FRUIT_REWARD = rewards['fruit']
        self.STEP_REWARD = rewards['step']
        self.ATE_HIMSELF_REWARD = rewards['ate_himself']
        self.HIT_WALL_REWARD = rewards['hit_wall']

    def check_actions(self, new_heads):
        # checks if heads collide with walls, returns board indices
        walls = np.argwhere(self.boards == BaseEnvironment.WALL)
        # hash positions to enable fast collision detection
        powers = (np.max(walls) + 1) ** np.arange(walls.shape[1])
        walls_code = np.sum(walls * powers, axis=-1)
        new_heads_code = np.sum(new_heads * powers, axis=-1)
        collision_mask = np.isin(new_heads_code, walls_code)
        colliding_boards = new_heads[collision_mask][:, 0]
        return colliding_boards

    def move(self, actions):
        # executes actions for all boards and returns rewards
        # initialize positions and rewards
        heads = np.argwhere(self.boards == self.HEAD)
        actions = np.array(actions)
        rewards = np.zeros(self.n_boards, dtype=float)
        
        # convert actions to coordinate offsets
        dx = np.zeros(len(actions))
        dx[np.where(actions == self.UP)[0]] = 1
        dx[np.where(actions == self.DOWN)[0]] = -1

        dy = np.zeros(len(actions))
        dy[np.where(actions == self.RIGHT)[0]] = 1
        dy[np.where(actions == self.LEFT)[0]] = -1

        offset = np.hstack((np.zeros_like(actions), dx[:, None], dy[:, None]))
        
        # calculate new head positions
        new_heads = (heads + offset).astype(int)
        
        # handle wall collisions: revert movement and penalize
        hit_wall = self.check_actions(new_heads)
        actions[hit_wall] = self.NONE
        new_heads[hit_wall] = heads[hit_wall]
        dx[hit_wall] = 0.
        dy[hit_wall] = 0.
        offset[hit_wall] = 0.
        rewards[hit_wall] = self.HIT_WALL_REWARD

        # detect fruit consumption
        fruits = np.argwhere(self.boards == self.FRUIT)
        fruits_eaten_bool = np.all(fruits == new_heads, axis=-1)
        boards_where_fruits_is_been_eaten = np.argwhere(fruits_eaten_bool)

        # handle self-collision: remove tail segments after collision point
        bodies_ = np.argwhere(self.boards == self.BODY)
        bodies_head = new_heads[bodies_[:, 0]]
        boards_where_bodies_is_been_eaten = bodies_head[:, 0]
        if len(bodies_head):
            bodies_eaten_bool = np.all(bodies_ == bodies_head, axis=-1)
            eaten_body_peace = bodies_[np.argwhere(bodies_eaten_bool)[:, 0]]
            # delete tail from collision point onwards
            for piece in eaten_body_peace:
                b = piece[0]
                piece = piece[1:]
                index = np.argwhere(np.all(np.array(self.bodies[b]) == piece, axis=-1))[0, 0]
                to_delete = self.bodies[b][index:]
                to_delete_np = np.array(to_delete)
                self.boards[b, to_delete_np[:, 0], to_delete_np[:, 1]] = self.EMPTY
                del self.bodies[b][index:]
                rewards[b] = self.ATE_HIMSELF_REWARD

        # update snake bodies: add new head, remove tail (unless fruit eaten)
        for i in range(self.n_boards):
            # add old head position to body
            self.bodies[i].insert(0, heads[i][1:])
            # clear previous body positions from board
            self.boards[i][np.where(self.boards[i] == self.BODY)] = self.EMPTY
            self.boards[i][heads[i, 1], heads[i, 2]] = self.EMPTY
            # remove tail if no fruit was eaten
            if i not in boards_where_fruits_is_been_eaten.reshape((-1,)).tolist():
                self.bodies[i].pop()
            # redraw body on board
            if self.bodies[i]:
                body = np.array(self.bodies[i])
                self.boards[i][body[:, 0], body[:, 1]] = self.BODY

        # update board positions for heads (including where fruit was eaten)
        boards_where_fruits_is_been_eaten = boards_where_fruits_is_been_eaten.reshape((-1,))
        self.boards[
            heads[boards_where_fruits_is_been_eaten][:, 0],
            heads[boards_where_fruits_is_been_eaten][:, 1],
            heads[boards_where_fruits_is_been_eaten][:, 2]] = self.BODY
        self.boards[new_heads[:, 0], new_heads[:, 1], new_heads[:, 2]] = self.HEAD

        # assign final rewards
        rewards[boards_where_fruits_is_been_eaten] = self.FRUIT_REWARD
        rewards[np.setdiff1d(
            np.arange(0, self.n_boards),
            np.union1d(np.union1d(boards_where_fruits_is_been_eaten, boards_where_bodies_is_been_eaten), hit_wall)
        )] = self.STEP_REWARD

        # spawn new fruit on boards where it was eaten (reset if board full)
        for b in boards_where_fruits_is_been_eaten:
            available = np.argwhere(self.boards[b] == self.EMPTY)
            # win condition: no empty cells (reset board)
            if len(available) == 0:
                self.boards[b] = self.get_board()
                self.bodies[b] = []
                rewards[b] = self.WIN_REWARD
                available = np.argwhere(self.boards[b] == self.EMPTY)
            # place new fruit at random empty position
            chosen = available[np.random.choice(range(len(available)))]
            self.boards[b][chosen[0], chosen[1]] = self.FRUIT

        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        reshaped_rewards = tf.reshape(rewards_tensor, (-1, 1))
        return reshaped_rewards

    def to_state(self):
        # converts boards to one-hot encoded state representation
        if self.isFullyObservable:
            # return entire board as one-hot encoded state
            one_hot_boards = K.utils.to_categorical(self.boards)
            state = one_hot_boards[..., 1:]
            return state
        else:
            # return masked region around head as one-hot encoded state
            bs = np.pad(self.boards, ((0,0), (self.mask_size,self.mask_size), (self.mask_size,self.mask_size)))
            heads = np.argwhere(bs == self.HEAD)[:,1:]
            res = []
            for b, h in zip(bs, heads):
                masked_region = b[(h[0]-self.mask_size):(h[0]+self.mask_size+1), (h[1]-self.mask_size):(h[1]+self.mask_size+1)]
                res.append(masked_region)
            one_hot_masked = K.utils.to_categorical(res)
            state = one_hot_masked[..., 1:]
            return state

    def get_board(self):
        # creates and returns new initialized board (must be implemented by subclasses)
        raise NotImplementedError()


class OriginalSnakeEnvironment(BaseEnvironment):
    # snake environment with wall boundaries and random initial positions
    def __init__(self, n_boards, board_size, mask_size=None, rewards=None):
        super().__init__(n_boards, board_size, mask_size, rewards)
        # create walls on board perimeter
        self.boards[:, [0, -1], :] = self.WALL
        self.boards[:, :, [0, -1]] = self.WALL
        # initialize each board with random head and fruit positions
        for board in self.boards:
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.HEAD
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.FRUIT

    def get_board(self):
        # creates fresh board with walls and randomly placed head
        board = np.ones((self.board_size, self.board_size)) * self.EMPTY
        board[[0, -1], :] = self.WALL
        board[:, [0, -1]] = self.WALL
        # place head at random empty position
        available = np.argwhere(board == self.EMPTY)
        ind = available[np.random.choice(range(len(available)))]
        board[ind[0], ind[1]] = self.HEAD
        return board
