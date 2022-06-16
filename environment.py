"""Environment the model will be trained in."""
import gym
from gym import spaces
import numpy as np
from training_game import TrainGame


def prepare_array(array: np.ndarray):
    """Perform log2 on each value of array."""
    hot_vectors = np.zeros((16, 18)).astype(int)
    for i, num in enumerate(array):
        if num != 0:
            hot_position = int(np.log2(num))
            hot_vectors[i][hot_position] = 1
        else:
            hot_vectors[i][0] = 1
    return hot_vectors.flatten()


class GameEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(GameEnvironment).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.game = TrainGame()
        self.done = False
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary((16 * 18))
        self.reward = 0

    def step(self, action):
        """Action is int between 0 and 3"""
        self.game.move(action)
        observation = prepare_array(self.game.grid.flatten())
        self.reward = self.game.reward
        self.done = self.game.is_terminated()
        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.game.__init__()
        self.reward = 0
        # reset everything
        observation = prepare_array(self.game.grid.flatten())
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
