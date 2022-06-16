"""Environment the model will be trained in."""
import gym
from gym import spaces
import numpy as np
from training_game import TrainGame


def prepare_array(array: np.ndarray):
    """Create a one-hot vector array of the game grid.

    Args:
        array (np.ndarray): flattened array of the game board.

    Returns:
        np.ndarray: flattened one-hot vectorization.
    """
    hot_vectors = np.zeros((16, 18)).astype(int)
    for i, num in enumerate(array):
        logged = int(max(0, np.log2(num)))
        hot_vectors[i][logged] = 1
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

    def reset(self):
        # This is referenced outside of my code, so be careful.
        self.game.__init__()
        # reset everything
        observation = prepare_array(self.game.grid.flatten())
        return observation  # reward, done, info can't be included

    def step(self, action):
        """Action is int between 0 and 3"""
        self.game.move(action)
        observation = prepare_array(self.game.grid.flatten())
        self.reward = self.game.reward
        self.done = self.game.is_terminated()
        info = {}
        return observation, self.reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
