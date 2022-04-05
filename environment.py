import gym
from gym import spaces
from training_game import TrainGame
from rendergame import RenderGame
import numpy as np


def prepare_array(array: np.ndarray) -> np.ndarray:
    """Perform log2 on each value of array."""
    np.seterr(divide="ignore")
    logged_array = np.log2(array)
    logged_array = np.maximum(0, logged_array)
    logged_array /= np.max(logged_array)
    return logged_array


class GameEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(GameEnvironment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.game = TrainGame(4)
        self.done = False
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.float32
        )
        self.reward = 0

    def step(self, action):
        """Action is int between 0 and 3"""
        self.game.move(action)
        observation = prepare_array(self.game.get_array()).astype(np.float32)
        self.reward = self.game.score
        self.done = self.game.is_game_terminated()
        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.game.restart()
        self.reward = 0
        # reset everything
        observation = prepare_array(self.game.get_array()).astype(np.float32)
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
