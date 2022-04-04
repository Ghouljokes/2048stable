import gym
from gym import spaces
from training_game import TrainGame


class GameEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, arg1, arg2):
        super(GameEnvironment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.game = TrainGame(4)
        self.done = False
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(16)
        self.reward = 0

    def step(self, action):
        """Action is int between 0 and 3"""
        self.game.move(action)
        self.observation = self.game.get_array()
        self.reward = self.game.score
        self.done = self.game.is_game_terminated()
        info = {}
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.game.restart()
        self.reward = 0
        # reset everything
        self.observation = self.game.get_array()
        return self.observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
