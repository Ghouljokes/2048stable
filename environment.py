import gym
from gym import spaces
from training_game import TrainGame


class GameEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, arg1, arg2):
        super(GameEnvironment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(16)

    def step(self, action):
        # move
        # observation = get board array
        # reward = get game score
        # done = is game terminated
        # info =
        return observation, reward, done, info

    def reset(self):
        self.done = False
        # reset game board
        # reset score
        # reset everything
        # self.observation = board array
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
