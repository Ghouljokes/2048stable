"""Contains defs for certain variables."""
import numpy as np
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from selenium.webdriver.common.keys import Keys

# CAN BE TWEAKED MID-TRAIN
MAX_STUCK = 1  # how many invalid moves can be made in a row
WRONG_MOVE_PUNISHMENT = -10
TIMESTEP_INTERVAL = 25000
MODEL = DQN  # technically. It doesn't alter current train, just moves to another.


# CAN NOT BE TWEAKED MID-TRAIN
# MISC, NOT TO BE CHANGED
VALUES = {
    "tile-2": 2,
    "tile-4": 4,
    "tile-8": 8,
    "tile-16": 16,
    "tile-32": 32,
    "tile-64": 64,
    "tile-128": 128,
    "tile-256": 256,
    "tile-512": 512,
    "tile-1024": 1024,
    "tile-2048": 2048,
    "tile-4096": 4096,
    "tile-8192": 8192,
    "tile-16384": 16384,
    "tile-32768": 32768,
    "tile-65536": 65536,
    "tile-131072": 131072,
}
POSITIONS = {
    "tile-position-1-1": 0,
    "tile-position-2-1": 1,
    "tile-position-3-1": 2,
    "tile-position-4-1": 3,
    "tile-position-1-2": 4,
    "tile-position-2-2": 5,
    "tile-position-3-2": 6,
    "tile-position-4-2": 7,
    "tile-position-1-3": 8,
    "tile-position-2-3": 9,
    "tile-position-3-3": 10,
    "tile-position-4-3": 11,
    "tile-position-1-4": 12,
    "tile-position-2-4": 13,
    "tile-position-3-4": 14,
    "tile-position-4-4": 15,
}
MOVES = [Keys.UP, Keys.RIGHT, Keys.DOWN, Keys.LEFT]
WINDOW_POS = (10, 10)
WINDOW_SIZE = (500, 600)
DIR_VECTORS = np.array(
    [
        (-1, 0),  # up
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
    ]
)
TRAVERSALS = np.array(
    [
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        [[0, 1, 2, 3], [3, 2, 1, 0]],
        [[3, 2, 1, 0], [0, 1, 2, 3]],
        [[0, 1, 2, 3], [0, 1, 2, 3]],
    ]
)
MODELS = {"A2C": A2C, "DQN": DQN, "PPO": PPO}
