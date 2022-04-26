import os
import argparse
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame


# choose model
def ensure_dirs(model_name: str):
    """Check if dirs for model name exist, and if not, make them."""
    model_dir = f"models/{model_name}"
    log_dir = f"logs/{model_name}"
    for directory in [model_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)


# load a model from a directory
def find_latest_model(model_dir):
    """Gets name of latest model from a dir. It is assumed dir is not empty.

    Args:
        model_dir (str): Name of directory to load model from.
    Returns:
        str: name of latest model to load.
    """
    model_list = os.listdir(model_dir)
    model_list.sort(key=lambda model: int(model.split(".")[0]))
    return model_list[-1]


# initialize model in directory if no model exists
# train a model
