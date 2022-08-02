"""Module for running learning algorithm."""

import os
import argparse
import numpy as np
from sb3_contrib.trpo.trpo import TRPO
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame
from defs import TIMESTEP_INTERVAL, MAX_STUCK

policy_kwargs = dict(net_arch=[128, 64, 64, 32, 8])


def get_dirs(model_name: str):
    """Get dirs for model name, and make them if they don't exist.

    Args:
        model_name (str): Name of model to get dirs for.

    Returns:
        str, str: Model and log dirs.
    """
    model_dir = f"models/{model_name}"
    log_dir = f"logs/{model_name}"
    for directory in [model_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return model_dir, log_dir


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
def initialize_model(model_name):
    """Initialize a model.

    Args:
        model_type (str): name of model to use.

    Returns:
        model: Initialized model.
        int: Timestep to start model from.
    """
    model_dir, logdir = get_dirs(model_name)
    env = GameEnvironment()
    if os.listdir(model_dir):
        model_name = find_latest_model(model_dir)
        print(f"Loading {model_name}")
        model = TRPO.load(
            f"{model_dir}/{model_name}",
            env,
            verbose=1,
            tensorboard_log=logdir,
        )
        timesteps = int(model_name.split(".")[0])

    else:
        model = TRPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=logdir,
            policy_kwargs=policy_kwargs,
        )
        timesteps = 0
    return model, timesteps


# train a model
def train_model(model: TRPO):
    """Have a model run its learning algorithm.

    Args:
        model (Model): Model to train.
    """
    model.learn(total_timesteps=TIMESTEP_INTERVAL, reset_num_timesteps=False)


def show_game(ml_model: TRPO):
    """Show game based off current model."""
    game = RenderGame()
    stuck_counter = 0
    observation = prepare_array(game.get_array())
    while stuck_counter < MAX_STUCK and not game.is_terminated():
        direction = ml_model.predict(observation)[0]
        game.move(direction)
        new_observation = prepare_array(game.get_array())
        if (new_observation == observation).all():
            stuck_counter += 1
        else:
            stuck_counter = 0
        observation = new_observation
    game.driver.quit()


if __name__ == "__main__":
    np.seterr(divide="ignore")
    parser = argparse.ArgumentParser(description="Program to train the ai.")
    parser.add_argument(
        "--showgame",
        help="Enables html replay each time a model is saved.",
        action="store_true",
    )

    args = parser.parse_args()

    model, total_timesteps = initialize_model("TRPO")
    while True:
        train_model(model)
        total_timesteps += TIMESTEP_INTERVAL
        model.save(f"models/TRPO/{total_timesteps}")
        if args.showgame:
            show_game(model)
