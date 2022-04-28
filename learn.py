"""Module for running learning algorithm."""

import os
import argparse
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame


MODELS = {"A2C": A2C, "DQN": DQN, "PPO": PPO}


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
    env.reset()
    model_type = MODELS[model_name]
    if os.listdir(model_dir):
        model_name = find_latest_model(model_dir)
        print(f"Loading {model_name}")
        init_model = model_type.load(
            f"{model_dir}/{model_name}",
            env,
            verbose=1,
            tensorboard_log=logdir,
        )
        timesteps = int(model_name.split(".")[0])

    else:
        init_model = model_type("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
        timesteps = 0
    return init_model, timesteps


# train a model
def train_model(model: A2C | PPO | DQN, timesteps: int):
    """Have a model run its learning algorithm.

    Args:
        model (Model): Model to train.
        timesteps (int): Timesteps to train for.
    """
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)


def show_game(ml_model: A2C | PPO | DQN):
    """Show game based off current model."""
    game = RenderGame()
    stuck_counter = 0
    observation = prepare_array(game.get_array())
    while stuck_counter < 5 and not game.is_game_terminated():
        direction = ml_model.predict(observation)[0]
        game.move(direction)
        new_observation = prepare_array(game.get_array())
        if (new_observation == observation).all():
            stuck_counter += 1
        else:
            stuck_counter = 0
        observation = new_observation
    game.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to train the ai.")
    parser.add_argument(
        "--showgame",
        help="Enables html replay each time a model is saved.",
        action="store_true",
    )
    parser.add_argument(
        "--a2c", help="Uses A2C for the model instead of PPO.", action="store_true"
    )
    parser.add_argument(
        "--dqn", help="Uses DQN for the model instead of PPO.", action="store_true"
    )
    args = parser.parse_args()

    if args.a2c:
        MODEL_NAME = "A2C"
    elif args.dqn:
        MODEL_NAME = "DQN"
    else:
        MODEL_NAME = "PPO"

    model, total_timesteps = initialize_model(MODEL_NAME)
    TIMESTEP_INTERVAL = 25000
    while True:
        train_model(model, TIMESTEP_INTERVAL)
        total_timesteps += TIMESTEP_INTERVAL
        model.save(f"models/{MODEL_NAME}/{total_timesteps}")
        if args.showgame:
            show_game(model)
