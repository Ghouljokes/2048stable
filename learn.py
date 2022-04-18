import gym
from stable_baselines3 import PPO
import os
import typing
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame

TIMESTEPS = 25000

moddir = "models/PPO"
logdir = "logs/PPO"


def ensure_dir(directory):
    """Check if directory exists, and if not, make it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_model(environment):
    """Create model if none exists, else load latest and get model count."""
    prev_models = os.listdir(moddir)
    if len(prev_models) == 0:
        model = PPO("MlpPolicy", environment, verbose=1, tensorboard_log=logdir)
        model_count = 1
    else:
        last_model = prev_models[-1]
        last_step = int(last_model.split(".")[0])
        model = PPO.load(
            f"{moddir}/{last_model}",
            environment,
            verbose=1,
            tensorboard_log=logdir,
        )
        model_count = last_step // TIMESTEPS
    return model, model_count


def show_game(ml_model: PPO):
    """Show game based off current model."""
    show_game = RenderGame()
    stuck_counter = 0
    observation = prepare_array(show_game.get_array())
    while stuck_counter < 5 and not show_game.is_game_terminated():
        direction = ml_model.predict(observation)[0]
        show_game.move(direction)
        new_observation = prepare_array(show_game.get_array())
        if (new_observation == observation).all():
            stuck_counter += 1
        else:
            stuck_counter = 0
        observation = new_observation
    show_game.quit()


if __name__ == "__main__":
    ensure_dir(moddir)
    ensure_dir(logdir)

    env = GameEnvironment()
    env.reset()

    model, model_count = initialize_model(env)

    for i in range(model_count, 1000000000):
        model.learn(
            total_timesteps=TIMESTEPS, tb_log_name="PPO", reset_num_timesteps=False
        )
        model.save(f"{moddir}/{TIMESTEPS*i}")
        show_game(model)
