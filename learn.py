import gym
from stable_baselines3 import PPO
import os
import typing
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame


moddir = "models/PPO"
logdir = "logs/PPO"

if not os.path.exists(moddir):
    os.makedirs(moddir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GameEnvironment()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
# model = PPO.load(f"{moddir}/150000.zip", env, verbose=1, tensorboard_log=logdir)


def count_models():
    """Count how many models are in models directory"""
    files_list = next(os.walk(moddir))[2]
    return len(files_list)


def show_game(ml_model: PPO):  # type: ignore
    """Show game based off current model."""
    show_game = RenderGame()
    stuck_counter = 0
    observation = prepare_array(show_game.get_array())
    while stuck_counter < 5 and not show_game.is_game_terminated():
        direction: int = ml_model.predict(observation)[0]  # type: ignore
        show_game.move(direction)
        new_observation = prepare_array(show_game.get_array())
        if (new_observation == observation).all():
            stuck_counter += 1
        else:
            stuck_counter = 0
        observation = new_observation
    show_game.quit()


TIMESTEPS = 50000
for i in range(1, 1000000000):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO", reset_num_timesteps=False)
    model.save(f"{moddir}/{TIMESTEPS*i}")
    #show_game(model)  # type: ignore
