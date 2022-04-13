import gym
from stable_baselines3 import PPO
import os
import time
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame


models_dir = "models/PPO"
logdir = "logs/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GameEnvironment()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

def count_models():
    """Count how many models are in models directory"""
    files_list = next(os.walk(models_dir))[2]
    return len(files_list)


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


TIMESTEPS = 50000
for i in range(count_models() + 1, 1000000000):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO", reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    show_game(model)
