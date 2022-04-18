from stable_baselines3 import PPO
import os
from rendergame import RenderGame
from environment import GameEnvironment, prepare_array

moddir = "models/PPO"

env = GameEnvironment()
env.reset()


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


def play_all():
    """Play all saved models."""
    all_models = os.listdir(moddir)
    for model_name in all_models:
        model = PPO.load(f"{moddir}/{model_name}", env, verbose=0)
        show_game(model)


if __name__ == "__main__":
    print(
        "Please enter number of model you'd wish to play,'latest' for latest, or 'all' to play all."
    )
    to_play = input()
    if to_play == "all":
        play_all()
    elif to_play == "latest":
        all_models = os.listdir(moddir)
        model = PPO.load(f"{moddir}/{all_models[-1]}", env, verbose=0)
        show_game(model)
    else:
        model = PPO.load(f"{moddir}/{to_play}.zip", env, verbose=0)
        show_game(model)
