"""Show model playing game in the html player."""
import os
from stable_baselines3.ppo.ppo import PPO
from rendergame import RenderGame
from environment import GameEnvironment, prepare_array

MODDIR = "models/PPO"

env = GameEnvironment()
env.reset()


def show_game(ml_model):
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


def play_all():
    """Play all saved models."""
    model_list = os.listdir(MODDIR)
    for model_name in model_list:
        display_model = PPO.load(f"{MODDIR}/{model_name}", env, verbose=0)
        show_game(display_model)


if __name__ == "__main__":
    print(
        "Please enter number of model you'd wish to play,"
        + "'latest' for latest, or 'all' to play all."
    )
    to_play = input()
    if to_play == "all":
        play_all()
    elif to_play == "latest":
        all_models = os.listdir(MODDIR)
        model = PPO.load(f"{MODDIR}/{all_models[-1]}", env, verbose=0)
        show_game(model)
    else:
        model = PPO.load(f"{MODDIR}/{to_play}.zip", env, verbose=0)
        show_game(model)
