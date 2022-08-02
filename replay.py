"""Show model playing game in the html player."""
import os
from sb3_contrib.trpo.trpo import TRPO

from environment import GameEnvironment
from learn import show_game

ModelType = TRPO
MODEL_NAME = "TRPO"
MODDIR = f"models/{MODEL_NAME}"
env = GameEnvironment()


def play_all():
    """Play all saved models."""
    model_list = os.listdir(MODDIR)
    model_list.sort(key=lambda model: int(model.split(".")[0]))
    for model_num in model_list:
        display_model = ModelType.load(f"{MODDIR}/{model_num}", env, verbose=0)
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
        all_models.sort(key=lambda model: int(model.split(".")[0]))
        model = ModelType.load(f"{MODDIR}/{all_models[-1]}", env, verbose=0)
        show_game(model)
    else:
        model = ModelType.load(f"{MODDIR}/{to_play}.zip", env, verbose=0)
        show_game(model)
