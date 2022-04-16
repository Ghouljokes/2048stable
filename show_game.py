from stable_baselines3 import PPO
from rendergame import RenderGame
from environment import GameEnvironment, prepare_array

env = GameEnvironment()
env.reset()

model = PPO.load("models/PPO/68500000.zip", env, verbose=1)


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
    show_game(model)
