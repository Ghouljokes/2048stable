"""Main file to train the ai."""
import os
from stable_baselines3.ppo.ppo import PPO
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame

TIMESTEPS = 25000

MODDIR = "models/PPO"
LOGDIR = "logs/PPO"


def ensure_dir(directory):
    """Check if directory exists, and if not, make it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_model(env):
    """Create model if none exists, else load latest and get model count."""
    prev_models = os.listdir(MODDIR)
    if len(prev_models) == 0:
        init_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGDIR)
        init_model_count = 1
    else:
        last_model = prev_models[-1]
        last_step = int(last_model.split(".")[0])
        init_model = PPO.load(
            f"{MODDIR}/{last_model}",
            env,
            verbose=1,
            tensorboard_log=LOGDIR,
        )
        init_model_count = last_step // TIMESTEPS
    return init_model, init_model_count


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


if __name__ == "__main__":
    ensure_dir(MODDIR)
    ensure_dir(LOGDIR)

    environment = GameEnvironment()
    environment.reset()

    model, model_count = initialize_model(environment)

    for i in range(model_count, 1000000000):
        model.learn(
            total_timesteps=TIMESTEPS, tb_log_name="PPO", reset_num_timesteps=False
        )
        model.save(f"{MODDIR}/{TIMESTEPS*i}")
        show_game(model)
