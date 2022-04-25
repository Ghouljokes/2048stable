"""Main file to train the ai."""
import os
import argparse
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from environment import GameEnvironment, prepare_array
from rendergame import RenderGame

TIMESTEPS = 50000

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
    Model = A2C
    MODEL_NAME = "A2C"
elif args.dqn:
    Model = DQN
    MODEL_NAME = "DQN"
else:
    Model = PPO
    MODEL_NAME = "PPO"
MODDIR = f"models/{MODEL_NAME}"
LOGDIR = f"logs/{MODEL_NAME}"


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


def ensure_dir(directory):
    """Check if directory exists, and if not, make it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_model(env):
    """Create model if none exists, else load latest and get model count."""
    prev_models = os.listdir(MODDIR)
    prev_models.sort(key=lambda model: int(model.split(".")[0]))
    if len(prev_models) == 0:
        init_model = Model("MlpPolicy", env, verbose=1, tensorboard_log=LOGDIR)
        init_model_count = 1
    else:
        last_model = prev_models[-1]
        print(f"Loading model {last_model}")
        last_step = int(last_model.split(".")[0])
        init_model = Model.load(
            f"{MODDIR}/{last_model}",
            env,
            verbose=1,
            tensorboard_log=LOGDIR,
        )
        init_model_count = last_step // TIMESTEPS
    return init_model, init_model_count


if __name__ == "__main__":
    ensure_dir(MODDIR)
    ensure_dir(LOGDIR)

    environment = GameEnvironment()
    environment.reset()

    model, model_count = initialize_model(environment)

    for i in range(model_count, 1000000000):
        model.learn(
            total_timesteps=TIMESTEPS, tb_log_name=MODEL_NAME, reset_num_timesteps=False
        )
        model.save(f"{MODDIR}/{TIMESTEPS*i}")
        if args.showgame:
            show_game(model)
