from stable_baselines3.common.env_checker import check_env
from environment import GameEnvironment


env = GameEnvironment()
# It will check your custom environment and output additional warnings
check_env(env)