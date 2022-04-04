from telnetlib import GA
import gym
from stable_baselines3 import PPO
import os
import time
from environment import GameEnvironment

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GameEnvironment()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 1000000000):
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO", reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
