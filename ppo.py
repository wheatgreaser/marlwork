import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from sb3customenv import GridWorldEnv
from stable_baselines3 import PPO

env = GridWorldEnv()
check_env(env)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cold_model")

env = GridWorldEnv(render_mode="human")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
