import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from sb3customenv import GridWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

env = GridWorldEnv(render_mode="human")
env = TimeLimit(env, max_episode_steps=100)
check_env(env)

model = PPO.load("ppo_cold_model")
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    next_obs, rewards, terminated, truncated, info = env.step(action)

