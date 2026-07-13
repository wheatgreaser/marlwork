import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from customenv import GridWorldEnv

env = GridWorldEnv()
check_env(env)


