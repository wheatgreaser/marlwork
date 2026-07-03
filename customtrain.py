from customenv import GridWorldEnv
from collections import defaultdict
import gymnasium as gym
import numpy as np 
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict


env = GridWorldEnv()
obs, info = env.reset(seed=42)  
print(f"Starting position - Agent: {obs['agent']}, Target: {obs['vertiport_locations']}")
print(obs)

learning_rate = 0.9
discount_factor = 0.9
terminated = False
start_epsilon = 1.0
lr = 0.01
num_episodes = 10000
q_values = defaultdict(lambda: np.zeros(env.action_space.n))
final_epsilon = 0.0
epsilon_decay = (start_epsilon) / (num_episodes/2)
reward_list = []
reward_count = 0
epsilon = start_epsilon
def q_obs_gen(obs):
    return((obs['agent'], obs['vertiport_locations'][0], obs['vertiport_locations'][1], obs['vertiport_locations'][2], obs['vertiport_locations'][3]))

for episode in range(num_episodes):
    print(episode)
    obs, info = env.reset()
    print(obs)
    done = False
    while(not done):
        old_pos = obs['agent'].copy()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_values[q_obs_gen(obs)]))
        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward_count += reward
        future_q_value = (not terminated) * (np.max(q_values[q_obs_gen(next_obs)]))
        target = reward + discount_factor * future_q_value
        temporal_difference = target - q_values[q_obs_gen(obs)][action]
        q_values[q_obs_gen(obs)][action] += (lr * temporal_difference)

        done = terminated
        obs = next_obs
        new_pos = obs['agent']
    reward_list.append(reward_count)
    reward_count = 0

print(reward_list)


done = False
env = GridWorldEnv(render_mode="human")
obs, info = env.reset()
while(not done):
    old_pos = obs['agent'].copy() 
    action = int(np.argmax(q_values[q_obs_gen(obs)]))
    next_obs, reward, terminated, truncated, info = env.step(action)

    q_values[q_obs_gen(obs)][action] = reward
    future_q_value = (not terminated) * (np.max(q_values[q_obs_gen(next_obs)]))
    target = reward + discount_factor * future_q_value
    temporal_difference = target - q_values[q_obs_gen(obs)][action]
    q_values[q_obs_gen(obs)][action] += (lr * temporal_difference)

    done = terminated
    obs = next_obs
    new_pos = obs['agent']
    






