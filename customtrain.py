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
q_values_1 = defaultdict(lambda: np.zeros(env.action_space.n))
q_values_2 = defaultdict(lambda: np.zeros(env.action_space.n))

final_epsilon = 0.0
epsilon_decay = (start_epsilon) / (num_episodes/2)
reward_list = []
reward_count = 0
epsilon = start_epsilon
old_pos = [0, 0]
action = [0, 0]

def q_obs_gen(obs):
    return((obs['agent'][0], obs['agent'][1], obs['vertiport_locations'][0], obs['vertiport_locations'][1], obs['vertiport_locations'][2], obs['vertiport_locations'][3]))

for episode in range(num_episodes):
    print(episode)
    obs, info = env.reset()
    print(obs)
    done = False
    future_q_value = [0, 0]
    target = [0, 0]
    temporal_difference = [0, 0]
    while(not done):

        if np.random.random() < epsilon:
            action[0] = env.action_space.sample()
            action[1] = env.action_space.sample()
        else:
            action[0] = int(np.argmax((q_values_1[q_obs_gen(obs)])))
            action[1] = int(np.argmax((q_values_2[q_obs_gen(obs)])))

        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        
        next_obs, rewards, terminated, truncated, info = env.step(action)

        reward_count += rewards[0]
        future_q_value[0] = (not terminated) * (np.max(q_values_1[q_obs_gen(next_obs)]))
        future_q_value[1] = (not terminated) * (np.max(q_values_2[q_obs_gen(next_obs)]))

        target[0] = rewards[0] + discount_factor * future_q_value[0]
        target[1] = rewards[1] + discount_factor * future_q_value[1]

        temporal_difference[0] = target[0] - ((q_values_1[q_obs_gen(obs)][action]))
        temporal_difference[1] = target[1] - ((q_values_2[q_obs_gen(obs)][action]))

        q_values_1[q_obs_gen(obs)][action] += (lr * temporal_difference[0])
        q_values_2[q_obs_gen(obs)][action] += (lr * temporal_difference[1])

        done = terminated
        obs = next_obs
    reward_list.append(reward_count)
    reward_count = 0

print(reward_list)


done = False
env = GridWorldEnv(render_mode="human")
obs, info = env.reset()
future_q_value = [0, 0]
target = [0, 0]
temporal_difference = [0, 0]
while(not done):
    
    action[0] = int(np.argmax(q_values_1[q_obs_gen(obs)]))
    action[1] = int(np.argmax(q_values_2[q_obs_gen(obs)]))
    next_obs, rewards, terminated, truncated, info = env.step(action)

    (q_values_1[q_obs_gen(obs)][action]) = rewards[0]
    (q_values_2[q_obs_gen(obs)][action]) = rewards[1]

    future_q_value[0] = (not terminated) * (np.max((q_values_1[q_obs_gen(next_obs)])))
    future_q_value[1] = (not terminated) * (np.max((q_values_2[q_obs_gen(next_obs)])))

    target[0] = rewards[0] + discount_factor * future_q_value[0]
    target[1] = rewards[1] + discount_factor * future_q_value[1]

    temporal_difference[0] = target[0] - (q_values_1[q_obs_gen(obs)][action])
    temporal_difference[1] = target[1] - (q_values_2[q_obs_gen(obs)][action])

    q_values_1[q_obs_gen(obs)][action] += (lr * temporal_difference[0])
    q_values_2[q_obs_gen(obs)][action] += (lr * temporal_difference[1])

    done = terminated
    obs = next_obs
    






