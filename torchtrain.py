import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from customenv import GridWorldEnv
from gymnasium.wrappers import TimeLimit


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)




env = GridWorldEnv()
state_size = 18
action_size = 4

gamma = 0.99             
epsilon = 1.0           
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory_size = 10000
start_epsilon = 1.0
num_episodes = 1000
epsilon_decay = (start_epsilon) / (num_episodes)
memory = deque(maxlen=memory_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net_1 = DQN(state_size, action_size).to(device)
target_net_1 = DQN(state_size, action_size).to(device)
target_net_1.load_state_dict(policy_net_1.state_dict())  
target_net_1.eval()


policy_net_2 = DQN(state_size, action_size).to(device)
target_net_2 = DQN(state_size, action_size).to(device)
target_net_2.load_state_dict(policy_net_2.state_dict())  
target_net_2.eval()


optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=learning_rate)
optimizer_2 = optim.Adam(policy_net_2.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
action = [0, 0]

def get_action(state, epsilon):
    if random.random() < epsilon:
        return [random.choice(range(action_size)), random.choice(range(action_size))] 
    else:
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values_1 = policy_net_1(state)
            q_values_2 = policy_net_2(state)
        return [q_values_1.argmax().item(), q_values_2.argmax().item()]   

def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = torch.FloatTensor(states).to(device)
    
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    actions1 = actions[:, 0].unsqueeze(1)
    actions2 = actions[:, 1].unsqueeze(1)

    current_q_1 = policy_net_1(states).gather(1, actions1)
    current_q_2 = policy_net_2(states).gather(1, actions2)

    rewards1 = rewards[:, 0].unsqueeze(1)
    rewards2 = rewards[:, 1].unsqueeze(1)
    next_q_1 = target_net_1(next_states).max(1)[0].detach().unsqueeze(1)
    target_q_1 = rewards1 + (gamma * next_q_1 * (1 - dones))

    next_q_2 = target_net_2(next_states).max(1)[0].detach().unsqueeze(1)
    target_q_2 = rewards2 + (gamma * next_q_2 * (1 - dones))

    loss_1 = loss_fn(current_q_1, target_q_1)
    loss_2 = loss_fn(current_q_2, target_q_2)

    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

    loss_1.backward()
    loss_2.backward()
    optimizer_1.step()
    optimizer_2.step()

final_epsilon = 0.0
target_update_freq = 100
epsilon_decay_2 = 0.995
reward_list = []
for episode in range(num_episodes):
    reset_result = env.reset()
    state = reset_result[0]
    total_reward = 0

    for t in range(10):
        action = get_action(state, epsilon)
        step_result = env.step(action)

        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward[0]
        total_reward += reward[1]
        replay()
        if done:
            break
    
    reward_list.append(total_reward)


    epsilon = max(final_epsilon, epsilon * epsilon_decay_2)
    if episode % target_update_freq == 0:
        target_net_1.load_state_dict(policy_net_1.state_dict())
        target_net_2.load_state_dict(policy_net_2.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")



df = pd.DataFrame({'Reward': reward_list})
df['rolling_av'] = df.Reward.rolling(10).mean()
fig, ax = plt.subplots()

ax.plot(list(range(1,(num_episodes+1))), df['rolling_av'])
plt.show()


env = GridWorldEnv(render_mode="human")
reset_result = env.reset()
state = reset_result[0]
total_reward = 0
reset_result = env.reset()
state = reset_result[0]
total_reward = 0

for t in range(10):
    action = get_action(state, 0)
    step_result = env.step(action)

    if len(step_result) == 5:
        next_state, reward, terminated, truncated, _ = step_result
        done = terminated or truncated
    else:
        next_state, reward, done, _ = step_result
    memory.append((state, action, reward, next_state, done))
    state = next_state
    total_reward += reward[0]
    total_reward += reward[1]
    replay()
    if done:
        break

reward_list.append(total_reward)

print(f"Episode final, Total Reward: {total_reward}, Epsilon: 0")


