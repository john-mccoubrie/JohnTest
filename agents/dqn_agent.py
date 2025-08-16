# DESCRIPTION: This script implements a Deep Q-Network (DQN) agent for playing Connect 4.
#              The agent uses a neural network to approximate the Q-function and learns by
#              sampling past experiences from a replay buffer. It balances exploration and
#              exploitation using an epsilon-greedy policy, and periodically updates a target
#              network to stabilize training.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env.connect4_env import Connect4Env, ROWS, COLUMNS

# -----------------------------------------------------------------------------------
# Step 2: Define the Deep Q-Network architecture
# -----------------------------------------------------------------------------------
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# -----------------------------------------------------------------------------------
# Step 3: Define the DQN Agent
# -----------------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, replay_buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=replay_buffer_size)
        self.steps_done = 0
        self.update_target_every = 1000

    def encode_state(self, board):
        return np.array(board.reshape(-1), dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env):
        state = self.encode_state(env.board)
        if random.random() < self.epsilon:
            return random.choice(env.available_actions())
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            valid_actions = env.available_actions()
            q_values = q_values.cpu().numpy().flatten()
            masked_q = np.full_like(q_values, -np.inf)
            for a in valid_actions:
                masked_q[a] = q_values[a]
            return int(np.argmax(masked_q))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(curr_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path='dqn_model.pth'):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path='dqn_model.pth'):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_model()
