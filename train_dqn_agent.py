# DESCRIPTION:
#   This script trains a Deep Q-Network (DQN) agent to play Connect 4 against a 
#   random-move opponent. The DQN agent uses:
#     - Experience replay
#     - A neural network to approximate Q-values
#     - Epsilon-greedy action selection
#     - Target network synchronization

import random
import numpy as np
import torch
from env.connect4_env import Connect4Env
from agents.dqn_agent import DQNAgent

# -----------------------------------------------------------------------------------
# Step 2: Training parameters
# -----------------------------------------------------------------------------------
episodes = 10000
max_steps_per_episode = 42
log_interval = 500
target_update_freq = 1000

# -----------------------------------------------------------------------------------
# Step 3: Initialize environment and agent
# -----------------------------------------------------------------------------------
env = Connect4Env()
state_size = env.board.size
action_size = 7
agent = DQNAgent(state_size=state_size, action_size=action_size)

wins = {1: 0, 2: 0, 'draw': 0}

# -----------------------------------------------------------------------------------
# Step 4: Training loop
# -----------------------------------------------------------------------------------
for ep in range(1, episodes + 1):
    env.reset()
    state = agent.encode_state(env.board)
    done = False
    current_player = 1

    for step in range(max_steps_per_episode):
        if current_player == 1:
            action = agent.act(env)
        else:
            action = random.choice(env.available_actions())

        success = env.drop_piece(action)
        if not success:
            continue

        next_state = agent.encode_state(env.board)
        done = env.is_win(current_player) or env.is_draw()

        if env.is_win(current_player):
            reward = 1 if current_player == 1 else -1
        elif env.is_draw():
            reward = 0.5
        else:
            reward = 0

        if current_player == 1:
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        state = next_state

        if done:
            if env.is_win(1):
                wins[1] += 1
            elif env.is_win(2):
                wins[2] += 1
            else:
                wins['draw'] += 1
            break

        current_player = 3 - current_player
        env.switch_player()

    # -----------------------------------------------------------------------------------
    # Step 5: Log progress and sync models
    # -----------------------------------------------------------------------------------
    if ep % log_interval == 0:
        print(
            f"Episode {ep} - "
            f"Wins: {wins[1]} | Losses: {wins[2]} | Draws: {wins['draw']} "
            f"| Epsilon: {agent.epsilon:.3f}"
        )

    if ep % target_update_freq == 0:
        agent.update_target_model()

# -----------------------------------------------------------------------------------
# Step 6: Save trained model
# -----------------------------------------------------------------------------------
agent.save("dqn_model.pth")
