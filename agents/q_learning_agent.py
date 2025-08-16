# DESCRIPTION: This script implements a Q-learning agent for playing Connect 4.
#              The agent uses tabular Q-learning to learn optimal actions by interacting with the environment.
#              It balances exploration and exploitation using an epsilon-greedy policy and updates Q-values using
#              the Bellman equation. Q-table is stored as a Python dictionary and can be saved or loaded via pickle.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
#              [2] OpenAI Gym Q-Learning Examples.
#              [3] StackOverflow. (2022). *How to implement Q-learning in Python?*
#                  https://stackoverflow.com/questions/41682126/q-learning-algorithm-python

# -----------------------------------------------------------------------------------
# Step 1: Import required libraries
# -----------------------------------------------------------------------------------
import numpy as np                                      # For numerical operations (e.g., random sampling, array reshaping)
import random                                           # For random action selection (exploration)
import pickle                                           # For saving/loading the Q-table to disk
from env.connect4_env import Connect4Env, ROWS, COLUMNS # Import Connect4 environment and constants

# -----------------------------------------------------------------------------------
# Step 2: Define the Q-learning agent class
# -----------------------------------------------------------------------------------
class QlearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
        self.q_table = {}                                                                           # Initialize empty Q-table as a dictionary: {(state, action): Q-value}
        self.alpha = alpha                                                                          # Learning rate: controls how much new info overrides old
        self.gamma = gamma                                                                          # Discount factor: importance of future rewards
        self.epsilon = epsilon                                                                      # Exploration rate: probability of choosing a random action
        self.epsilon_decay = epsilon_decay                                                          # Decay rate for epsilon to gradually reduce exploration
        self.epsilon_min = epsilon_min                                                              # Minimum value for epsilon to prevent zero exploration
        self.player = 1                                                                             # Initialize player ID (set dynamically during gameplay)

    # -----------------------------------------------------------------------------------
    # Step 3: Encode the game board state into a hashable format (string)
    # -----------------------------------------------------------------------------------
    def encode_state(self, board):
        return str(board.reshape(-1))   # Flatten the board and convert to string for dictionary key

    # -----------------------------------------------------------------------------------
    # Step 4: Retrieve Q-value for a given (state, action) pair
    # -----------------------------------------------------------------------------------
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)   # Return Q-value or 0.0 if unseen

    # -----------------------------------------------------------------------------------
    # Step 5: Choose an action using epsilon-greedy strategy
    # -----------------------------------------------------------------------------------
    def choose_action(self, env: Connect4Env):
        self.player = env.current_player                                                # Update player ID from environment
        state = self.encode_state(env.board)                                            # Encode current board state
        valid_actions = env.available_actions()                                         # Get list of legal moves (columns)

        if np.random.rand() < self.epsilon:                                             # With probability epsilon, explore
            return random.choice(valid_actions)                                         # Choose a random valid action
        else:                                                                           # Otherwise, exploit best known Q-values
            q_values = [self.get_q(state, a) for a in valid_actions]                    # Get Q-values for all valid actions
            max_q = max(q_values)                                                       # Find the highest Q-value
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]   # Filter best actions
            return random.choice(best_actions)                                          # Break ties randomly among best actions

    # -----------------------------------------------------------------------------------
    # Step 6: Update Q-table based on observed transition
    # -----------------------------------------------------------------------------------
    def learn(self, state, action, reward, next_state, done, valid_actions):
        old_q = self.get_q(state, action)                                       # Get the current Q-value
        future_q = 0.0                                                          # Initialize future reward estimate

        if not done:                                                            # If episode not over, estimate future value
            future_q = max([self.get_q(next_state, a) for a in valid_actions])  # Use max Q for next state

        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)   # Bellman equation: new Q = old Q + α * (reward + γ * max_future_q - old_q)
        self.q_table[(state, action)] = new_q                                   # Update Q-table with new Q-value

        if self.epsilon > self.epsilon_min:                                     # Decay exploration rate (epsilon), but not below minimum
            self.epsilon *= self.epsilon_decay                                  # Apply decay factor

    # -----------------------------------------------------------------------------------
    # Step 7: Save the learned Q-table to disk using pickle
    # -----------------------------------------------------------------------------------
    def save(self, path='q_table.pkl'):
        with open(path, 'wb') as f:         # Open file in binary write mode
            pickle.dump(self.q_table, f)    # Serialize and write Q-table to file

    # -----------------------------------------------------------------------------------
    # Step 8: Load a previously saved Q-table from disk
    # -----------------------------------------------------------------------------------
    def load(self, path='q_table.pkl'):
        with open(path, 'rb') as f:         # Open file in binary read mode
            self.q_table = pickle.load(f)   # Load Q-table from file into memory