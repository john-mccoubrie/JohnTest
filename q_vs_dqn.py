# DESCRIPTION: This script runs a head-to-head evaluation between a Q-learning agent 
#              (tabular approach) and a Deep Q-Network (DQN) agent in Connect 4.
#              The Q-learning agent uses a trained Q-table, while the DQN uses a 
#              pre-trained neural network model. Results are tallied across episodes.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.
#              [2] Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
#              [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
#              [4] PyTorch Documentation – DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#              [5] StackOverflow. (2021). *How to run AI agent vs AI agent simulations?*
#                  https://stackoverflow.com/questions/65943215/ai-vs-ai-in-reinforcement-learning

# -----------------------------------------------------------------------------------
# Step 1: Import dependencies
# -----------------------------------------------------------------------------------
from env.connect4_env import Connect4Env, ROWS, COLUMNS    # Game environment and board constants
from agents.q_learning_agent import QlearningAgent         # Tabular Q-learning agent
from agents.dqn_agent import DQNAgent                      # Deep Q-Network agent
import random                                              # (Optional) Could be used for randomness if needed

# -----------------------------------------------------------------------------------
# Step 2: Define state & action sizes (for DQN)
# -----------------------------------------------------------------------------------
state_size = ROWS * COLUMNS   # Flattened board representation
action_size = COLUMNS         # Number of legal moves (columns)

# -----------------------------------------------------------------------------------
# Step 3: Initialize and load agents
# -----------------------------------------------------------------------------------
# --- Q-learning Agent ---
q_agent = QlearningAgent()
q_agent.load('q_table.pkl')   # Load pre-trained Q-table from file

# --- DQN Agent ---
dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.load('dqn_model.pth')  # Load pre-trained neural network weights

# -----------------------------------------------------------------------------------
# Step 4: Simulation configuration
# -----------------------------------------------------------------------------------
episodes = 1000                                # Number of games to play
wins = {'Q-agent': 0, 'DQN-agent': 0, 'draw': 0}  # Dictionary to track results

# -----------------------------------------------------------------------------------
# Step 5: Run simulation
# -----------------------------------------------------------------------------------
for ep in range(1, episodes + 1):
    env = Connect4Env()       # Initialize a fresh game
    done = False              # Flag to indicate game over
    current_player = 1        # Player 1 = Q-agent starts first

    while not done:
        # --- Agent selects action ---
        if current_player == 1:
            action = q_agent.choose_action(env)  # Q-learning agent chooses move
        else:
            action = dqn_agent.act(env)          # DQN agent chooses move

        # --- Attempt to place piece ---
        if not env.drop_piece(action):  # Invalid move → retry
            continue

        # --- Check for terminal state ---
        done = env.is_win(current_player) or env.is_draw()

        if done:
            if env.is_win(current_player):
                # Assign winner based on who played last
                winner = 'Q-agent' if current_player == 1 else 'DQN-agent'
                wins[winner] += 1
            else:
                wins['draw'] += 1
            break

        # --- Switch turns ---
        current_player = 3 - current_player  # Toggle between player 1 and 2
        env.switch_player()

    # Progress logging every 10 episodes
    if ep % 10 == 0:
        print(f"Completed {ep} games...")

# -----------------------------------------------------------------------------------
# Step 6: Print final results
# -----------------------------------------------------------------------------------
print("\n===== FINAL RESULTS =====")
print(f"Q-agent wins:   {wins['Q-agent']}")
print(f"DQN-agent wins: {wins['DQN-agent']}")
print(f"Draws:          {wins['draw']}")