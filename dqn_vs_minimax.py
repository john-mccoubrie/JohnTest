# DESCRIPTION: This script runs multiple Connect 4 games between a pre-trained Deep Q-Network (DQN) agent
#              and a Minimax agent. The DQN agent plays first in each game. Results are tracked and displayed
#              after all episodes.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
#              [2] Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.).
#              [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
#              [4] PyTorch Documentation – DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#              [5] StackOverflow. (2021). *How to compare two AI agents in a game loop?*
#                  https://stackoverflow.com/questions/32357279/pitting-two-ai-algorithms-against-each-other

# -----------------------------------------------------------------------------------
# Step 1: Import required modules and agents
# -----------------------------------------------------------------------------------
from env.connect4_env import Connect4Env, ROWS, COLUMNS   # Connect 4 environment and board dimensions
from agents.dqn_agent import DQNAgent                     # Deep Q-Network agent
from agents.minimax_agent import MinimaxAgent             # Minimax search agent

# -----------------------------------------------------------------------------------
# Step 2: Initialize the agents
# -----------------------------------------------------------------------------------
# --- DQN Agent ---
state_size = ROWS * COLUMNS                               # Flattened board size
action_size = COLUMNS                                     # Number of possible moves (columns)
dqn_agent = DQNAgent(state_size, action_size)             # Create DQN agent instance
dqn_agent.load('dqn_model.pth')                           # Load pre-trained weights

# --- Minimax Agent ---
minimax_agent = MinimaxAgent(depth=2)                     # Search depth = 2 plies
minimax_agent.cache = {}                                  # Reset memoization cache

# -----------------------------------------------------------------------------------
# Step 3: Match configuration
# -----------------------------------------------------------------------------------
episodes = 1000                                           # Number of games to play
wins = {'DQN-agent': 0, 'Minimax-agent': 0, 'draw': 0}    # Win/draw counters

# -----------------------------------------------------------------------------------
# Step 4: Play multiple games
# -----------------------------------------------------------------------------------
for ep in range(1, episodes + 1):
    env = Connect4Env()                                   # Reset environment
    done = False
    current_player = 1                                    # Player 1 = DQN agent

    while not done:
        # --- Select action ---
        if current_player == 1:
            action = dqn_agent.act(env)                   # DQN selects action
        else:
            action = minimax_agent.get_move(env)          # Minimax selects action

        # --- Apply action ---
        if not env.drop_piece(action):                    # Invalid move → retry
            continue

        # --- Check game termination ---
        done = env.is_win(current_player) or env.is_draw()

        if done:
            if env.is_win(current_player):
                winner = 'DQN-agent' if current_player == 1 else 'Minimax-agent'
                wins[winner] += 1
            else:
                wins['draw'] += 1
            break

        # --- Switch players ---
        current_player = 3 - current_player                # Toggle between 1 and 2
        env.switch_player()

    # --- Progress log ---
    if ep % 10 == 0:
        print(f"Completed {ep} games...")

# -----------------------------------------------------------------------------------
# Step 5: Display final results
# -----------------------------------------------------------------------------------
print("\n===== FINAL RESULTS =====")
print(f"DQN-agent wins:     {wins['DQN-agent']}")
print(f"Minimax-agent wins: {wins['Minimax-agent']}")
print(f"Draws:              {wins['draw']}")