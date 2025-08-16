# DESCRIPTION: 
#   This script evaluates a Q-learning agent against a Deep Q-Network (DQN) agent 
#   in Connect 4 over multiple batches of games. 
#   Instead of running a single long simulation, results are grouped into batches 
#   for easier visualization of performance trends over time.
#
# LANGUAGE: PYTHON
# SOURCE(S):
#   [1] Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.
#   [2] Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
#   [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
#   [4] PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#   [5] StackOverflow – AI vs AI simulation discussion:
#       https://stackoverflow.com/questions/65943215/ai-vs-ai-in-reinforcement-learning

# -----------------------------------------------------------------------------------
# Step 1: Import dependencies
# -----------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from env.connect4_env import Connect4Env, ROWS, COLUMNS   # Game environment and constants
from agents.q_learning_agent import QlearningAgent        # Q-learning (tabular) agent
from agents.dqn_agent import DQNAgent                     # Deep Q-Network agent

# -----------------------------------------------------------------------------------
# Step 2: Define batch execution function
# -----------------------------------------------------------------------------------
def run_batch(batch_size, q_agent, dqn_agent):
    """
    Runs a fixed number of games (batch) between the Q-learning and DQN agents.
    
    Args:
        batch_size (int): Number of games to play in this batch.
        q_agent (QlearningAgent): Pre-trained Q-learning agent.
        dqn_agent (DQNAgent): Pre-trained Deep Q-Network agent.
    
    Returns:
        dict: Game results in the form {'Q': wins_by_Q, 'DQN': wins_by_DQN, 'Draw': draws}.
    """
    results = {'Q': 0, 'DQN': 0, 'Draw': 0}

    for _ in range(batch_size):
        # --- Initialize new game ---
        env = Connect4Env()
        env.reset()
        current_player = 1  # Q-agent always starts

        while True:
            # --- Agent selects a move ---
            if current_player == 1:
                move = q_agent.choose_action(env)    # Q-learning move selection
            else:
                move = dqn_agent.act(env)            # DQN move selection

            # --- Attempt to drop the piece ---
            if not env.drop_piece(move):  # Invalid move → retry
                continue

            # --- Check for win/draw conditions ---
            if env.is_win(current_player):
                winner = 'Q' if current_player == 1 else 'DQN'
                results[winner] += 1
                break
            elif env.is_draw():
                results['Draw'] += 1
                break
            else:
                # Switch turns
                env.switch_player()
                current_player = 3 - current_player  # Toggle player (1 ↔ 2)

    return results

# -----------------------------------------------------------------------------------
# Step 3: Initialize agents and load pre-trained models
# -----------------------------------------------------------------------------------
# --- Q-learning Agent ---
q_agent = QlearningAgent()
q_agent.load("q_table.pkl")  # Load trained Q-table

# --- DQN Agent ---
state_size = ROWS * COLUMNS  # Flattened board representation
action_size = COLUMNS        # Number of columns in Connect 4
dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.load("dqn_model.pth")  # Load trained neural network model

# -----------------------------------------------------------------------------------
# Step 4: Simulation parameters
# -----------------------------------------------------------------------------------
batches = 20       # Number of batches to run
batch_size = 50    # Games per batch
q_win_percent = [] # Store Q-agent win rate per batch for plotting

# -----------------------------------------------------------------------------------
# Step 5: Run evaluation over batches
# -----------------------------------------------------------------------------------
for i in range(batches):
    result = run_batch(batch_size, q_agent, dqn_agent)
    win_rate = result['Q'] / batch_size
    q_win_percent.append(win_rate)

    print(f"Batch {i+1}: Q wins = {result['Q']}, DQN = {result['DQN']}, Draws = {result['Draw']}")

# -----------------------------------------------------------------------------------
# Step 6: Plot results
# -----------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, batches + 1), [w * 100 for w in q_win_percent], marker='o')
plt.title("Q-Learning Agent Win Rate Over Batches vs DQN Agent")
plt.xlabel("Batch # (50 games each)")
plt.ylabel("Q-Agent Win Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("q_vs_dqn_winrate.png")  # Save plot to file
plt.show()