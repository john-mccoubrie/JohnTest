# DESCRIPTION: This script evaluates the performance of a trained Deep Q-Network (DQN) agent
#              against a Minimax agent in Connect 4 over multiple batches of games. It tracks
#              win/draw rates per batch and plots the DQN agent’s win percentage progression.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
#              [2] Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.).
#              [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
#              [4] PyTorch Documentation – DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#              [5] StackOverflow. (2021). *How to run AI agents in batches for evaluation?*
#                  https://stackoverflow.com/questions/64871143/running-batches-of-games-for-ai-comparison

# -----------------------------------------------------------------------------------
# Step 1: Import required modules
# -----------------------------------------------------------------------------------
import matplotlib.pyplot as plt                              # For plotting win rates over time
from env.connect4_env import Connect4Env, ROWS, COLUMNS      # Connect 4 environment and board constants
from agents.dqn_agent import DQNAgent                        # Deep Q-Network agent
from agents.minimax_agent import MinimaxAgent                # Minimax search agent

# -----------------------------------------------------------------------------------
# Step 2: Function to run a batch of games
# -----------------------------------------------------------------------------------
def run_batch(batch_size, dqn_agent, minimax_agent):
    """
    Plays a batch of Connect 4 games between the DQN agent and Minimax agent.

    Args:
        batch_size (int): Number of games to play in this batch.
        dqn_agent (DQNAgent): Pre-trained DQN agent instance.
        minimax_agent (MinimaxAgent): Minimax agent instance.

    Returns:
        dict: Dictionary with counts of wins for each agent and draws.
    """
    results = {'DQN': 0, 'Minimax': 0, 'Draw': 0}  # Initialize results counter

    # Play 'batch_size' number of games
    for _ in range(batch_size):
        env = Connect4Env()                       # Reset game environment
        env.reset()
        current_player = 1                        # Player 1 = DQN starts first

        while True:
            # --- Select move based on current player ---
            if current_player == 1:
                move = dqn_agent.act(env)         # DQN chooses action
            else:
                move = minimax_agent.get_move(env) # Minimax chooses action

            # --- Execute move ---
            if not env.drop_piece(move):          # Invalid move → retry turn
                continue

            # --- Check for win/draw ---
            if env.is_win(current_player):
                winner = 'DQN' if current_player == 1 else 'Minimax'
                results[winner] += 1
                break
            elif env.is_draw():
                results['Draw'] += 1
                break

            # --- Switch turns ---
            env.switch_player()
            current_player = 3 - current_player   # Toggle between 1 and 2

    return results  # Return aggregate results for this batch

# -----------------------------------------------------------------------------------
# Step 3: Initialize agents
# -----------------------------------------------------------------------------------
# --- DQN Agent ---
state_size = ROWS * COLUMNS                       # Flattened board state size
action_size = COLUMNS                             # Number of possible moves (columns)
dqn_agent = DQNAgent(state_size, action_size)     # Create DQN instance
dqn_agent.load("dqn_model.pth")                   # Load pre-trained model weights

# --- Minimax Agent ---
minimax_agent = MinimaxAgent(depth=4)             # Deeper search for stronger play

# -----------------------------------------------------------------------------------
# Step 4: Batch configuration
# -----------------------------------------------------------------------------------
batches = 20                                      # Total number of batches
batch_size = 50                                   # Games per batch
dqn_win_percent = []                              # Store DQN win rate for each batch

# -----------------------------------------------------------------------------------
# Step 5: Run evaluation over multiple batches
# -----------------------------------------------------------------------------------
for i in range(batches):
    result = run_batch(batch_size, dqn_agent, minimax_agent)
    win_rate = result['DQN'] / batch_size
    dqn_win_percent.append(win_rate)

    # Print progress for monitoring
    print(f"Batch {i+1}: DQN wins = {result['DQN']}, "
          f"Minimax = {result['Minimax']}, Draws = {result['Draw']}")

# -----------------------------------------------------------------------------------
# Step 6: Plot DQN win rate progression
# -----------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, batches + 1), [w * 100 for w in dqn_win_percent],
         marker='o', label="DQN Win Rate")

plt.title("DQN Agent Win Rate Over Batches vs Minimax Agent")
plt.xlabel("Batch # (50 games each)")
plt.ylabel("DQN-Agent Win Rate (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot as PNG file and display
plt.savefig("dqn_vs_minimax_winrate.png")
plt.show()