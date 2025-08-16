# DESCRIPTION: This script evaluates the performance of a Q-learning agent against a Minimax agent
#              over multiple batches of Connect 4 games. It records win rates and visualizes
#              the Q-agent's performance trend over time using Matplotlib.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*
#              [2] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach*
#              [3] Matplotlib Docs: https://matplotlib.org/stable/index.html

# -----------------------------------------------------------------------------------
# Step 1: Import necessary libraries and agent/environment classes
# -----------------------------------------------------------------------------------
import matplotlib.pyplot as plt                     # For plotting performance results
from env.connect4_env import Connect4Env            # Connect 4 environment logic
from agents.q_learning_agent import QlearningAgent  # Q-learning agent for reinforcement learning
from agents.minimax_agent import MinimaxAgent       # Minimax agent for adversarial search

# -----------------------------------------------------------------------------------
# Step 2: Define function to run a batch of games between Q and Minimax agents
# -----------------------------------------------------------------------------------
def run_batch(batch_size, q_agent, minimax_agent):
    results = {'Q': 0, 'Minimax': 0, 'Draw': 0}                                                         # Initialize result counters

    for _ in range(batch_size):                                                                         # Loop over number of games in the batch
        env = Connect4Env()                                                                             # Create a fresh game environment
        env.reset()                                                                                     # Reset board and player to initial state
        current_player = 1                                                                              # Player 1 is Q-learning agent

        while True:
            move = q_agent.choose_action(env) if current_player == 1 else minimax_agent.get_move(env)   # Choose action based on which player is currently active
            
            if not env.drop_piece(move):                                                                # If move is invalid (column full), retry turn
                continue

            if env.is_win(current_player):                                                              # Check for win condition after the move
                winner = 'Q' if current_player == 1 else 'Minimax'
                results[winner] += 1                                                                    # Increment win counter
                break

            elif env.is_draw():                                                                         # Check for draw condition (board full, no winner)
                results['Draw'] += 1
                break

            else:
                env.switch_player()                                                                     # Switch to the other player
                current_player = 3 - current_player                                                     # Toggle between 1 and 2

    return results                                                                                      # Return final game outcomes for the batch

# -----------------------------------------------------------------------------------
# Step 3: Initialize agents
# -----------------------------------------------------------------------------------
q_agent = QlearningAgent()              # Create Q-learning agent
q_agent.load("q_table.pkl")             # Load pre-trained Q-table from file

minimax_agent = MinimaxAgent(depth=4)   # Create Minimax agent with depth limit 4

# -----------------------------------------------------------------------------------
# Step 4: Run multiple batches and collect Q-agent win rates
# -----------------------------------------------------------------------------------
batches = 20                                                                                                # Number of evaluation batches
batch_size = 50                                                                                             # Games per batch
q_win_percent = []                                                                                          # List to store Q-agent win rates

for i in range(batches):                                                                                    # Loop through each batch
    result = run_batch(batch_size, q_agent, minimax_agent)                                                  # Simulate a batch of games
    win_rate = result['Q'] / batch_size                                                                     # Calculate win rate as a proportion
    q_win_percent.append(win_rate)                                                                          # Store win rate for plotting
    print(f"Batch {i+1}: Q wins = {result['Q']}, Minimax = {result['Minimax']}, Draws = {result['Draw']}")

# -----------------------------------------------------------------------------------
# Step 5: Visualize Q-agent performance over batches
# -----------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))                          # Create figure with defined size
plt.plot(range(1, batches + 1),                     # X-axis: batch numbers 1 to 20
         [w * 100 for w in q_win_percent],          # Y-axis: win rate in percentage
         marker='o')                                # Use circles to mark data points

plt.title("Q-Learning Agent Win Rate Over Batches") # Plot title
plt.xlabel("Batch # (20 games each)")               # X-axis label
plt.ylabel("Q-Agent Win Rate (%)")                  # Y-axis label
plt.grid(True)                                      # Enable grid for readability
plt.tight_layout()                                  # Adjust layout to prevent clipping
plt.savefig("q_vs_minimax_winrate.png")             # Save figure to file
plt.show()                                          # Display the plot