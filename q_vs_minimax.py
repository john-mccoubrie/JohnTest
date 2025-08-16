# DESCRIPTION: This script evaluates the performance of a Q-learning agent against a Minimax agent
#              by running a series of Connect 4 games. It tracks and prints the number of wins and draws.
#              The Q-learning agent loads a pre-trained Q-table and uses an epsilon-greedy policy.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
#              [2] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.)
#              [3] StackOverflow. (2021). *How to benchmark two agents in Connect Four?*

# -----------------------------------------------------------------------------------
# Step 1: Import necessary classes and agents
# -----------------------------------------------------------------------------------
from env.connect4_env import Connect4Env            # Import the Connect 4 game environment
from agents.q_learning_agent import QlearningAgent  # Import the Q-learning agent
from agents.minimax_agent import MinimaxAgent       # Import the Minimax agent

# -----------------------------------------------------------------------------------
# Step 2: Initialize agents and evaluation settings
# -----------------------------------------------------------------------------------
games = 100                                 # Total number of games to play
minimax_agent = MinimaxAgent(depth=4)       # Instantiate Minimax agent with search depth 4
q_agent = QlearningAgent()                  # Instantiate Q-learning agent (hyperparameters default)
q_agent.load("q_table.pkl")                 # Load pre-trained Q-table from file

results = {'Q': 0, 'Minimax': 0, 'Draw': 0} # Dictionary to store game outcomes

# -----------------------------------------------------------------------------------
# Step 3: Run multiple games between Q-agent and Minimax agent
# -----------------------------------------------------------------------------------
for game in range(1, games + 1):                                # Loop over each game number
    env = Connect4Env()                                         # Create a new game environment
    env.reset()                                                 # Reset board to empty state
    current_player = 1                                          # Q-learning agent (Player 1) starts first

    while True:                                                 # Game loop: continues until win or draw
        if current_player == 1:                                 # If it's Q-agent's turn
            move = q_agent.choose_action(env)                   # Q-agent chooses a move using epsilon-greedy
        else:                                                   # If it's Minimax's turn
            move = minimax_agent.get_move(env)                  # Minimax chooses best move via search

        valid = env.drop_piece(move)                            # Drop the chosen piece into the board
        if not valid:                                           # If the move is invalid (e.g., full column), skip turn
            continue

        if env.is_win(current_player):                          # Check if current player has won
            winner = 'Q' if current_player == 1 else 'Minimax'  # Identify winner by player ID
            results[winner] += 1                                # Update win counter for the winner
            break                                               # Exit the game loop
        elif env.is_draw():                                     # Check for draw (no moves left)
            results['Draw'] += 1                                # Update draw counter
            break                                               # Exit the game loop
        else:
            env.switch_player()                                 # Switch to the other player
            current_player = 3 - current_player                 # Toggle player ID: 1 ↔ 2

    if game % 10 == 0:                                          # Print progress every 10 games
        print(f"Completed {game} games...")

# -----------------------------------------------------------------------------------
# Step 4: Print final aggregated results after all games
# -----------------------------------------------------------------------------------
print("\n===== FINAL RESULTS =====")
print(f"Q-learning wins:   {results['Q']}")         # Total wins by Q-learning agent
print(f"Minimax wins:      {results['Minimax']}")   # Total wins by Minimax agent
print(f"Draws:             {results['Draw']}")      # Total number of draws