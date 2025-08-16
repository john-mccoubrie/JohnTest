# DESCRIPTION: This script runs an interactive Connect 4 game where a human (Player 1)
#              plays against a Minimax AI agent (Player 2). The human selects columns via input,
#              while the Minimax agent chooses moves using depth-limited minimax search.
#              The game continues until a win or draw condition is reached.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.)
#              [2] Python Docs: https://docs.python.org/3/
#              [3] StackOverflow. (2022). *How to build an interactive board game in Python?*

# -----------------------------------------------------------------------------------
# Step 1: Import environment and Minimax agent
# -----------------------------------------------------------------------------------
from env.connect4_env import Connect4Env        # Import Connect 4 game logic and board state handling
from agents.minimax_agent import MinimaxAgent   # Import AI agent using Minimax search algorithm

# -----------------------------------------------------------------------------------
# Step 2: Initialize environment and agent
# -----------------------------------------------------------------------------------
env = Connect4Env()             # Create new game environment
agent = MinimaxAgent(depth=3)   # Instantiate Minimax agent with search depth of 3
env.reset()                     # Reset environment to empty board and Player 1
env.render()                    # Display the initial empty board

# -----------------------------------------------------------------------------------
# Step 3: Begin the game loop
# -----------------------------------------------------------------------------------
while True:
    if env.current_player == 1:                     # If it's the human player's turn
        col = int(input("Enter column (0–6): "))    # Prompt for human move (must be integer 0-6)
    else:                                           # If it's the Minimax agent's turn
        col = agent.get_move(env)                   # Get AI's move using Minimax
        print(f"Minimax chooses column: {col}")     # Display AI's chosen column

    if not env.drop_piece(col):                     # Try placing piece in the selected column
        print("Invalid move. Try again.")           # If column is full or invalid, show message
        continue                                    # Skip the rest of loop and re-prompt or retry

    env.render()                                    # Display the current state of the board

    if env.is_win(env.current_player):              # Check if the current player has won
        print(f"Player {env.current_player} wins!") # Announce winner
        break                                       # Exit game loop
    elif env.is_draw():                             # Check for draw condition (board is full)
        print("It's a draw.")                       # Announce draw
        break                                       # Exit game loop
    else:
        env.switch_player()                         # Switch to the other player (1 ↔ 2)