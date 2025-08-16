# DESCRIPTION: This script tests the functionality of the MinimaxAgent class used for playing Connect 4.
#              It checks whether the agent selects valid moves and responds correctly in winning situations.
#              These tests simulate controlled board states and verify the behavior of the minimax logic.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] pytest Documentation. https://docs.pytest.org/en/stable/
#              [2] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.)
#              [3] StackOverflow. (2022). *How to test AI agents using mock environments?*

# -----------------------------------------------------------------------------------
# Step 1: Import libraries and modules needed for testing
# -----------------------------------------------------------------------------------
import pytest                                  # Imports pytest framework for writing and running unit tests
from agents.minimax_agent import MinimaxAgent  # Imports the MinimaxAgent class to be tested
from env.connect4_env import Connect4Env       # Imports the Connect4 environment to simulate gameplay

# -----------------------------------------------------------------------------------
# Step 2: Test that the Minimax agent returns a valid move from a fresh board
# -----------------------------------------------------------------------------------
def test_minimax_returns_valid_move():
    env = Connect4Env()                          # Initializes a new Connect 4 environment
    agent = MinimaxAgent(depth=2)                # Creates a MinimaxAgent with depth 2 for fast evaluation
    move = agent.get_move(env)                   # Gets the move recommended by the agent
    assert move in env.available_actions()       # Asserts that the move is within valid column indices

# -----------------------------------------------------------------------------------
# Step 3: Test that Minimax agent makes the winning move when it's available
# -----------------------------------------------------------------------------------
def test_minimax_selects_winning_move():
    env = Connect4Env()                          # Initializes a new Connect 4 game environment
    env.board[5][0:3] = 1                        # Manually sets up a near-win scenario for player 1
    env.current_player = 1                       # Sets the agent as the current player
    agent = MinimaxAgent(depth=2)                # Instantiates MinimaxAgent with reasonable depth
    move = agent.get_move(env)                   # Computes the move
    assert move == 3                             # Expects agent to drop in column 3 to win the game

# -----------------------------------------------------------------------------------
# Step 4: Test that Minimax agent blocks opponent’s winning move
# -----------------------------------------------------------------------------------
def test_minimax_blocks_opponent_win():
    env = Connect4Env()                          # Sets up a new game
    env.board[5][0:3] = 2                        # Opponent has three in a row at bottom row
    env.current_player = 1                       # Agent is player 1 and must block opponent
    agent = MinimaxAgent(depth=4)                # Uses depth 2 to allow one lookahead
    move = agent.get_move(env)                   # Gets move to respond
    assert move == 3                             # Expects the agent to block column 3

# -----------------------------------------------------------------------------------
# Step 5: Test that Minimax handles full board without crashing
# -----------------------------------------------------------------------------------
def test_minimax_handles_full_board():
    env = Connect4Env()                          # Initializes Connect 4 environment
    env.board[:, :] = 1                          # Fills the board to simulate draw
    env.current_player = 1                       # Agent’s turn
    agent = MinimaxAgent(depth=2)                # Creates agent
    move = agent.get_move(env)                   # Should gracefully return None
    assert move is None                          # No valid move expected on full board

# -----------------------------------------------------------------------------------
# Step 6: Test that evaluation function prefers own pieces
# -----------------------------------------------------------------------------------
def test_evaluation_score_direction():
    env = Connect4Env()                          # Creates a new board
    agent = MinimaxAgent(depth=1)                # Agent with depth 1
    env.board[5][0:2] = 1                        # Agent has two pieces
    env.board[5][2:4] = 2                        # Opponent has two pieces
    env.current_player = 1                       # Agent is player 1
    agent.player = 1                             # Set internal player
    score = agent.evaluate(env)                  # Compute heuristic score
    assert score == 0                            # Two pieces each → score = 0

# -----------------------------------------------------------------------------------
# Step 7: Test that evaluation increases with more agent pieces
# -----------------------------------------------------------------------------------
def test_evaluation_prefers_more_agent_pieces():
    env = Connect4Env()                          # New game board
    agent = MinimaxAgent(depth=1)                # Agent with shallow depth
    env.board[5][0:3] = 1                        # Agent has three
    env.board[5][3:4] = 2                        # Opponent has one
    env.current_player = 1                       # Agent's turn
    agent.player = 1                             # Set internal player
    score = agent.evaluate(env)                  # Score should be 2 (3 - 1)
    assert score > 0                             # Confirms correct evaluation behavior