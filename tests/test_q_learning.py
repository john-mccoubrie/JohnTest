# DESCRIPTION: This script contains unit tests for the Q-learning agent and the Connect 4 environment.
#              It ensures that the Q-learning agent can choose valid actions, learn correctly,
#              save/load its Q-table, and that the environment enforces valid moves and detects
#              game-ending conditions like win or draw properly.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] pytest Documentation: https://docs.pytest.org/
#              [2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
#              [3] Source code for Connect4Env and QlearningAgent

# -----------------------------------------------------------------------------------
# Step 1: Import required libraries and modules
# -----------------------------------------------------------------------------------
import pytest                                     # Testing framework for unit tests
import os                                         # Used to check file existence
from env.connect4_env import Connect4Env, ROWS, COLUMNS     # Import environment and constants
from agents.q_learning_agent import QlearningAgent          # Import Q-learning agent implementation

# -----------------------------------------------------------------------------------
# Step 2: Test if Q-learning agent always selects a valid action
# -----------------------------------------------------------------------------------
def test_choose_action_returns_valid_column():
    env = Connect4Env()                           # Initialize game environment
    agent = QlearningAgent()                      # Create a new Q-learning agent
    action = agent.choose_action(env)             # Agent selects an action
    assert action in env.available_actions(), "Agent selected invalid column"

# -----------------------------------------------------------------------------------
# Step 3: Test saving and loading of the Q-table to/from a file
# -----------------------------------------------------------------------------------
def test_q_table_load_and_save(tmp_path):
    agent = QlearningAgent()                      # Create Q-learning agent
    test_file = tmp_path / "test_q.pkl"           # Create temporary file path

    dummy_state = "dummy"                         # Define a fake state
    agent.q_table[(dummy_state, 0)] = 0.1         # Assign dummy Q-value to (state, action) pair

    agent.save(test_file)                         # Save Q-table to file
    assert os.path.exists(test_file), "Q-table file not saved"

    new_agent = QlearningAgent()                  # Create a fresh agent
    new_agent.load(test_file)                     # Load Q-table from file
    assert (dummy_state, 0) in new_agent.q_table, "Q-table not loaded properly"
    assert new_agent.q_table[(dummy_state, 0)] == 0.1  # Check if value is retained

# -----------------------------------------------------------------------------------
# Step 4: Test if the agent updates its Q-values after receiving a reward
# -----------------------------------------------------------------------------------
def test_agent_learns_after_win():
    env = Connect4Env()                           # Initialize environment
    agent = QlearningAgent()                      # Create Q-learning agent
    state = agent.encode_state(env.board)         # Encode initial board state
    action = env.available_actions()[0]           # Choose first available column

    env.drop_piece(action)                        # Apply move to environment
    next_state = agent.encode_state(env.board)    # Encode new board state
    agent.learn(state, action, reward=1, next_state=next_state,
                done=True, valid_actions=env.available_actions())  # Apply Q-learning update

    assert (state, action) in agent.q_table, "Agent did not store Q-value after learning"
    assert isinstance(agent.q_table[(state, action)], float), "Q-value is not numeric"

# -----------------------------------------------------------------------------------
# Step 5: Test if valid moves are accepted and pieces are correctly placed
# -----------------------------------------------------------------------------------
def test_valid_move_and_piece_drop():
    env = Connect4Env()                           # Create game environment
    col = 0                                       # Use the first column
    success = env.drop_piece(col)                 # Attempt to place piece
    assert success, "Valid move was rejected"     # Should succeed

    # Check if any row in that column now has a piece
    assert any(env.board[row][col] != 0 for row in range(ROWS)), "Piece was not placed correctly"

# -----------------------------------------------------------------------------------
# Step 6: Test that invalid moves (e.g., full column) are correctly rejected
# -----------------------------------------------------------------------------------
def test_invalid_move():
    env = Connect4Env()                           # Initialize game
    col = 0                                       # Target first column
    for _ in range(ROWS):                         # Fill the column
        env.drop_piece(col)
    assert not env.drop_piece(col), "Allowed move in full column"

# -----------------------------------------------------------------------------------
# Step 7: Test if the player switching logic works correctly
# -----------------------------------------------------------------------------------
def test_player_switch():
    env = Connect4Env()                           # New game
    original = env.current_player                 # Record initial player
    env.switch_player()                           # Switch to the next player
    assert env.current_player != original, "Player did not switch"

# -----------------------------------------------------------------------------------
# Step 8: Test win condition detection (vertical win in a single column)
# -----------------------------------------------------------------------------------
def test_win_detection():
    env = Connect4Env()                           # New game
    col = 0
    for _ in range(4):                            # Drop 4 pieces in the same column
        env.drop_piece(col)
    assert env.is_win(env.current_player), "Win condition not detected"

# -----------------------------------------------------------------------------------
# Step 9: Test draw detection when board is completely filled
# -----------------------------------------------------------------------------------
def test_draw_detection():
    env = Connect4Env()                           # New game
    for col in range(COLUMNS):                    # Loop over all columns
        for _ in range(ROWS):                     # Fill each column
            env.drop_piece(col)
    assert env.is_draw(), "Draw condition not detected"