# DESCRIPTION: This script defines the Connect4 game environment for use in reinforcement learning or AI agents.
#              The environment supports standard Connect 4 rules with a 6x7 board. It provides methods to:
#              - Drop a piece
#              - Switch players
#              - Check for win/draw conditions
#              - Return available actions
#              - Render the board
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Official Connect Four Game Rules (Milton Bradley/Hasbro)
#              [2] NumPy Documentation. https://numpy.org/doc/stable/
#              [3] OpenAI Gym-style API conventions

# -----------------------------------------------------------------------------------
# Step 1: Import necessary library
# -----------------------------------------------------------------------------------
import numpy as np  # NumPy is used for efficient 2D array operations

# -----------------------------------------------------------------------------------
# Step 2: Define constants for board dimensions
# -----------------------------------------------------------------------------------
ROWS = 6    # Number of rows on the Connect 4 board
COLUMNS = 7 # Number of columns on the Connect 4 board

# -----------------------------------------------------------------------------------
# Step 3: Define the Connect4Env class
# -----------------------------------------------------------------------------------
class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLUMNS), dtype=int)   # Create an empty board initialized with 0s
        self.current_player = 1                             # Player 1 starts by default

    # -----------------------------------------------------------------------------------
    # Reset the environment to its initial state
    # -----------------------------------------------------------------------------------
    def reset(self):
        self.board = np.zeros((ROWS, COLUMNS), dtype=int)   # Reset board to all zeros
        self.current_player = 1                             # Reset to player 1's turn
        return self.board                                   # Return the reset board

    # -----------------------------------------------------------------------------------
    # Return a list of valid (non-full) column indices for the next move
    # -----------------------------------------------------------------------------------
    def available_actions(self):
        return [col for col in range(COLUMNS) if self.board[0][col] == 0]   # Only columns where the top row is empty are available

    # -----------------------------------------------------------------------------------
    # Drop the current player's piece into the specified column
    # -----------------------------------------------------------------------------------
    def drop_piece(self, col):
        if col not in self.available_actions():             # Invalid move if column is full
            return False

        for row in reversed(range(ROWS)):                   # Start from bottom row upwards
            if self.board[row][col] == 0:                   # Find first empty row
                self.board[row][col] = self.current_player  # Place player's piece
                return True                                 # Return success
        return False                                        # Fallback (should not occur)

    # -----------------------------------------------------------------------------------
    # Switch turns between players (1 ↔ 2)
    # -----------------------------------------------------------------------------------
    def switch_player(self):
        self.current_player = 3 - self.current_player   # Toggles between 1 and 2

    # -----------------------------------------------------------------------------------
    # Check whether the given player has won the game
    # -----------------------------------------------------------------------------------
    def is_win(self, player):
        for r in range(ROWS):                                               # Check horizontal wins (left to right)
            for c in range(COLUMNS - 3):                                    # Ensure 4-in-a-row fits
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True

        for r in range(ROWS - 3):                                           # Check vertical wins (top to bottom)
            for c in range(COLUMNS):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True

        for r in range(ROWS - 3):                                           # Check positive diagonal (\) wins
            for c in range(COLUMNS - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True

        for r in range(3, ROWS):                                            # Check negative diagonal (/) wins
            for c in range(COLUMNS - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True

        return False                                                        # No winning pattern found

    # -----------------------------------------------------------------------------------
    # Check whether the game is a draw (i.e., board is full)
    # -----------------------------------------------------------------------------------
    def is_draw(self):
        return all(self.board[0][c] != 0 for c in range(COLUMNS))   # If all top-row positions are filled, no moves left → draw

    # -----------------------------------------------------------------------------------
    # Print the current board state to the console
    # -----------------------------------------------------------------------------------
    def render(self):
        print(np.flip(self.board, 0))   # Flip vertically so bottom row is displayed at bottom

    def make_move(self, col):
        """Drop piece in column, return True if successful."""
        for row in reversed(range(ROWS)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                return True
        return False  # Column full

    def undo_move(self, col):
        """Remove topmost piece from column."""
        for row in range(ROWS):
            if self.board[row][col] != 0:
                self.board[row][col] = 0
                return