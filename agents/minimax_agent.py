# DESCRIPTION: This script implements a Minimax-based AI agent for playing Connect 4.
#              The agent explores the game tree to a given depth, evaluates states using heuristics,
#              and uses alpha-beta pruning to reduce computation. It can play both offensively and defensively.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.).
#                  Pearson Education.
#              [2] GeeksForGeeks. (2023). *Minimax Algorithm in Game Theory*.
#                  https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/
#              [3] StackOverflow. (2020). *Why use deepcopy with minimax game trees?*
#                  https://stackoverflow.com/questions/11867516/deepcopy-in-a-minimax-algorithm

# -----------------------------------------------------------------------------------
# MinimaxAgent with Alpha-Beta Pruning, Move Ordering, Memoization, and Undo Move
# -----------------------------------------------------------------------------------
import copy
from env.connect4_env import Connect4Env, ROWS, COLUMNS  # Import environment and board size

class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth                    # Search depth for minimax lookahead
        self.player = 1                      # Agent's player ID (updated before move)
        self.cache = {}                     # Cache for memoization of board states

    # --------------------------------------------------------------------------------
    # Utility to create a unique hash for the board + current player for memoization
    # --------------------------------------------------------------------------------
    def board_hash(self, env):
        # Flatten the board into a tuple (immutable) plus current player
        return (tuple(env.board.flatten()), env.current_player)

    # --------------------------------------------------------------------------------
    # Move ordering to prioritize center columns for better pruning efficiency
    # --------------------------------------------------------------------------------
    def order_moves(self, valid_moves):
        center = COLUMNS // 2
        # Sort moves by distance from center column (closer to center is better)
        return sorted(valid_moves, key=lambda x: abs(center - x))

    # --------------------------------------------------------------------------------
    # Minimax search with alpha-beta pruning and memoization cache
    # --------------------------------------------------------------------------------
    def minimax(self, env, depth, maximizing_player, alpha, beta):
        state_key = (self.board_hash(env), depth, maximizing_player)
        if state_key in self.cache:               # Return cached result if available
            return self.cache[state_key]

        # Base case: terminal node or depth limit reached
        if depth == 0 or env.is_win(1) or env.is_win(2) or env.is_draw():
            eval_score = self.evaluate(env)
            self.cache[state_key] = (eval_score, None)
            return eval_score, None

        valid_moves = self.order_moves(env.available_actions())

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None

            for col in valid_moves:
                env.make_move(col)                  # Make the move (mutates env)
                env.switch_player()                # Switch player for next turn

                eval, _ = self.minimax(env, depth - 1, False, alpha, beta)  # Recurse minimizing

                env.switch_player()                # Undo player switch
                env.undo_move(col)                 # Undo the move to restore env

                if eval > max_eval:
                    max_eval = eval
                    best_move = col

                alpha = max(alpha, eval)           # Update alpha value
                if beta <= alpha:                  # Alpha-beta pruning cutoff
                    break

            self.cache[state_key] = (max_eval, best_move)
            return max_eval, best_move

        else:
            min_eval = float('inf')
            best_move = None

            for col in valid_moves:
                env.make_move(col)                  # Make the move
                env.switch_player()                # Switch player

                eval, _ = self.minimax(env, depth - 1, True, alpha, beta)   # Recurse maximizing

                env.switch_player()                # Undo player switch
                env.undo_move(col)                 # Undo the move

                if eval < min_eval:
                    min_eval = eval
                    best_move = col

                beta = min(beta, eval)             # Update beta value
                if beta <= alpha:                  # Alpha-beta pruning cutoff
                    break

            self.cache[state_key] = (min_eval, best_move)
            return min_eval, best_move

    # --------------------------------------------------------------------------------
    # Get best move for the current environment state
    # --------------------------------------------------------------------------------
    def get_move(self, env: Connect4Env):
        self.player = env.current_player        # Update agent's player ID from environment
        self.cache.clear()                      # Clear cache before each new move
        _, best_move = self.minimax(env, self.depth, True, float('-inf'), float('inf'))
        return best_move

    # --------------------------------------------------------------------------------
    # Heuristic evaluation of board states
    # --------------------------------------------------------------------------------
    def evaluate(self, env):
        if env.is_win(self.player):
            return 1000                       # Large positive score for winning
        elif env.is_win(3 - self.player):
            return -1000                      # Large negative score for opponent win

        board = env.board

        # Center column preference score
        center_col = board[:, COLUMNS // 2]
        center_score = list(center_col).count(self.player) * 3

        # Difference in total pieces on board
        player_pieces = (board == self.player).sum()
        opponent_pieces = (board == (3 - self.player)).sum()
        piece_count_score = player_pieces - opponent_pieces

        score = 0
        score += center_score
        score += self.count_windows(board, self.player, 3) * 5       # Score 3-in-a-row for agent
        score -= self.count_windows(board, 3 - self.player, 3) * 50  # Penalize opponent 3-in-a-row heavily
        score += piece_count_score

        return score

    # --------------------------------------------------------------------------------
    # Count number of windows (groups of 4) matching the pattern for heuristic
    # --------------------------------------------------------------------------------
    def count_windows(self, board, player, count):
        def check_window(window):
            # Check if window has exactly `count` of `player` pieces and rest empty
            return list(window).count(player) == count and list(window).count(0) == 4 - count

        score = 0

        # Horizontal windows
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                window = board[row, col:col + 4]
                if check_window(window):
                    score += 1

        # Vertical windows
        for row in range(ROWS - 3):
            for col in range(COLUMNS):
                window = board[row:row + 4, col]
                if check_window(window):
                    score += 1

        # Positive slope diagonals
        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                if check_window(window):
                    score += 1

        # Negative slope diagonals
        for row in range(3, ROWS):
            for col in range(COLUMNS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                if check_window(window):
                    score += 1

        return score