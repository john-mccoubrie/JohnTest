# DESCRIPTION: This script trains a Q-learning agent to play Connect 4 by playing against a random opponent.
#              The agent learns by updating its Q-table based on game outcomes and adjusts over 10,000 episodes.
#              Periodic summaries track win/loss/draw counts and exploration rate (ε).
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] Sutton & Barto (2018) - Reinforcement Learning: An Introduction
#              [2] Russell & Norvig - AI: A Modern Approach
#              [3] Q-learning Theory: https://en.wikipedia.org/wiki/Q-learning

# -----------------------------------------------------------------------------------
# Step 1: Import required modules and initialize environment/agent
# -----------------------------------------------------------------------------------
from env.connect4_env import Connect4Env            # Import Connect 4 environment
from agents.q_learning_agent import QlearningAgent  # Import Q-learning agent class
import random                                       # Used for opponent's random action selection

# -----------------------------------------------------------------------------------
# Step 2: Set training configuration and initialize Q-agent
# -----------------------------------------------------------------------------------
episodes = 10000                # Number of training games
agent = QlearningAgent()        # Instantiate the Q-learning agent
agent.load("q_table.pkl")       # Load an existing Q-table if available
wins = {1: 0, 2: 0, 'draw': 0}  # Keys: player 1 = Q-agent, player 2 = random agent

# -----------------------------------------------------------------------------------
# Step 3: Start training over multiple episodes
# -----------------------------------------------------------------------------------
for ep in range(1, episodes + 1):           # Loop through each training episode
    env = Connect4Env()                     # Initialize a new game
    state = agent.encode_state(env.board)   # Encode current board into a state key for Q-table
    done = False                            # Game-over flag
    current_player = 1                      # Start with player 1 (Q-agent)

    # ------------------------------
    # Episode loop: until win or draw
    # ------------------------------
    while not done:
        if current_player == 1:                             # Agent action for Q-agent (player 1); random action for opponent (player 2)
            action = agent.choose_action(env)               # Choose best or exploratory action
        else:
            action = random.choice(env.available_actions()) # Random agent move

        success = env.drop_piece(action)                    # Try to apply the move on the board
        if not success:
            continue                                        # Retry if move is invalid (e.g., column full)

        next_state = agent.encode_state(env.board)          # Encode the next board state
        done = env.is_win(current_player) or env.is_draw()  # Check if game has ended

        # ------------------------------
        # Reward assignment and Q-table update
        # ------------------------------
        if env.is_win(current_player):                                                          # If the current player wins
            reward = 1 if current_player == 1 else -1                                           # Reward Q-agent for win; punish if opponent wins
            agent.learn(state, action, reward, next_state, done, env.available_actions())       # Q-learning update
            wins[current_player] += 1                                                           # Increment the winner's count
        elif env.is_draw():                                                                     # Handle draw case
            reward = 0.5                                                                        # Neutral reward for both agents
            agent.learn(state, action, reward, next_state, done, env.available_actions())       # Update Q-table
            wins['draw'] += 1                                                                   # Increment draw counter
        else:
            reward = 0                                                                          # No reward for intermediate step
            if current_player == 1:                                                             # Only Q-agent learns
                agent.learn(state, action, reward, next_state, done, env.available_actions())

        # ------------------------------
        # Prepare for next turn
        # ------------------------------
        state = next_state                  # Move to new state
        current_player = 3 - current_player # Switch player (1 <-> 2)
        env.switch_player()                 # Sync environment's player state

    # -----------------------------------------------------------------------------------
    # Step 4: Log training progress every 500 episodes
    # -----------------------------------------------------------------------------------
    if ep % 500 == 0:
        print(f"Episode {ep} - Wins: {wins[1]} | Losses: {wins[2]} | Draws: {wins['draw']} | ε={agent.epsilon:.3f}")

# -----------------------------------------------------------------------------------
# Step 5: Save the learned Q-table to disk
# -----------------------------------------------------------------------------------
agent.save("q_table.pkl")    # Persist the learned Q-values for later use