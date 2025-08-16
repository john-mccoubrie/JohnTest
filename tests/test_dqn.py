# DESCRIPTION: This script tests the functionality of the DQNAgent class for playing Connect 4.
#              It verifies correct initialization, state encoding, action selection, memory handling,
#              training updates, and model saving/loading behavior.
# LANGUAGE:    PYTHON
# SOURCE(S):   [1] pytest Documentation. https://docs.pytest.org/en/stable/
#              [2] Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning".
#              [3] StackOverflow. (2023). *Testing PyTorch-based RL agents*.

# -----------------------------------------------------------------------------------
# Step 1: Import libraries and modules needed for testing
# -----------------------------------------------------------------------------------
import os
import numpy as np
import torch
import pytest
from agents.dqn_agent import DQNAgent           # Imports the DQNAgent class to be tested
from env.connect4_env import Connect4Env, ROWS, COLUMNS  # Imports Connect4Env and board constants

# -----------------------------------------------------------------------------------
# Step 2: Test that the agent initializes correctly with matching policy and target networks
# -----------------------------------------------------------------------------------
def test_dqn_initialization():
    state_size = ROWS * COLUMNS
    action_size = COLUMNS
    agent = DQNAgent(state_size, action_size)
    assert isinstance(agent.policy_net, torch.nn.Module)
    assert isinstance(agent.target_net, torch.nn.Module)
    # Target net should start with same weights as policy net
    for p1, p2 in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p1, p2)

# -----------------------------------------------------------------------------------
# Step 3: Test that encode_state returns a correct flat float32 array
# -----------------------------------------------------------------------------------
def test_dqn_encode_state_shape_and_dtype():
    env = Connect4Env()
    agent = DQNAgent(ROWS * COLUMNS, COLUMNS)
    encoded = agent.encode_state(env.board)
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (ROWS * COLUMNS,)
    assert encoded.dtype == np.float32

# -----------------------------------------------------------------------------------
# Step 4: Test that act() returns a valid legal action from environment
# -----------------------------------------------------------------------------------
def test_dqn_act_returns_valid_action():
    env = Connect4Env()
    agent = DQNAgent(ROWS * COLUMNS, COLUMNS)
    move = agent.act(env)
    assert move in env.available_actions()

# -----------------------------------------------------------------------------------
# Step 5: Test that remember() correctly stores experiences in memory
# -----------------------------------------------------------------------------------
def test_dqn_remember_and_replay():
    agent = DQNAgent(ROWS * COLUMNS, COLUMNS, batch_size=4)
    state = np.zeros(agent.state_size, dtype=np.float32)
    next_state = np.ones(agent.state_size, dtype=np.float32)
    # Add a single memory
    agent.remember(state, 0, 1.0, next_state, False)
    assert len(agent.memory) == 1
    # Fill memory so replay can run
    for _ in range(agent.batch_size):
        agent.remember(state, 0, 1.0, next_state, False)
    old_epsilon = agent.epsilon
    agent.replay()
    assert agent.epsilon <= old_epsilon

# -----------------------------------------------------------------------------------
# Step 6: Test that save() and load() correctly persist and restore network weights
# -----------------------------------------------------------------------------------
def test_dqn_save_and_load(tmp_path):
    model_path = tmp_path / "dqn_model_test.pth"
    agent = DQNAgent(ROWS * COLUMNS, COLUMNS)
    agent.save(model_path)
    assert os.path.exists(model_path)
    # Create a new agent and load saved weights
    new_agent = DQNAgent(agent.state_size, agent.action_size)
    new_agent.load(model_path)
    for p1, p2 in zip(agent.policy_net.parameters(), new_agent.policy_net.parameters()):
        assert torch.equal(p1, p2)