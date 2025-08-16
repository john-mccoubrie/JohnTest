# AI801
Connect 4

Design and implementation details
Game Environment:
  •	A custom Connect 4 environment will be implemented using a 6x7 matrix that represents the Connect 4 game board. The environment will include methods to check for legal moves, actions applied, detecting wins, losses, and draws, and switching turns between players. This logic will be isolated from the AI agents.
Minimax Agent:
  •	The minimax agent will use a depth-limited tree search to evaluate possible future board states. Additionally, we will implement alpha-beta pruning to optimize performance by skipping over branches that do not affect the final decision. A heuristic function will score board states based on patterns such as two or three in a row combinations.
Reinforcement Learning Agent:
  •	The Q-learning agent will learn optimal moves by updating a Q-table based on reward received for actions taken in different game states. A reinforcement learning (RL) agent will be trained through self play. The RL agent will not be given any built-in knowledge about winning strategies, but will learn from trial and error. It will use the Q-table to refence wins or losses based on previous moves.
Comparison Framework:
  •	We will use agent vs agent, human vs agent, human vs human, and rule-based agents to compare various result sets. We will track wins/draws/losses, average number of moves per game, repeat moves, etc. To ensure the agents are working as intended.
Visualization:
  •	We will use Matplotlib to generate plots showing the progression of game states, win/loss trends over multiple games, and Q-value convergence for the reinforcement learning agent. A simple GUI or command-line visualization will display the game board after each move, helping in debugging and demonstrating AI decision-making during presentations.
Testing/Debugging:
  •	Unit tests will be created for game logic (win detections, move validation, evinronment setup, etc.) and integration tests will simulate full games between agents. Human play testing will be used to detect bugs, unexpected behaviors, and accuracy of results.
Tools and Libraries:
  •	Python 3.10+
  •	NumPy
  •	Matplotlib
  •	PyTorch
  •	Git

Implementation
  •	The Connect 4 AI project will be implemented using:
  •	Python 3.10+: Core language for game logic and AI algorithms. 
  •	OpenAI Gym: To provide a familiar API structure for the game environment, supporting agent interaction.
  •	PyTorch: For reinforcement learning (Q-learning/DQN).
  •	Matplotlib: For visualization of game state evolution and agent performance.
  •	Git: Version control and collaboration.
