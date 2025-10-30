"""Custom environments for reinforcement learning."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any


class GridWorldEnv(gym.Env):
    """A simple grid world environment for testing RL algorithms."""
    
    def __init__(self, size: int = 5, goal_reward: float = 10.0, step_penalty: float = -0.1):
        """Initialize the grid world environment.
        
        Args:
            size: Size of the grid (size x size)
            goal_reward: Reward for reaching the goal
            step_penalty: Penalty for each step taken
        """
        super().__init__()
        
        self.size = size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(size * size)
        
        # Initialize state
        self.state = 0
        self.goal_state = size * size - 1
        
        # Action mappings
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.state = 0
        return self.state, {}
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take a step in the environment."""
        # Get direction from action
        direction = self.action_to_direction[action]
        
        # Calculate new position
        row, col = divmod(self.state, self.size)
        new_row = max(0, min(self.size - 1, row + direction[0]))
        new_col = max(0, min(self.size - 1, col + direction[1]))
        
        # Update state
        self.state = new_row * self.size + new_col
        
        # Calculate reward
        if self.state == self.goal_state:
            reward = self.goal_reward
            terminated = True
        else:
            reward = self.step_penalty
            terminated = False
        
        truncated = False
        info = {}
        
        return self.state, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """Render the environment."""
        if mode == "human":
            grid = np.zeros((self.size, self.size))
            row, col = divmod(self.state, self.size)
            grid[row, col] = 1  # Agent position
            
            goal_row, goal_col = divmod(self.goal_state, self.size)
            grid[goal_row, goal_col] = 2  # Goal position
            
            print("Grid World:")
            print(grid)
            print(f"Agent at: ({row}, {col}), Goal at: ({goal_row}, {goal_col})")
        
        return None


class CliffWalkingEnv(gym.Env):
    """Cliff Walking environment similar to Sutton & Barto."""
    
    def __init__(self, width: int = 12, height: int = 4):
        """Initialize the cliff walking environment.
        
        Args:
            width: Width of the grid
            height: Height of the grid
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.n_states = width * height
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Start and goal positions
        self.start_state = (height - 1) * width  # Bottom left
        self.goal_state = (height - 1) * width + width - 1  # Bottom right
        
        # Cliff positions (all states in bottom row except start and goal)
        self.cliff_states = []
        for i in range(1, width - 1):
            self.cliff_states.append((height - 1) * width + i)
        
        # Initialize state
        self.state = self.start_state
        
        # Action mappings
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.state = self.start_state
        return self.state, {}
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take a step in the environment."""
        # Get direction from action
        direction = self.action_to_direction[action]
        
        # Calculate new position
        row, col = divmod(self.state, self.width)
        new_row = max(0, min(self.height - 1, row + direction[0]))
        new_col = max(0, min(self.width - 1, col + direction[1]))
        
        # Update state
        self.state = new_row * self.width + new_col
        
        # Calculate reward
        if self.state == self.goal_state:
            reward = 10.0
            terminated = True
        elif self.state in self.cliff_states:
            reward = -100.0
            terminated = True
            self.state = self.start_state  # Reset to start
        else:
            reward = -1.0
            terminated = False
        
        truncated = False
        info = {}
        
        return self.state, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """Render the environment."""
        if mode == "human":
            grid = np.zeros((self.height, self.width))
            
            # Mark cliff
            for cliff_state in self.cliff_states:
                row, col = divmod(cliff_state, self.width)
                grid[row, col] = -1
            
            # Mark agent
            row, col = divmod(self.state, self.width)
            grid[row, col] = 1
            
            # Mark goal
            goal_row, goal_col = divmod(self.goal_state, self.width)
            grid[goal_row, goal_col] = 2
            
            print("Cliff Walking:")
            print(grid)
            print("Legend: 1=Agent, 2=Goal, -1=Cliff")
        
        return None
