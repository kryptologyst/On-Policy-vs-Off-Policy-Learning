"""Base agent class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode_rewards: list = None
    episode_lengths: list = None
    epsilon_values: list = None
    q_table_updates: int = 0
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.epsilon_values is None:
            self.epsilon_values = []


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        **kwargs
    ):
        """Initialize the base agent.
        
        Args:
            env: The gymnasium environment
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Training metrics
        self.metrics = TrainingMetrics()
        
    def epsilon_greedy_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    @abstractmethod
    def update_q_value(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool,
        **kwargs
    ) -> None:
        """Update Q-value based on the specific algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def train_episode(self) -> Tuple[float, int]:
        """Train for one episode.
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        pass
    
    def train(self, episodes: int) -> TrainingMetrics:
        """Train the agent for multiple episodes.
        
        Args:
            episodes: Number of episodes to train
            
        Returns:
            Training metrics
        """
        for episode in range(episodes):
            reward, length = self.train_episode()
            
            # Update metrics
            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_lengths.append(length)
            self.metrics.epsilon_values.append(self.epsilon)
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.metrics.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return self.metrics
    
    def get_value_function(self) -> np.ndarray:
        """Get the value function (max Q-value for each state).
        
        Returns:
            Value function as numpy array
        """
        return np.max(self.q_table, axis=1)
    
    def get_policy(self) -> np.ndarray:
        """Get the greedy policy (best action for each state).
        
        Returns:
            Policy as numpy array
        """
        return np.argmax(self.q_table, axis=1)
    
    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file.
        
        Args:
            filepath: Path to save the Q-table
        """
        np.save(filepath, self.q_table)
    
    def load_q_table(self, filepath: str) -> None:
        """Load Q-table from file.
        
        Args:
            filepath: Path to load the Q-table from
        """
        self.q_table = np.load(filepath)
