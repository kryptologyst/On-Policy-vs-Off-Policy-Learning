"""Q-Learning agent implementation."""

from typing import Tuple
import numpy as np
import gymnasium as gym
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent implementing off-policy temporal difference learning."""
    
    def __init__(self, env: gym.Env, **kwargs):
        """Initialize Q-Learning agent.
        
        Args:
            env: The gymnasium environment
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(env, **kwargs)
        self.name = "Q-Learning"
        self.type = "off_policy"
    
    def update_q_value(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool,
        **kwargs
    ) -> None:
        """Update Q-value using Q-Learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - current_q)
        self.metrics.q_table_updates += 1
    
    def train_episode(self) -> Tuple[float, int]:
        """Train for one episode using Q-Learning.
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        state, _ = self.env.reset()
        
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            # Select action using epsilon-greedy policy
            action = self.epsilon_greedy_action(state)
            
            # Take action and observe result
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
