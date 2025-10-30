"""SARSA (State-Action-Reward-State-Action) agent implementation."""

from typing import Tuple
import numpy as np
import gymnasium as gym
from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """SARSA agent implementing on-policy temporal difference learning."""
    
    def __init__(self, env: gym.Env, **kwargs):
        """Initialize SARSA agent.
        
        Args:
            env: The gymnasium environment
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(env, **kwargs)
        self.name = "SARSA"
        self.type = "on_policy"
    
    def update_q_value(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool,
        next_action: int = None,
        **kwargs
    ) -> None:
        """Update Q-value using SARSA update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_action: Next action to be taken (required for SARSA)
        """
        if next_action is None:
            raise ValueError("next_action is required for SARSA update")
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table[next_state, next_action]
        
        self.q_table[state, action] += self.learning_rate * (target - current_q)
        self.metrics.q_table_updates += 1
    
    def train_episode(self) -> Tuple[float, int]:
        """Train for one episode using SARSA.
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        state, _ = self.env.reset()
        action = self.epsilon_greedy_action(state)
        
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            # Take action and observe result
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Select next action using current policy
            next_action = self.epsilon_greedy_action(next_state)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, done, next_action)
            
            # Update state and action
            state = next_state
            action = next_action
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
