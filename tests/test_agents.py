"""Unit tests for RL agents and utilities."""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import SARSAAgent, QLearningAgent, BaseAgent
from envs import make_env, GridWorldEnv, CliffWalkingEnv
from config_manager import ConfigManager, Config


class TestEnvironments:
    """Test environment functionality."""
    
    def test_frozenlake_env(self):
        """Test FrozenLake environment creation."""
        env = make_env("FrozenLake-v1")
        assert env.observation_space.n == 16
        assert env.action_space.n == 4
    
    def test_gridworld_env(self):
        """Test custom GridWorld environment."""
        env = GridWorldEnv(size=5)
        assert env.observation_space.n == 25
        assert env.action_space.n == 4
        
        # Test reset
        state, info = env.reset()
        assert state == 0
        assert isinstance(info, dict)
        
        # Test step
        next_state, reward, terminated, truncated, info = env.step(0)
        assert isinstance(next_state, int)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_cliffwalking_env(self):
        """Test CliffWalking environment."""
        env = CliffWalkingEnv(width=4, height=3)
        assert env.observation_space.n == 12
        assert env.action_space.n == 4
        
        # Test cliff penalty
        state, _ = env.reset()
        # Move to cliff position
        next_state, reward, terminated, truncated, _ = env.step(3)  # Right
        if next_state in env.cliff_states:
            assert reward == -100.0
            assert terminated == True


class TestAgents:
    """Test agent functionality."""
    
    def setup_method(self):
        """Set up test environment and agents."""
        self.env = make_env("FrozenLake-v1")
        self.sarsa_agent = SARSAAgent(self.env)
        self.qlearning_agent = QLearningAgent(self.env)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.sarsa_agent.name == "SARSA"
        assert self.sarsa_agent.type == "on_policy"
        assert self.qlearning_agent.name == "Q-Learning"
        assert self.qlearning_agent.type == "off_policy"
        
        # Check Q-table shape
        assert self.sarsa_agent.q_table.shape == (16, 4)
        assert self.qlearning_agent.q_table.shape == (16, 4)
    
    def test_epsilon_greedy_action(self):
        """Test epsilon-greedy action selection."""
        # Test with high epsilon (should be mostly random)
        self.sarsa_agent.epsilon = 1.0
        actions = [self.sarsa_agent.epsilon_greedy_action(0) for _ in range(100)]
        assert all(0 <= action < 4 for action in actions)
        
        # Test with low epsilon (should be mostly greedy)
        self.sarsa_agent.epsilon = 0.0
        self.sarsa_agent.q_table[0, 2] = 1.0  # Set action 2 as best
        action = self.sarsa_agent.epsilon_greedy_action(0)
        assert action == 2
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.sarsa_agent.epsilon
        self.sarsa_agent.decay_epsilon()
        assert self.sarsa_agent.epsilon < initial_epsilon
        
        # Test minimum epsilon
        self.sarsa_agent.epsilon = self.sarsa_agent.epsilon_min
        self.sarsa_agent.decay_epsilon()
        assert self.sarsa_agent.epsilon == self.sarsa_agent.epsilon_min
    
    def test_sarsa_update(self):
        """Test SARSA Q-value update."""
        initial_q = self.sarsa_agent.q_table[0, 0]
        self.sarsa_agent.update_q_value(0, 0, 1.0, 1, False, next_action=1)
        assert self.sarsa_agent.q_table[0, 0] != initial_q
    
    def test_qlearning_update(self):
        """Test Q-Learning Q-value update."""
        initial_q = self.qlearning_agent.q_table[0, 0]
        self.qlearning_agent.update_q_value(0, 0, 1.0, 1, False)
        assert self.qlearning_agent.q_table[0, 0] != initial_q
    
    def test_train_episode(self):
        """Test training for one episode."""
        reward, length = self.sarsa_agent.train_episode()
        assert isinstance(reward, (int, float))
        assert isinstance(length, int)
        assert length > 0
    
    def test_get_value_function(self):
        """Test value function extraction."""
        value_func = self.sarsa_agent.get_value_function()
        assert len(value_func) == 16
        assert all(isinstance(v, (int, float)) for v in value_func)
    
    def test_get_policy(self):
        """Test policy extraction."""
        policy = self.sarsa_agent.get_policy()
        assert len(policy) == 16
        assert all(0 <= action < 4 for action in policy)


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ConfigManager.get_default_config()
        assert config.training.episodes == 1000
        assert config.training.alpha == 0.1
        assert config.environment.name == "FrozenLake-v1"
    
    def test_config_merge(self):
        """Test configuration merging."""
        base_config = ConfigManager.get_default_config()
        overrides = {
            'training': {'episodes': 500},
            'environment': {'name': 'CartPole-v1'}
        }
        
        merged_config = ConfigManager.merge_configs(base_config, overrides)
        assert merged_config.training.episodes == 500
        assert merged_config.environment.name == 'CartPole-v1'
        assert merged_config.training.alpha == 0.1  # Should remain unchanged


class TestTrainingMetrics:
    """Test training metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        from agents import TrainingMetrics
        
        metrics = TrainingMetrics()
        assert metrics.episode_rewards == []
        assert metrics.episode_lengths == []
        assert metrics.epsilon_values == []
        assert metrics.q_table_updates == 0
    
    def test_metrics_update(self):
        """Test metrics update during training."""
        env = make_env("FrozenLake-v1")
        agent = SARSAAgent(env)
        
        # Train for a few episodes
        agent.train(5)
        
        assert len(agent.metrics.episode_rewards) == 5
        assert len(agent.metrics.episode_lengths) == 5
        assert len(agent.metrics.epsilon_values) == 5
        assert agent.metrics.q_table_updates > 0


if __name__ == "__main__":
    pytest.main([__file__])
