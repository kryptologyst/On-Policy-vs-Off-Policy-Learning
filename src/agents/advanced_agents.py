"""Advanced RL agents using stable-baselines3."""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os


class AdvancedAgent:
    """Wrapper for advanced RL algorithms from stable-baselines3."""
    
    def __init__(
        self,
        env: gym.Env,
        algorithm: str = "PPO",
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize advanced agent.
        
        Args:
            env: The gymnasium environment
            algorithm: Algorithm name (PPO, SAC, TD3, DQN)
            learning_rate: Learning rate
            **kwargs: Additional algorithm-specific parameters
        """
        self.env = env
        self.algorithm_name = algorithm
        self.learning_rate = learning_rate
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Initialize algorithm
        self._create_algorithm(**kwargs)
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': []
        }
    
    def _create_algorithm(self, **kwargs):
        """Create the specified algorithm."""
        algorithm_params = {
            'learning_rate': self.learning_rate,
            'verbose': 0,
            **kwargs
        }
        
        if self.algorithm_name == "PPO":
            self.algorithm = PPO(
                "MlpPolicy",
                self.vec_env,
                **algorithm_params
            )
        elif self.algorithm_name == "SAC":
            self.algorithm = SAC(
                "MlpPolicy",
                self.vec_env,
                **algorithm_params
            )
        elif self.algorithm_name == "TD3":
            self.algorithm = TD3(
                "MlpPolicy",
                self.vec_env,
                **algorithm_params
            )
        elif self.algorithm_name == "DQN":
            self.algorithm = DQN(
                "MlpPolicy",
                self.vec_env,
                **algorithm_params
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def train(self, total_timesteps: int = 10000, eval_freq: int = 1000) -> Dict[str, Any]:
        """Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            
        Returns:
            Training metrics
        """
        # Create evaluation environment
        eval_env = Monitor(self.env)
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./checkpoints/{self.algorithm_name.lower()}_best',
            log_path=f'./logs/{self.algorithm_name.lower()}_eval',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.algorithm.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Collect training metrics
        self.training_metrics['eval_rewards'] = eval_callback.evaluations_results
        
        return self.training_metrics
    
    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate the trained agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def save(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.algorithm.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        self.algorithm = self.algorithm.load(filepath)


class RainbowDQNAgent:
    """Rainbow DQN implementation using stable-baselines3 DQN with extensions."""
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 1000,
        train_freq: int = 4,
        gradient_steps: int = 1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        **kwargs
    ):
        """Initialize Rainbow DQN agent.
        
        Args:
            env: The gymnasium environment
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer
            learning_starts: Number of steps before learning starts
            batch_size: Batch size for training
            target_update_interval: Frequency of target network updates
            train_freq: Training frequency
            gradient_steps: Number of gradient steps per update
            exploration_fraction: Fraction of total timesteps for exploration
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            **kwargs: Additional parameters
        """
        self.env = env
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Rainbow DQN parameters
        rainbow_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'target_update_interval': target_update_interval,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps,
            'verbose': 0,
            **kwargs
        }
        
        self.algorithm = DQN("MlpPolicy", self.vec_env, **rainbow_params)
        self.algorithm_name = "Rainbow DQN"
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': []
        }
    
    def train(self, total_timesteps: int = 10000, eval_freq: int = 1000) -> Dict[str, Any]:
        """Train the Rainbow DQN agent."""
        # Create evaluation environment
        eval_env = Monitor(self.env)
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./checkpoints/rainbow_dqn_best',
            log_path=f'./logs/rainbow_dqn_eval',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.algorithm.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Collect training metrics
        self.training_metrics['eval_rewards'] = eval_callback.evaluations_results
        
        return self.training_metrics
    
    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate the trained agent."""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.algorithm.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load a trained model."""
        self.algorithm = self.algorithm.load(filepath)


def create_advanced_agent(env: gym.Env, algorithm: str, **kwargs) -> AdvancedAgent:
    """Factory function to create advanced agents.
    
    Args:
        env: The gymnasium environment
        algorithm: Algorithm name
        **kwargs: Additional parameters
        
    Returns:
        Advanced agent instance
    """
    return AdvancedAgent(env, algorithm, **kwargs)


def compare_advanced_algorithms(
    env: gym.Env,
    algorithms: list = ["PPO", "SAC", "DQN"],
    total_timesteps: int = 10000,
    eval_episodes: int = 10
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple advanced algorithms.
    
    Args:
        env: The gymnasium environment
        algorithms: List of algorithm names to compare
        total_timesteps: Training timesteps
        eval_episodes: Episodes for evaluation
        
    Returns:
        Dictionary with results for each algorithm
    """
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🚀 Training {algorithm}...")
        
        try:
            # Create agent
            agent = create_advanced_agent(env, algorithm)
            
            # Train agent
            training_metrics = agent.train(total_timesteps=total_timesteps)
            
            # Evaluate agent
            mean_reward, std_reward = agent.evaluate(n_episodes=eval_episodes)
            
            results[algorithm] = {
                'agent': agent,
                'training_metrics': training_metrics,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            }
            
            print(f"✅ {algorithm} training completed!")
            print(f"   Final performance: {mean_reward:.2f} ± {std_reward:.2f}")
            
        except Exception as e:
            print(f"❌ Error training {algorithm}: {e}")
            results[algorithm] = {'error': str(e)}
    
    return results
