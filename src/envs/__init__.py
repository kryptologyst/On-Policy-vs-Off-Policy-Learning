"""Environment utilities and factory functions."""

import gymnasium as gym
from typing import Dict, Any, Optional
from .custom_envs import GridWorldEnv, CliffWalkingEnv


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create an environment by name.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Created environment
    """
    env_registry = {
        "FrozenLake-v1": lambda: gym.make("FrozenLake-v1", **kwargs),
        "CartPole-v1": lambda: gym.make("CartPole-v1", **kwargs),
        "MountainCar-v0": lambda: gym.make("MountainCar-v0", **kwargs),
        "Acrobot-v1": lambda: gym.make("Acrobot-v1", **kwargs),
        "GridWorld": lambda: GridWorldEnv(**kwargs),
        "CliffWalking": lambda: CliffWalkingEnv(**kwargs),
    }
    
    if env_name not in env_registry:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env_registry[env_name]()


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get information about an environment.
    
    Args:
        env: The environment
        
    Returns:
        Dictionary with environment information
    """
    return {
        "name": env.spec.id if env.spec else "Unknown",
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "n_states": env.observation_space.n if hasattr(env.observation_space, 'n') else None,
        "n_actions": env.action_space.n if hasattr(env.action_space, 'n') else None,
    }
