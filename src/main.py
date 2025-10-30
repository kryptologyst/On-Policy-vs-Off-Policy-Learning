"""Main training script for RL agents comparison."""

import argparse
import os
import sys
from typing import Dict, Any, List
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import SARSAAgent, QLearningAgent
from envs import make_env, get_env_info
from visualization import RLVisualizer
from config_manager import ConfigManager, Config


def train_agents(config: Config) -> Dict[str, Dict[str, Any]]:
    """Train multiple agents and return results.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with training results for each agent
    """
    # Create environment
    env = make_env(config.environment.name, **config.environment.kwargs)
    env_info = get_env_info(env)
    print(f"Environment: {env_info['name']}")
    print(f"States: {env_info['n_states']}, Actions: {env_info['n_actions']}")
    
    # Initialize agents
    agents = {
        'SARSA': SARSAAgent(
            env,
            learning_rate=config.training.alpha,
            discount_factor=config.training.gamma,
            epsilon=config.training.epsilon,
            epsilon_decay=config.training.epsilon_decay,
            epsilon_min=config.training.epsilon_min
        ),
        'Q-Learning': QLearningAgent(
            env,
            learning_rate=config.training.alpha,
            discount_factor=config.training.gamma,
            epsilon=config.training.epsilon,
            epsilon_decay=config.training.epsilon_decay,
            epsilon_min=config.training.epsilon_min
        )
    }
    
    # Train agents
    results = {}
    for agent_name, agent in agents.items():
        print(f"\nTraining {agent_name}...")
        metrics = agent.train(config.training.episodes)
        
        results[agent_name] = {
            'agent': agent,
            'metrics': metrics,
            'rewards': metrics.episode_rewards,
            'lengths': metrics.episode_lengths,
            'epsilons': metrics.epsilon_values,
            'value_function': agent.get_value_function(),
            'policy': agent.get_policy()
        }
        
        # Print final performance
        final_reward = np.mean(metrics.episode_rewards[-100:])
        print(f"{agent_name} final performance: {final_reward:.2f}")
    
    return results


def create_visualizations(results: Dict[str, Dict[str, Any]], config: Config) -> None:
    """Create visualizations from training results.
    
    Args:
        results: Training results
        config: Configuration object
    """
    if not config.visualization.plot_learning_curves and not config.visualization.plot_value_functions:
        return
    
    visualizer = RLVisualizer(
        save_dir="plots",
        dpi=config.visualization.dpi
    )
    
    # Learning curves
    if config.visualization.plot_learning_curves:
        learning_curves = {name: data['rewards'] for name, data in results.items()}
        visualizer.plot_learning_curves(
            learning_curves,
            title="SARSA vs Q-Learning Learning Curves",
            save=config.visualization.save_plots
        )
    
    # Value functions
    if config.visualization.plot_value_functions:
        value_functions = {name: data['value_function'] for name, data in results.items()}
        visualizer.plot_value_functions(
            value_functions,
            env_shape=(4, 4),  # FrozenLake shape
            title="SARSA vs Q-Learning Value Functions",
            save=config.visualization.save_plots
        )
    
    # Policies
    if config.visualization.plot_policies:
        policies = {name: data['policy'] for name, data in results.items()}
        action_names = ['Left', 'Down', 'Right', 'Up']  # FrozenLake actions
        visualizer.plot_policies(
            policies,
            env_shape=(4, 4),
            action_names=action_names,
            title="SARSA vs Q-Learning Policies",
            save=config.visualization.save_plots
        )
    
    # Epsilon decay
    epsilon_data = {name: data['epsilons'] for name, data in results.items()}
    visualizer.plot_epsilon_decay(
        epsilon_data,
        title="Epsilon Decay",
        save=config.visualization.save_plots
    )
    
    # Summary comparison
    summary_data = {
        name: {
            'rewards': data['rewards'],
            'lengths': data['lengths'],
            'epsilons': data['epsilons']
        }
        for name, data in results.items()
    }
    visualizer.plot_comparison_summary(
        summary_data,
        save=config.visualization.save_plots
    )


def save_results(results: Dict[str, Dict[str, Any]], config: Config) -> None:
    """Save training results and models.
    
    Args:
        results: Training results
        config: Configuration object
    """
    if not config.model_saving.save_checkpoints:
        return
    
    os.makedirs(config.model_saving.checkpoint_dir, exist_ok=True)
    
    for agent_name, data in results.items():
        agent = data['agent']
        
        # Save Q-table
        q_table_path = os.path.join(config.model_saving.checkpoint_dir, f"{agent_name.lower()}_q_table.npy")
        agent.save_q_table(q_table_path)
        
        # Save metrics
        metrics_path = os.path.join(config.model_saving.checkpoint_dir, f"{agent_name.lower()}_metrics.npy")
        np.save(metrics_path, {
            'rewards': data['rewards'],
            'lengths': data['lengths'],
            'epsilons': data['epsilons']
        })
        
        print(f"Saved {agent_name} model to {config.model_saving.checkpoint_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and compare RL agents")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                      help="Path to configuration file")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--alpha", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Initial epsilon")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigManager.load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = ConfigManager.get_default_config()
    
    # Override with command line arguments
    overrides = {}
    if args.episodes:
        overrides['training'] = {'episodes': args.episodes}
    if args.env:
        overrides['environment'] = {'name': args.env}
    if args.alpha:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['alpha'] = args.alpha
    if args.gamma:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['gamma'] = args.gamma
    if args.epsilon:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['epsilon'] = args.epsilon
    
    if overrides:
        config = ConfigManager.merge_configs(config, overrides)
    
    print("Configuration:")
    print(f"  Environment: {config.environment.name}")
    print(f"  Episodes: {config.training.episodes}")
    print(f"  Learning Rate: {config.training.alpha}")
    print(f"  Discount Factor: {config.training.gamma}")
    print(f"  Initial Epsilon: {config.training.epsilon}")
    
    # Train agents
    results = train_agents(config)
    
    # Create visualizations
    create_visualizations(results, config)
    
    # Save results
    save_results(results, config)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
