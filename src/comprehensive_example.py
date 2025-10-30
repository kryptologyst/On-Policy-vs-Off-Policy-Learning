"""Comprehensive example demonstrating all RL features."""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import (
    SARSAAgent, QLearningAgent, 
    AdvancedAgent, RainbowDQNAgent,
    compare_advanced_algorithms
)
from envs import make_env, get_env_info
from visualization import RLVisualizer
from config_manager import ConfigManager


def run_tabular_comparison(env_name: str = "FrozenLake-v1", episodes: int = 1000):
    """Run comparison between SARSA and Q-Learning."""
    print(f"\n🔬 Tabular Methods Comparison: {env_name}")
    print("=" * 50)
    
    # Create environment
    env = make_env(env_name, is_slippery=False)
    env_info = get_env_info(env)
    print(f"Environment: {env_info['name']}")
    print(f"States: {env_info['n_states']}, Actions: {env_info['n_actions']}")
    
    # Initialize agents
    agents = {
        'SARSA': SARSAAgent(env, learning_rate=0.1, epsilon=0.1),
        'Q-Learning': QLearningAgent(env, learning_rate=0.1, epsilon=0.1)
    }
    
    # Train agents
    results = {}
    for agent_name, agent in agents.items():
        print(f"\n🚀 Training {agent_name}...")
        metrics = agent.train(episodes)
        
        results[agent_name] = {
            'agent': agent,
            'metrics': metrics,
            'rewards': metrics.episode_rewards,
            'value_function': agent.get_value_function(),
            'policy': agent.get_policy()
        }
        
        final_reward = np.mean(metrics.episode_rewards[-100:])
        print(f"✅ {agent_name} final performance: {final_reward:.2f}")
    
    # Create visualizations
    visualizer = RLVisualizer(save_dir="plots")
    
    # Learning curves
    learning_curves = {name: data['rewards'] for name, data in results.items()}
    visualizer.plot_learning_curves(
        learning_curves,
        title=f"SARSA vs Q-Learning on {env_name}",
        save=True
    )
    
    # Value functions
    value_functions = {name: data['value_function'] for name, data in results.items()}
    visualizer.plot_value_functions(
        value_functions,
        env_shape=(4, 4) if env_info['n_states'] == 16 else None,
        title=f"Value Functions: {env_name}",
        save=True
    )
    
    return results


def run_advanced_comparison(env_name: str = "CartPole-v1", timesteps: int = 10000):
    """Run comparison between advanced algorithms."""
    print(f"\n🚀 Advanced Algorithms Comparison: {env_name}")
    print("=" * 50)
    
    # Create environment
    env = make_env(env_name)
    env_info = get_env_info(env)
    print(f"Environment: {env_info['name']}")
    print(f"Observation Space: {env_info['observation_space']}")
    print(f"Action Space: {env_info['action_space']}")
    
    # Define algorithms to compare
    algorithms = ["PPO", "SAC", "DQN"]
    
    # Run comparison
    results = compare_advanced_algorithms(
        env=env,
        algorithms=algorithms,
        total_timesteps=timesteps,
        eval_episodes=10
    )
    
    # Print results
    print(f"\n📊 Performance Summary:")
    print("-" * 30)
    for algorithm, data in results.items():
        if 'error' not in data:
            print(f"{algorithm:>10}: {data['mean_reward']:>8.2f} ± {data['std_reward']:>6.2f}")
        else:
            print(f"{algorithm:>10}: Error - {data['error']}")
    
    return results


def run_environment_comparison():
    """Compare algorithms across different environments."""
    print(f"\n🌍 Environment Comparison")
    print("=" * 50)
    
    environments = [
        ("FrozenLake-v1", "tabular"),
        ("CartPole-v1", "continuous"),
        ("MountainCar-v0", "continuous")
    ]
    
    all_results = {}
    
    for env_name, env_type in environments:
        print(f"\n🔬 Testing {env_name} ({env_type})...")
        
        try:
            if env_type == "tabular":
                # Use tabular methods
                env = make_env(env_name, is_slippery=False)
                if hasattr(env.observation_space, 'n') and hasattr(env.action_space, 'n'):
                    results = run_tabular_comparison(env_name, episodes=500)
                    all_results[env_name] = results
                else:
                    print(f"⚠️  {env_name} not suitable for tabular methods")
                    
            elif env_type == "continuous":
                # Use advanced methods
                results = run_advanced_comparison(env_name, timesteps=5000)
                all_results[env_name] = results
                
        except Exception as e:
            print(f"❌ Error with {env_name}: {e}")
    
    return all_results


def create_comprehensive_report(results: Dict[str, Any]):
    """Create a comprehensive report of all results."""
    print(f"\n📋 Comprehensive Report")
    print("=" * 50)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Tabular methods performance
    ax1 = axes[0, 0]
    tabular_envs = [env for env, data in results.items() if 'SARSA' in data]
    if tabular_envs:
        env_names = tabular_envs
        sarsa_performance = []
        qlearning_performance = []
        
        for env_name in env_names:
            data = results[env_name]
            sarsa_performance.append(np.mean(data['SARSA']['rewards'][-100:]))
            qlearning_performance.append(np.mean(data['Q-Learning']['rewards'][-100:]))
        
        x = np.arange(len(env_names))
        width = 0.35
        
        ax1.bar(x - width/2, sarsa_performance, width, label='SARSA', alpha=0.8)
        ax1.bar(x + width/2, qlearning_performance, width, label='Q-Learning', alpha=0.8)
        
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Final Performance')
        ax1.set_title('Tabular Methods Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(env_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Advanced methods performance
    ax2 = axes[0, 1]
    advanced_envs = [env for env, data in results.items() if 'PPO' in data]
    if advanced_envs:
        env_names = advanced_envs
        algorithms = ['PPO', 'SAC', 'DQN']
        
        for i, algorithm in enumerate(algorithms):
            performance = []
            for env_name in env_names:
                data = results[env_name]
                if algorithm in data and 'mean_reward' in data[algorithm]:
                    performance.append(data[algorithm]['mean_reward'])
                else:
                    performance.append(0)
            
            ax2.plot(env_names, performance, 'o-', label=algorithm, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Environment')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Advanced Methods Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning curves comparison
    ax3 = axes[1, 0]
    if tabular_envs:
        env_name = tabular_envs[0]  # Use first tabular environment
        data = results[env_name]
        
        for agent_name, agent_data in data.items():
            rewards = agent_data['rewards']
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes = np.arange(window_size-1, len(rewards))
                ax3.plot(episodes, moving_avg, label=f"{agent_name} (MA-{window_size})", linewidth=2)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title(f'Learning Curves: {env_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Algorithm comparison summary
    ax4 = axes[1, 1]
    all_algorithms = ['SARSA', 'Q-Learning', 'PPO', 'SAC', 'DQN']
    avg_performance = []
    
    for algorithm in all_algorithms:
        performances = []
        for env_name, data in results.items():
            if algorithm in data:
                if 'mean_reward' in data[algorithm]:
                    performances.append(data[algorithm]['mean_reward'])
                elif 'rewards' in data[algorithm]:
                    performances.append(np.mean(data[algorithm]['rewards'][-100:]))
        
        if performances:
            avg_performance.append(np.mean(performances))
        else:
            avg_performance.append(0)
    
    bars = ax4.bar(all_algorithms, avg_performance, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
    ax4.set_ylabel('Average Performance')
    ax4.set_title('Overall Algorithm Performance')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_performance):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/comprehensive_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n📊 Summary Statistics:")
    print("-" * 30)
    for algorithm, performance in zip(all_algorithms, avg_performance):
        print(f"{algorithm:>12}: {performance:>8.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive RL comparison")
    parser.add_argument("--mode", type=str, choices=["tabular", "advanced", "all"], 
                      default="all", help="Comparison mode")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", 
                      help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, 
                      help="Number of episodes for tabular methods")
    parser.add_argument("--timesteps", type=int, default=10000, 
                      help="Number of timesteps for advanced methods")
    
    args = parser.parse_args()
    
    print("🤖 Reinforcement Learning Comprehensive Comparison")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    results = {}
    
    if args.mode in ["tabular", "all"]:
        try:
            tabular_results = run_tabular_comparison(args.env, args.episodes)
            results[args.env] = tabular_results
        except Exception as e:
            print(f"❌ Error in tabular comparison: {e}")
    
    if args.mode in ["advanced", "all"]:
        try:
            advanced_results = run_advanced_comparison(args.env, args.timesteps)
            results[f"{args.env}_advanced"] = advanced_results
        except Exception as e:
            print(f"❌ Error in advanced comparison: {e}")
    
    if args.mode == "all":
        try:
            env_results = run_environment_comparison()
            results.update(env_results)
        except Exception as e:
            print(f"❌ Error in environment comparison: {e}")
    
    # Create comprehensive report
    if results:
        create_comprehensive_report(results)
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Results saved to: plots/, logs/, checkpoints/")


if __name__ == "__main__":
    main()
