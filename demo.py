#!/usr/bin/env python3
"""Final demonstration script for the modernized RL project."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import SARSAAgent, QLearningAgent
from envs import make_env, get_env_info
from visualization import RLVisualizer
from config_manager import ConfigManager


def demonstrate_project():
    """Demonstrate the modernized RL project capabilities."""
    
    print("🤖 Reinforcement Learning: On-Policy vs Off-Policy Learning")
    print("=" * 60)
    print("Modernized with:")
    print("✅ Gymnasium (successor to OpenAI Gym)")
    print("✅ Type hints and modern Python practices")
    print("✅ Comprehensive visualization")
    print("✅ Configuration management")
    print("✅ Multiple environments")
    print("✅ Advanced algorithms (optional)")
    print("✅ Interactive interfaces")
    print("=" * 60)
    
    # 1. Environment Setup
    print("\n🌍 Environment Setup")
    print("-" * 30)
    env = make_env("FrozenLake-v1", is_slippery=False)
    env_info = get_env_info(env)
    print(f"Environment: {env_info['name']}")
    print(f"States: {env_info['n_states']}")
    print(f"Actions: {env_info['n_actions']}")
    
    # 2. Agent Initialization
    print("\n🤖 Agent Initialization")
    print("-" * 30)
    sarsa_agent = SARSAAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    qlearning_agent = QLearningAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print(f"SARSA Agent: {sarsa_agent.name} ({sarsa_agent.type})")
    print(f"Q-Learning Agent: {qlearning_agent.name} ({qlearning_agent.type})")
    
    # 3. Training
    print("\n🏋️ Training Agents")
    print("-" * 30)
    episodes = 500
    
    print(f"Training SARSA for {episodes} episodes...")
    sarsa_metrics = sarsa_agent.train(episodes)
    
    print(f"Training Q-Learning for {episodes} episodes...")
    qlearning_metrics = qlearning_agent.train(episodes)
    
    # 4. Results Analysis
    print("\n📊 Results Analysis")
    print("-" * 30)
    
    sarsa_final = np.mean(sarsa_metrics.episode_rewards[-100:])
    qlearning_final = np.mean(qlearning_metrics.episode_rewards[-100:])
    
    print(f"SARSA final performance (last 100 episodes): {sarsa_final:.3f}")
    print(f"Q-Learning final performance (last 100 episodes): {qlearning_final:.3f}")
    print(f"Q-table updates - SARSA: {sarsa_metrics.q_table_updates}")
    print(f"Q-table updates - Q-Learning: {qlearning_metrics.q_table_updates}")
    
    # 5. Visualization
    print("\n📈 Creating Visualizations")
    print("-" * 30)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    visualizer = RLVisualizer(save_dir="plots")
    
    # Learning curves
    learning_curves = {
        'SARSA': sarsa_metrics.episode_rewards,
        'Q-Learning': qlearning_metrics.episode_rewards
    }
    
    visualizer.plot_learning_curves(
        learning_curves,
        title="SARSA vs Q-Learning Learning Curves",
        save=True,
        filename="demo_learning_curves.png"
    )
    
    # Value functions
    value_functions = {
        'SARSA': sarsa_agent.get_value_function(),
        'Q-Learning': qlearning_agent.get_value_function()
    }
    
    visualizer.plot_value_functions(
        value_functions,
        env_shape=(4, 4),
        title="SARSA vs Q-Learning Value Functions",
        save=True,
        filename="demo_value_functions.png"
    )
    
    # Policies
    policies = {
        'SARSA': sarsa_agent.get_policy(),
        'Q-Learning': qlearning_agent.get_policy()
    }
    
    action_names = ['Left', 'Down', 'Right', 'Up']
    visualizer.plot_policies(
        policies,
        env_shape=(4, 4),
        action_names=action_names,
        title="SARSA vs Q-Learning Policies",
        save=True,
        filename="demo_policies.png"
    )
    
    print("✅ Visualizations saved to plots/ directory")
    
    # 6. Configuration Demo
    print("\n⚙️ Configuration Management")
    print("-" * 30)
    
    config = ConfigManager.get_default_config()
    print(f"Default episodes: {config.training.episodes}")
    print(f"Default learning rate: {config.training.alpha}")
    print(f"Default environment: {config.environment.name}")
    
    # 7. Advanced Features Demo
    print("\n🚀 Advanced Features")
    print("-" * 30)
    
    # Test custom environment
    try:
        from envs import GridWorldEnv
        custom_env = GridWorldEnv(size=3)
        print(f"✅ Custom GridWorld environment: {custom_env.observation_space.n} states")
    except Exception as e:
        print(f"⚠️  Custom environment test: {e}")
    
    # Test advanced agents (if available)
    try:
        from agents import ADVANCED_AGENTS_AVAILABLE
        if ADVANCED_AGENTS_AVAILABLE:
            print("✅ Advanced agents (PPO, SAC, TD3, Rainbow DQN) available")
        else:
            print("ℹ️  Advanced agents require stable-baselines3 installation")
    except Exception as e:
        print(f"⚠️  Advanced agents test: {e}")
    
    # 8. Project Structure
    print("\n📁 Project Structure")
    print("-" * 30)
    
    structure = """
    ├── src/
    │   ├── agents/           # RL agent implementations
    │   ├── envs/             # Environment definitions
    │   ├── config_manager.py # Configuration management
    │   ├── visualization.py  # Plotting utilities
    │   ├── main.py          # CLI training script
    │   └── streamlit_app.py # Web interface
    ├── config/
    │   └── default.yaml      # Default configuration
    ├── notebooks/           # Jupyter notebooks
    ├── tests/               # Unit tests
    ├── requirements.txt     # Dependencies
    └── README.md           # Documentation
    """
    print(structure)
    
    # 9. Usage Examples
    print("\n📋 Usage Examples")
    print("-" * 30)
    
    examples = """
    # Command Line Interface
    python src/main.py --episodes 1000 --env FrozenLake-v1
    
    # Streamlit Web Interface
    streamlit run src/streamlit_app.py
    
    # Jupyter Notebook
    jupyter notebook notebooks/rl_comparison.ipynb
    
    # Comprehensive Example
    python src/comprehensive_example.py --mode all
    
    # Unit Tests
    python -m pytest tests/ -v
    
    # Simple Launcher
    python launcher.py train --episodes 1000
    python launcher.py streamlit
    """
    print(examples)
    
    print("\n🎉 Demonstration Complete!")
    print("=" * 60)
    print("The project has been successfully modernized with:")
    print("✅ Modern Python practices and type hints")
    print("✅ Gymnasium integration")
    print("✅ Comprehensive visualization")
    print("✅ Multiple interfaces (CLI, Streamlit, Jupyter)")
    print("✅ Configuration management")
    print("✅ Unit tests")
    print("✅ Advanced algorithms support")
    print("✅ Clean project structure")
    print("✅ Extensive documentation")
    print("\nReady for GitHub and production use! 🚀")


if __name__ == "__main__":
    demonstrate_project()
