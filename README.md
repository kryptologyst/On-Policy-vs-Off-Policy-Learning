# On-Policy vs Off-Policy Learning

A comprehensive implementation comparing SARSA (on-policy) and Q-Learning (off-policy) algorithms with state-of-the-art features and visualization capabilities.

## 🚀 Features

- **Modern RL Framework**: Built with Gymnasium (successor to OpenAI Gym)
- **Algorithm Comparison**: Side-by-side comparison of SARSA and Q-Learning
- **Multiple Environments**: Support for FrozenLake, CartPole, MountainCar, and custom environments
- **Rich Visualizations**: Learning curves, value functions, policies, and epsilon decay plots
- **Interactive UI**: Streamlit web interface for easy experimentation
- **Configuration Management**: YAML-based configuration system
- **Type Safety**: Full type hints and modern Python practices
- **Extensible Design**: Easy to add new algorithms and environments

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/On-Policy-vs-Off-Policy-Learning.git
cd On-Policy-vs-Off-Policy-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Command Line Interface

Train agents with default settings:
```bash
python src/main.py
```

Customize training parameters:
```bash
python src/main.py --episodes 2000 --alpha 0.05 --gamma 0.95 --env FrozenLake-v1
```

### Streamlit Web Interface

Launch the interactive web interface:
```bash
streamlit run src/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Jupyter Notebook

For interactive development and analysis:
```bash
jupyter notebook notebooks/
```

## 🏗️ Project Structure

```
├── src/
│   ├── agents/           # RL agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── sarsa_agent.py
│   │   └── qlearning_agent.py
│   ├── envs/             # Environment definitions
│   │   ├── __init__.py
│   │   └── custom_envs.py
│   ├── config_manager.py # Configuration management
│   ├── visualization.py  # Plotting utilities
│   ├── main.py          # CLI training script
│   └── streamlit_app.py # Web interface
├── config/
│   └── default.yaml      # Default configuration
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
├── logs/                # Training logs
├── checkpoints/         # Saved models
├── plots/               # Generated plots
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🧠 Algorithms

### SARSA (State-Action-Reward-State-Action)
- **Type**: On-policy temporal difference learning
- **Update Rule**: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **Characteristics**: Learns from actions actually taken, more conservative

### Q-Learning
- **Type**: Off-policy temporal difference learning  
- **Update Rule**: Q(s,a) = Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
- **Characteristics**: Learns from optimal actions, more optimistic

## 🌍 Supported Environments

- **FrozenLake-v1**: Classic grid world navigation task
- **CartPole-v1**: Balance a pole on a cart
- **MountainCar-v0**: Drive a car up a mountain
- **Acrobot-v1**: Swing up a two-link pendulum
- **GridWorld**: Custom 5x5 grid world
- **CliffWalking**: Cliff walking environment from Sutton & Barto

## 📊 Visualizations

The framework provides comprehensive visualizations:

- **Learning Curves**: Episode rewards over time with moving averages
- **Value Functions**: Heatmaps showing state values
- **Policies**: Action maps showing optimal actions per state
- **Epsilon Decay**: Exploration rate decay over episodes
- **Performance Comparison**: Side-by-side algorithm comparison

## ⚙️ Configuration

Configuration is managed through YAML files. Key parameters:

```yaml
training:
  episodes: 1000
  alpha: 0.1      # Learning rate
  gamma: 0.99     # Discount factor
  epsilon: 0.1    # Initial exploration rate
  epsilon_decay: 0.995
  epsilon_min: 0.01

environment:
  name: "FrozenLake-v1"
  kwargs:
    is_slippery: false

visualization:
  plot_learning_curves: true
  plot_value_functions: true
  plot_policies: true
  save_plots: true
```

## 🔬 Usage Examples

### Basic Training
```python
from src.agents import SARSAAgent, QLearningAgent
from src.envs import make_env

# Create environment
env = make_env("FrozenLake-v1")

# Initialize agents
sarsa_agent = SARSAAgent(env)
qlearning_agent = QLearningAgent(env)

# Train agents
sarsa_metrics = sarsa_agent.train(1000)
qlearning_metrics = qlearning_agent.train(1000)

# Compare results
print(f"SARSA final reward: {np.mean(sarsa_metrics.episode_rewards[-100:]):.2f}")
print(f"Q-Learning final reward: {np.mean(qlearning_metrics.episode_rewards[-100:]):.2f}")
```

### Custom Environment
```python
from src.envs import GridWorldEnv

# Create custom environment
env = GridWorldEnv(size=5, goal_reward=10.0)

# Train agent
agent = SARSAAgent(env)
metrics = agent.train(500)
```

### Visualization
```python
from src.visualization import RLVisualizer

# Create visualizer
viz = RLVisualizer()

# Plot learning curves
viz.plot_learning_curves({
    'SARSA': sarsa_metrics.episode_rewards,
    'Q-Learning': qlearning_metrics.episode_rewards
})
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src
```

## 📈 Performance Tips

1. **Hyperparameter Tuning**: Experiment with learning rates (0.01-0.3) and discount factors (0.9-0.99)
2. **Exploration Strategy**: Adjust epsilon decay for different exploration schedules
3. **Environment Selection**: Start with deterministic environments (is_slippery=False) for easier learning
4. **Episode Count**: Use at least 1000 episodes for stable learning curves

## 🔧 Advanced Features

### Adding New Algorithms
1. Inherit from `BaseAgent`
2. Implement `update_q_value()` and `train_episode()` methods
3. Add to agent registry in `src/agents/__init__.py`

### Custom Environments
1. Inherit from `gymnasium.Env`
2. Implement required methods: `reset()`, `step()`, `render()`
3. Add to environment factory in `src/envs/__init__.py`

### Logging and Monitoring
- TensorBoard integration for training metrics
- Weights & Biases support for experiment tracking
- Automatic checkpoint saving

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press.
- Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium team for the RL environment framework
- The reinforcement learning community for algorithms and insights
- Contributors and users who provide feedback and improvements
# On-Policy-vs-Off-Policy-Learning
