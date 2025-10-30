"""RL Agents package."""

from .base_agent import BaseAgent, TrainingMetrics
from .sarsa_agent import SARSAAgent
from .qlearning_agent import QLearningAgent

# Try to import advanced agents (optional dependency)
try:
    from .advanced_agents import AdvancedAgent, RainbowDQNAgent, create_advanced_agent, compare_advanced_algorithms
    ADVANCED_AGENTS_AVAILABLE = True
except ImportError:
    ADVANCED_AGENTS_AVAILABLE = False

__all__ = [
    "BaseAgent",
    "TrainingMetrics", 
    "SARSAAgent",
    "QLearningAgent"
]

if ADVANCED_AGENTS_AVAILABLE:
    __all__.extend([
        "AdvancedAgent",
        "RainbowDQNAgent",
        "create_advanced_agent",
        "compare_advanced_algorithms"
    ])
