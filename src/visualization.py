"""Visualization utilities for RL training and results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os


class RLVisualizer:
    """Class for creating RL visualizations."""
    
    def __init__(self, save_dir: str = "plots", dpi: int = 300):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots
            dpi: DPI for saved plots
        """
        self.save_dir = save_dir
        self.dpi = dpi
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_learning_curves(
        self, 
        metrics: Dict[str, List[float]], 
        title: str = "Learning Curves",
        save: bool = True,
        filename: str = "learning_curves.png"
    ) -> None:
        """Plot learning curves for multiple agents.
        
        Args:
            metrics: Dictionary with agent names as keys and reward lists as values
            title: Title for the plot
            save: Whether to save the plot
            filename: Filename for saving
        """
        plt.figure(figsize=(12, 6))
        
        for agent_name, rewards in metrics.items():
            # Calculate moving average
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes = np.arange(window_size-1, len(rewards))
                plt.plot(episodes, moving_avg, label=f"{agent_name} (MA-{window_size})", linewidth=2)
            else:
                plt.plot(rewards, label=agent_name, alpha=0.7)
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_value_functions(
        self, 
        value_functions: Dict[str, np.ndarray],
        env_shape: Tuple[int, int] = (4, 4),
        title: str = "Value Functions Comparison",
        save: bool = True,
        filename: str = "value_functions.png"
    ) -> None:
        """Plot value functions as heatmaps.
        
        Args:
            value_functions: Dictionary with agent names as keys and value functions as values
            env_shape: Shape of the environment grid
            title: Title for the plot
            save: Whether to save the plot
            filename: Filename for saving
        """
        n_agents = len(value_functions)
        fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 4))
        
        if n_agents == 1:
            axes = [axes]
        
        for i, (agent_name, values) in enumerate(value_functions.items()):
            # Reshape values to grid
            if len(values) == env_shape[0] * env_shape[1]:
                value_grid = values.reshape(env_shape)
            else:
                value_grid = values
            
            # Create heatmap
            im = axes[i].imshow(value_grid, cmap='viridis', aspect='auto')
            axes[i].set_title(f"{agent_name}")
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
            
            # Add text annotations
            for y in range(value_grid.shape[0]):
                for x in range(value_grid.shape[1]):
                    text = axes[i].text(x, y, f'{value_grid[y, x]:.2f}',
                                      ha="center", va="center", color="white", fontsize=8)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_policies(
        self, 
        policies: Dict[str, np.ndarray],
        env_shape: Tuple[int, int] = (4, 4),
        action_names: List[str] = None,
        title: str = "Policy Comparison",
        save: bool = True,
        filename: str = "policies.png"
    ) -> None:
        """Plot policies as action maps.
        
        Args:
            policies: Dictionary with agent names as keys and policy arrays as values
            env_shape: Shape of the environment grid
            action_names: Names for actions (e.g., ['Up', 'Down', 'Left', 'Right'])
            title: Title for the plot
            save: Whether to save the plot
            filename: Filename for saving
        """
        if action_names is None:
            action_names = ['Up', 'Down', 'Left', 'Right']
        
        n_agents = len(policies)
        fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 4))
        
        if n_agents == 1:
            axes = [axes]
        
        for i, (agent_name, policy) in enumerate(policies.items()):
            # Reshape policy to grid
            if len(policy) == env_shape[0] * env_shape[1]:
                policy_grid = policy.reshape(env_shape)
            else:
                policy_grid = policy
            
            # Create heatmap
            im = axes[i].imshow(policy_grid, cmap='tab10', aspect='auto', vmin=0, vmax=len(action_names)-1)
            axes[i].set_title(f"{agent_name}")
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
            
            # Add colorbar with action names
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_ticks(range(len(action_names)))
            cbar.set_ticklabels(action_names)
            
            # Add text annotations
            for y in range(policy_grid.shape[0]):
                for x in range(policy_grid.shape[1]):
                    action_name = action_names[int(policy_grid[y, x])]
                    text = axes[i].text(x, y, action_name,
                                      ha="center", va="center", color="white", fontsize=8)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_epsilon_decay(
        self, 
        epsilon_values: Dict[str, List[float]],
        title: str = "Epsilon Decay",
        save: bool = True,
        filename: str = "epsilon_decay.png"
    ) -> None:
        """Plot epsilon decay over episodes.
        
        Args:
            epsilon_values: Dictionary with agent names as keys and epsilon lists as values
            title: Title for the plot
            save: Whether to save the plot
            filename: Filename for saving
        """
        plt.figure(figsize=(10, 6))
        
        for agent_name, epsilons in epsilon_values.items():
            plt.plot(epsilons, label=agent_name, linewidth=2)
        
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison_summary(
        self, 
        results: Dict[str, Dict[str, Any]],
        save: bool = True,
        filename: str = "comparison_summary.png"
    ) -> None:
        """Create a comprehensive comparison plot.
        
        Args:
            results: Dictionary with agent results
            save: Whether to save the plot
            filename: Filename for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning curves
        ax1 = axes[0, 0]
        for agent_name, data in results.items():
            rewards = data.get('rewards', [])
            if rewards:
                window_size = min(50, len(rewards) // 10)
                if window_size > 1:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    episodes = np.arange(window_size-1, len(rewards))
                    ax1.plot(episodes, moving_avg, label=agent_name, linewidth=2)
        
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Learning Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance
        ax2 = axes[0, 1]
        agent_names = list(results.keys())
        final_rewards = [np.mean(data.get('rewards', [])[-100:]) for data in results.values()]
        bars = ax2.bar(agent_names, final_rewards)
        ax2.set_ylabel("Average Reward (Last 100 episodes)")
        ax2.set_title("Final Performance")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 3: Episode lengths
        ax3 = axes[1, 0]
        for agent_name, data in results.items():
            lengths = data.get('lengths', [])
            if lengths:
                ax3.plot(lengths, label=agent_name, alpha=0.7)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Episode Length")
        ax3.set_title("Episode Lengths")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epsilon decay
        ax4 = axes[1, 1]
        for agent_name, data in results.items():
            epsilons = data.get('epsilons', [])
            if epsilons:
                ax4.plot(epsilons, label=agent_name, linewidth=2)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")
        ax4.set_title("Epsilon Decay")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
