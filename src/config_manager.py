"""Configuration management for RL project."""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    episodes: int = 1000
    max_steps_per_episode: int = 100
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings."""
    name: str = "FrozenLake-v1"
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {"is_slippery": False}


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    plot_learning_curves: bool = True
    plot_value_functions: bool = True
    plot_policies: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    tensorboard: bool = True
    wandb: bool = False
    log_dir: str = "logs"


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    type: str = "cli"  # Options: cli, streamlit, jupyter
    streamlit_port: int = 8501


@dataclass
class ModelSavingConfig:
    """Configuration for model saving."""
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Main configuration class."""
    training: TrainingConfig
    environment: EnvironmentConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    ui: UIConfig
    model_saving: ModelSavingConfig
    
    def __init__(self, **kwargs):
        """Initialize configuration with defaults."""
        self.training = TrainingConfig(**kwargs.get('training', {}))
        self.environment = EnvironmentConfig(**kwargs.get('environment', {}))
        self.visualization = VisualizationConfig(**kwargs.get('visualization', {}))
        self.logging = LoggingConfig(**kwargs.get('logging', {}))
        self.ui = UIConfig(**kwargs.get('ui', {}))
        self.model_saving = ModelSavingConfig(**kwargs.get('model_saving', {}))


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> Config:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return Config(**config_dict)
    
    @staticmethod
    def save_config(config: Config, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        config_dict = {
            'training': asdict(config.training),
            'environment': asdict(config.environment),
            'visualization': asdict(config.visualization),
            'logging': asdict(config.logging),
            'ui': asdict(config.ui),
            'model_saving': asdict(config.model_saving)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
    @staticmethod
    def get_default_config() -> Config:
        """Get default configuration.
        
        Returns:
            Default configuration
        """
        return Config()
    
    @staticmethod
    def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
        """Merge configuration with overrides.
        
        Args:
            base_config: Base configuration
            override_config: Configuration overrides
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        base_dict = {
            'training': asdict(base_config.training),
            'environment': asdict(base_config.environment),
            'visualization': asdict(base_config.visualization),
            'logging': asdict(base_config.logging),
            'ui': asdict(base_config.ui),
            'model_saving': asdict(base_config.model_saving)
        }
        
        # Merge with overrides
        for key, value in override_config.items():
            if key in base_dict and isinstance(value, dict):
                base_dict[key].update(value)
            else:
                base_dict[key] = value
        
        return Config(**base_dict)
