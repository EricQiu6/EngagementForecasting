"""
Configuration management for the time series framework.
Provides centralized configuration and eliminates hardcoded values.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from dataclasses import dataclass, asdict

from .core.schema import DataSchema


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_path: str
    schema_name: str = 'legacy'  # Name of predefined schema or 'custom'
    custom_schema: Optional[Dict] = None  # Custom schema definition
    sequence_length: int = 5
    batch_size: int = 32
    validation_split: float = 0.2
    test_size: int = 1
    n_splits: int = 5
    load_in_memory: bool = True
    validate_data: bool = True


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_type: str = 'linear'  # 'linear', 'ridge', 'random_forest', etc.
    model_params: Dict[str, Any] = None
    lag_window: int = 5
    adapter: str = 'sklearn'  # 'sklearn', 'pytorch'
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001
    optimizer: str = 'adam'
    loss_function: str = 'mse'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    output_dir: str = "experiments/outputs"
    random_seed: int = 42
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig(data_path="")
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Create default configs if they don't exist
        self._create_default_configs()
    
    def load_config(self, name: str) -> ExperimentConfig:
        """
        Load configuration by name.
        
        Args:
            name: Configuration name (without extension)
            
        Returns:
            ExperimentConfig object
        """
        # Try different extensions
        for ext in ['.json', '.yaml', '.yml']:
            config_path = self.config_dir / f"{name}{ext}"
            if config_path.exists():
                return self._load_from_file(config_path)
        
        raise FileNotFoundError(f"Configuration '{name}' not found in {self.config_dir}")
    
    def save_config(self, config: ExperimentConfig, name: str, format: str = 'json'):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            name: Configuration name
            format: 'json' or 'yaml'
        """
        if format == 'json':
            path = self.config_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(self._config_to_dict(config), f, indent=2)
        elif format in ['yaml', 'yml']:
            path = self.config_dir / f"{name}.yaml"
            with open(path, 'w') as f:
                yaml.dump(self._config_to_dict(config), f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Configuration saved to {path}")
    
    def create_from_dict(self, config_dict: Dict) -> ExperimentConfig:
        """Create configuration from dictionary."""
        data_dict = config_dict.get('data', {})
        model_dict = config_dict.get('model', {})
        training_dict = config_dict.get('training', {})
        
        return ExperimentConfig(
            name=config_dict.get('name', 'unnamed'),
            description=config_dict.get('description', ''),
            data=DataConfig(**data_dict) if data_dict else DataConfig(data_path=""),
            model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
            training=TrainingConfig(**training_dict) if training_dict else TrainingConfig(),
            output_dir=config_dict.get('output_dir', 'experiments/outputs'),
            random_seed=config_dict.get('random_seed', 42)
        )
    
    def _load_from_file(self, path: Path) -> ExperimentConfig:
        """Load configuration from file."""
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown file extension: {path.suffix}")
        
        return self.create_from_dict(config_dict)
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'name': config.name,
            'description': config.description,
            'data': asdict(config.data),
            'model': asdict(config.model),
            'training': asdict(config.training),
            'output_dir': config.output_dir,
            'random_seed': config.random_seed
        }
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist."""
        # Legacy configuration (for backward compatibility)
        legacy_config = ExperimentConfig(
            name="legacy_default",
            description="Default configuration for legacy data format",
            data=DataConfig(
                data_path="data/data_tidied.csv",
                schema_name="legacy",
                sequence_length=5,
                batch_size=32
            ),
            model=ModelConfig(
                model_type="linear",
                lag_window=2
            )
        )
        
        # Student week configuration
        student_week_config = ExperimentConfig(
            name="student_week_default",
            description="Default configuration for student week data",
            data=DataConfig(
                data_path="data/student_week_aggregations.csv",
                schema_name="student_week",
                sequence_length=8,
                batch_size=64
            ),
            model=ModelConfig(
                model_type="ridge",
                model_params={"alpha": 1.0},
                lag_window=5
            )
        )
        
        # Custom schema example
        custom_schema_config = ExperimentConfig(
            name="custom_example",
            description="Example of custom schema configuration",
            data=DataConfig(
                data_path="data/custom_data.csv",
                schema_name="custom",
                custom_schema={
                    "student_column": "user_id",
                    "time_column": "timestamp",
                    "target_column": "performance",
                    "feature_columns": ["activity", "duration", "difficulty"],
                    "time_format": "%Y-%m-%d"
                }
            ),
            model=ModelConfig(
                model_type="random_forest",
                model_params={"n_estimators": 100, "max_depth": 10}
            )
        )
        
        # Save default configs if they don't exist
        for config, name in [
            (legacy_config, "legacy_default"),
            (student_week_config, "student_week_default"),
            (custom_schema_config, "custom_example")
        ]:
            config_path = self.config_dir / f"{name}.json"
            if not config_path.exists():
                self.save_config(config, name)


def get_schema_from_config(config: ExperimentConfig) -> DataSchema:
    """
    Get DataSchema from ExperimentConfig.
    
    Args:
        config: Experiment configuration
        
    Returns:
        DataSchema object
    """
    if config.data.schema_name == 'custom':
        if not config.data.custom_schema:
            raise ValueError("Custom schema selected but no schema definition provided")
        return DataSchema.from_config(config.data.custom_schema)
    else:
        # Use predefined schema
        from .core.schema import get_schema
        return get_schema(config.data.schema_name)


# Convenience functions
def load_config(name: str, config_dir: Optional[Path] = None) -> ExperimentConfig:
    """Load configuration by name."""
    manager = ConfigManager(config_dir)
    return manager.load_config(name)


def save_config(config: ExperimentConfig, name: str, 
                config_dir: Optional[Path] = None, format: str = 'json'):
    """Save configuration."""
    manager = ConfigManager(config_dir)
    manager.save_config(config, name, format)


def create_config(**kwargs) -> ExperimentConfig:
    """Create configuration from keyword arguments."""
    manager = ConfigManager()
    return manager.create_from_dict(kwargs) 