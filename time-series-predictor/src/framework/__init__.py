"""
Time Series Prediction Framework V2

A flexible, schema-driven framework for time series prediction that eliminates
hardcoded column names and provides configurable data handling.

Key Features:
- Schema-based data handling (no hardcoded columns)
- Configurable feature extraction
- Built-in data validation
- Type consistency enforcement
- Configuration-driven experiments
- Backward compatibility with legacy code
"""

# Core components
from .core import (
    # Base classes
    TimeSeriesModel,
    TimeSeriesDataset,
    MetricsCalculator,
    CrossValidator,
    CrossValidatorFixed,
    
    # Data components
    StudentTimeSeriesDataset,
    DataLoaderFactory,
    convert_legacy_format,
    
    # Schema components
    DataSchema,
    ColumnSchema,
    DataValidator,
    FeatureExtractor,
    get_schema,
    SCHEMAS,
    
    # Schema-based data components
    SchemaBasedTimeSeriesDataset,
    DataLoaderFactoryV2
)

# Adapters
from .adapters import (
    SKLearnAdapter,
    PyTorchAdapter,
    LegacyFrameworkAdapter,
    SchemaBasedSKLearnAdapter,
    LegacyCompatibilityAdapter
)

# Models
from .models import create_model

# Utilities
from .utils import (
    get_device,
    get_device_info,
    print_device_info,
    clear_cuda_cache
)

# Configuration management
from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ConfigManager,
    load_config,
    save_config,
    create_config,
    get_schema_from_config
)

# Version info
__version__ = '2.0.0'

__all__ = [
    # Core classes
    'TimeSeriesModel',
    'TimeSeriesDataset',
    'MetricsCalculator',
    'CrossValidator',
    'CrossValidatorFixed',
    
    # Data components
    'StudentTimeSeriesDataset',
    'SchemaBasedTimeSeriesDataset',
    'DataLoaderFactory',
    'DataLoaderFactoryV2',
    'convert_legacy_format',
    
    # Schema components
    'DataSchema',
    'ColumnSchema',
    'DataValidator',
    'FeatureExtractor',
    'get_schema',
    'SCHEMAS',
    
    # Adapters
    'SKLearnAdapter',
    'SchemaBasedSKLearnAdapter',
    'PyTorchAdapter',
    'LegacyFrameworkAdapter',
    'LegacyCompatibilityAdapter',
    
    # Models
    'create_model',
    
    # Utilities
    'get_device',
    'get_device_info',
    'print_device_info',
    'clear_cuda_cache',
    
    # Configuration
    'ExperimentConfig',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'ConfigManager',
    'load_config',
    'save_config',
    'create_config',
    'get_schema_from_config'
] 