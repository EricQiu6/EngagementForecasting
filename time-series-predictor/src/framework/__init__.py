"""
Time Series Prediction Framework

A comprehensive framework for time series prediction with support for
both traditional machine learning and deep learning models.
"""

from .core import (
    # Base classes
    TimeSeriesModel,
    TimeSeriesDataset,
    CrossValidator,
    MetricsCalculator,
    
    # Data handling
    SchemaBasedTimeSeriesDataset,
    DataLoaderFactory,
    
    # Schema and validation
    DataSchema,
    DataValidator,
    FeatureExtractor,
    week_string_to_numeric,
    get_schema
)

from .adapters import (
    SKLearnAdapter,
    PyTorchAdapter,
    SchemaBasedSKLearnAdapter
)

from .models import (
    SimpleMLP,
    SimpleLSTM,
    TimeSeriesCNN,
    AttentionLSTM,
    SimpleTransformer,
    create_model
)

from .utils import (
    get_device,
    get_device_info,
    print_device_info,
    clear_cuda_cache
)

__all__ = [
    # Core components
    'TimeSeriesModel',
    'TimeSeriesDataset', 
    'CrossValidator',
    'MetricsCalculator',
    
    # Data handling
    'SchemaBasedTimeSeriesDataset',
    'DataLoaderFactory',
    
    # Schema and validation
    'DataSchema',
    'DataValidator',
    'FeatureExtractor',
    'week_string_to_numeric',
    'get_schema',
    
    # Adapters
    'SKLearnAdapter',
    'PyTorchAdapter',
    'SchemaBasedSKLearnAdapter',
    
    # Models
    'SimpleMLP',
    'SimpleLSTM',
    'TimeSeriesCNN',
    'AttentionLSTM',
    'SimpleTransformer',
    'create_model',
    
    # Utils
    'get_device',
    'get_device_info',
    'print_device_info',
    'clear_cuda_cache'
] 

# Version info
__version__ = '2.0.0' 