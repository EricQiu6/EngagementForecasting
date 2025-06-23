"""
Framework V2: Scalable Time Series Prediction Framework

A modern, scalable framework for time series prediction that supports:
- Traditional sklearn models
- Deep learning with PyTorch
- GPU acceleration
- Streaming data processing
- Professional ML workflows
"""

__version__ = "2.0.0"

# Core components
from .core.base import TimeSeriesModel, TimeSeriesDataset, MetricsCalculator, CrossValidator
from .core.data import StudentTimeSeriesDataset, DataLoaderFactory, convert_legacy_format

# Adapters
from .adapters.sklearn_adapter import SKLearnAdapter, LegacyFrameworkAdapter
from .adapters.pytorch_adapter import PyTorchAdapter

# Utilities
from .utils.device import get_device, get_device_info, print_device_info, clear_cuda_cache

__all__ = [
    # Core
    'TimeSeriesModel',
    'TimeSeriesDataset', 
    'MetricsCalculator',
    'CrossValidator',
    'StudentTimeSeriesDataset',
    'DataLoaderFactory',
    'convert_legacy_format',
    
    # Adapters
    'SKLearnAdapter',
    'LegacyFrameworkAdapter', 
    'PyTorchAdapter',
    
    # Utilities
    'get_device',
    'get_device_info',
    'print_device_info',
    'clear_cuda_cache'
] 