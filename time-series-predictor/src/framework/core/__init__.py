"""
Core components of the time series framework.
"""

# Original components (for backward compatibility)
from .base import (
    TimeSeriesModel,
    TimeSeriesDataset,
    MetricsCalculator,
    CrossValidator
)

from .data import (
    StudentTimeSeriesDataset,
    DataLoaderFactory,
    convert_legacy_format,
    week_string_to_numeric,
    safe_float_conversion
)

# New schema-based components
from .schema import (
    DataSchema,
    ColumnSchema,
    DataValidator,
    FeatureExtractor,
    get_schema,
    SCHEMAS,
    week_string_to_numeric as week_to_numeric_v2  # Updated version
)

from .data_v2 import (
    SchemaBasedTimeSeriesDataset,
    DataLoaderFactory as DataLoaderFactoryV2,
    convert_legacy_format as convert_legacy_format_v2
)

# Fixed memory-efficient version
try:
    from .base_fixed import CrossValidator as CrossValidatorFixed
except ImportError:
    CrossValidatorFixed = None

__all__ = [
    # Base classes
    'TimeSeriesModel',
    'TimeSeriesDataset',
    'MetricsCalculator', 
    'CrossValidator',
    'CrossValidatorFixed',
    
    # Original data components
    'StudentTimeSeriesDataset',
    'DataLoaderFactory',
    'convert_legacy_format',
    'week_string_to_numeric',
    'safe_float_conversion',
    
    # Schema components
    'DataSchema',
    'ColumnSchema',
    'DataValidator',
    'FeatureExtractor',
    'get_schema',
    'SCHEMAS',
    
    # Schema-based data components
    'SchemaBasedTimeSeriesDataset',
    'DataLoaderFactoryV2',
    'convert_legacy_format_v2',
    'week_to_numeric_v2'
] 