"""
Core framework components.

This module provides the foundational classes and utilities for the time series
prediction framework, including data handling, model interfaces, and evaluation.
"""

from .base import (
    TimeSeriesModel,
    TimeSeriesDataset,
    CrossValidator,
    MetricsCalculator
)

from .data import (
    SchemaBasedTimeSeriesDataset,
    DataLoaderFactory
)

from .schema import (
    DataSchema,
    DataValidator,
    FeatureExtractor,
    week_string_to_numeric,
    get_schema
)

__all__ = [
    # Base classes
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
    'get_schema'
] 