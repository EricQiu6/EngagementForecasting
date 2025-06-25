"""
Model adapters for the time series framework.
"""

# Original adapters (for backward compatibility)
from .sklearn_adapter import SKLearnAdapter, LegacyFrameworkAdapter
from .pytorch_adapter import PyTorchAdapter

# New schema-based adapters
from .sklearn_adapter_v2 import (
    SchemaBasedSKLearnAdapter,
    LegacyCompatibilityAdapter
)
 
__all__ = [
    # Original adapters
    'SKLearnAdapter',
    'LegacyFrameworkAdapter',
    'PyTorchAdapter',
    
    # Schema-based adapters
    'SchemaBasedSKLearnAdapter',
    'LegacyCompatibilityAdapter'
] 