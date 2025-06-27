"""
Model adapters for the time series framework.
"""

from .sklearn_adapter import SKLearnAdapter, SchemaBasedSKLearnAdapter
from .pytorch_adapter import PyTorchAdapter, SchemaBasedPyTorchAdapter
 
__all__ = [
    'SKLearnAdapter',
    'PyTorchAdapter',
    'SchemaBasedSKLearnAdapter',  # Alias for backward compatibility
    'SchemaBasedPyTorchAdapter'   # Alias for backward compatibility
] 