"""
Mixed Effects SKLearn Adapter
============================

This module provides imports for mixed effects models to work with the 
comprehensive evaluation framework.
"""

# Import the actual implementations from the true_mixed_effects files
from .true_mixed_effects_sklearn_wrapper import TrueMixedEffectsSKLearnWrapper
from .true_mixed_effects_adapter import TrueMixedEffectsModel

# Alias them to the expected names
MixedEffectsSKLearnWrapper = TrueMixedEffectsSKLearnWrapper
MixedEffectsSKLearnAdapter = TrueMixedEffectsModel

# Make them available for import
__all__ = ['MixedEffectsSKLearnWrapper', 'MixedEffectsSKLearnAdapter'] 