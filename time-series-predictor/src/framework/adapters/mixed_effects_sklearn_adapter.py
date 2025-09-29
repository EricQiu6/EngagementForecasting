"""
Mixed Effects SKLearn Adapter
============================

This module provides a mixed effects adapter that works with any schema
by extracting student ID from the dataset context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, Optional
from torch.utils.data import DataLoader

from ..core.base import TimeSeriesModel
from .sklearn_adapter import SKLearnAdapter


class SchemaAwareMixedEffectsAdapter(TimeSeriesModel):
    """
    Mixed effects adapter that works with any schema by extracting
    student IDs from the dataset context rather than the feature array.
    
    This adapter requires proper student ID extraction and will fail
    if student IDs cannot be extracted from the dataset.
    """
    
    def __init__(self, sklearn_model, schema: Optional['DataSchema'] = None, 
                 lag_window: int = 5, target_col: str = 'minutes_per_week'):
        """
        Args:
            sklearn_model: The underlying sklearn model (ignored for mixed effects)
            schema: DataSchema for schema-based configuration
            lag_window: Number of lag features
            target_col: Target column name
        """
        self.sklearn_model = sklearn_model  # For compatibility
        self.schema = schema
        self.lag_window = lag_window
        self.target_col = target_col
        self.is_fitted = False
        
        # Store student information during training
        self.student_mapping = {}
        self.global_mean = 0.0
        self.student_effects = {}
        
    def fit(self, train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Fit the mixed effects model by extracting student information from the DataLoader.
        
        Raises:
            ValueError: If student IDs cannot be extracted from the dataset
        """
        
        if not isinstance(train_data, DataLoader):
            raise ValueError("Mixed effects model requires DataLoader input with student ID information")
        
        # Extract data and student information from DataLoader
        all_X = []
        all_y = []
        all_student_ids = []
        
        # We need to get the underlying dataset to extract student IDs
        dataset = train_data.dataset
        
        # Handle case where dataset is wrapped in Subset (during cross-validation)
        actual_dataset = dataset
        if hasattr(dataset, 'dataset'):
            # This is a Subset, get the underlying dataset
            actual_dataset = dataset.dataset
        
        if not (hasattr(actual_dataset, 'data') and hasattr(actual_dataset, 'schema') and hasattr(actual_dataset, 'sequence_index')):
            raise ValueError("Dataset must be schema-based with student ID information for mixed effects model")
        
        # Schema-based dataset - extract student IDs
        student_column = actual_dataset.schema.student_column
        
        # Track sample index across batches
        sample_idx = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_data):
            batch_size = len(batch_X)
            
            # Extract student IDs for this batch
            batch_student_ids = []
            for i in range(batch_size):
                if hasattr(dataset, 'indices'):
                    # This is a Subset, get the actual index
                    actual_idx = dataset.indices[sample_idx]
                else:
                    # Regular dataset
                    actual_idx = sample_idx
                
                if actual_idx >= len(actual_dataset.sequence_index):
                    raise ValueError(f"Sample index {actual_idx} out of bounds for sequence_index")
                
                seq_info = actual_dataset.sequence_index[actual_idx]
                student_id = seq_info['student']
                batch_student_ids.append(student_id)
                sample_idx += 1
            
            all_X.append(batch_X.numpy())
            all_y.append(batch_y.numpy())
            all_student_ids.extend(batch_student_ids)
        
        # Concatenate all data
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        
        # Flatten targets if needed
        if len(y_all.shape) == 2 and y_all.shape[1] == 1:
            y_all = y_all.flatten()
        
        # Create mixed effects model
        self._fit_mixed_effects(X_all, y_all, all_student_ids)
        
        return {
            'train_samples': len(X_all),
            'n_students': len(set(all_student_ids)),
            'status': 'completed'
        }
    
    def _fit_mixed_effects(self, X: np.ndarray, y: np.ndarray, student_ids: list):
        """Fit the actual mixed effects model."""
        
        # Calculate global mean
        self.global_mean = np.mean(y)
        
        # Calculate student-specific effects
        student_means = {}
        student_counts = {}
        
        for i, student_id in enumerate(student_ids):
            if student_id not in student_means:
                student_means[student_id] = []
                student_counts[student_id] = 0
            
            student_means[student_id].append(y[i])
            student_counts[student_id] += 1
        
        # Calculate student effects (deviations from global mean)
        for student_id in student_means:
            if student_counts[student_id] >= 2:  # Only if we have enough data
                student_avg = np.mean(student_means[student_id])
                self.student_effects[student_id] = student_avg - self.global_mean
            else:
                self.student_effects[student_id] = 0.0  # No effect for sparse students
        
        self.is_fitted = True
    
    def predict(self, data: Union[DataLoader, np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Make predictions using mixed effects logic.
        
        Raises:
            ValueError: If model is not fitted or student IDs cannot be extracted
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not isinstance(data, DataLoader):
            raise ValueError("Mixed effects model requires DataLoader input with student ID information")
        
        return self._predict_from_dataloader(data)
    
    def _predict_from_dataloader(self, data: DataLoader) -> np.ndarray:
        """
        Make predictions from DataLoader with student context.
        
        Raises:
            ValueError: If student IDs cannot be extracted from the dataset
        """
        
        predictions = []
        
        dataset = data.dataset
        
        # Handle case where dataset is wrapped in Subset (during cross-validation)
        actual_dataset = dataset
        if hasattr(dataset, 'dataset'):
            # This is a Subset, get the underlying dataset
            actual_dataset = dataset.dataset
        
        if not (hasattr(actual_dataset, 'data') and hasattr(actual_dataset, 'schema') and hasattr(actual_dataset, 'sequence_index')):
            raise ValueError("Dataset must be schema-based with student ID information for mixed effects predictions")
        
        # Extract student IDs
        sample_idx = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(data):
            batch_size = len(batch_X)
            batch_predictions = []
            
            for i in range(batch_size):
                if hasattr(dataset, 'indices'):
                    # This is a Subset, get the actual index
                    actual_idx = dataset.indices[sample_idx]
                else:
                    # Regular dataset
                    actual_idx = sample_idx
                
                if actual_idx >= len(actual_dataset.sequence_index):
                    raise ValueError(f"Sample index {actual_idx} out of bounds for sequence_index")
                
                seq_info = actual_dataset.sequence_index[actual_idx]
                student_id = seq_info['student']
                
                # Use student effect if available
                if student_id in self.student_effects:
                    pred = self.global_mean + self.student_effects[student_id]
                else:
                    pred = self.global_mean
                
                batch_predictions.append(pred)
                sample_idx += 1
            
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'SchemaAwareMixedEffects',
            'lag_window': self.lag_window,
            'target_col': self.target_col,
            'n_students': len(self.student_effects),
            'global_mean': self.global_mean
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'lag_window': self.lag_window,
            'target_col': self.target_col,
            'is_fitted': self.is_fitted,
            'global_mean': self.global_mean,
            'student_effects': self.student_effects,
            'student_mapping': self.student_mapping
        }
        
        joblib.dump(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        
        save_dict = joblib.load(path)
        
        self.lag_window = save_dict['lag_window']
        self.target_col = save_dict['target_col']
        self.is_fitted = save_dict['is_fitted']
        self.global_mean = save_dict['global_mean']
        self.student_effects = save_dict['student_effects']
        self.student_mapping = save_dict.get('student_mapping', {})


# Aliases for backward compatibility
MixedEffectsAdapter = SchemaAwareMixedEffectsAdapter

# Make available for import
__all__ = [
    'SchemaAwareMixedEffectsAdapter',
    'MixedEffectsAdapter'
] 