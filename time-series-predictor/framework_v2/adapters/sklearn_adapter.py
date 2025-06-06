import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import joblib
from pathlib import Path

from ..core.base import TimeSeriesModel


class SKLearnAdapter(TimeSeriesModel):
    """
    Adapter to make sklearn-style models work with the new framework.
    Handles both legacy numpy arrays and new DataLoader format.
    """
    
    def __init__(self, sklearn_model, lag_window: int = 5):
        """
        Args:
            sklearn_model: Any sklearn-compatible model with fit/predict methods
            lag_window: Number of lag features (for legacy compatibility)
        """
        self.sklearn_model = sklearn_model
        self.lag_window = lag_window
        self.is_fitted = False
        
    def fit(self, 
            train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the sklearn model.
        
        Args:
            train_data: Either DataLoader or (X, y) numpy arrays
            val_data: Validation data (optional, ignored for sklearn)
            **kwargs: Additional arguments (ignored for sklearn)
            
        Returns:
            Training history (empty for sklearn)
        """
        
        if isinstance(train_data, DataLoader):
            # Convert DataLoader to numpy arrays
            X_train, y_train = self._dataloader_to_arrays(train_data)
        else:
            # Already numpy arrays
            X_train, y_train = train_data
            
        # Fit the sklearn model
        self.sklearn_model.fit(X_train, y_train)
        self.is_fitted = True
        
        return {
            'train_samples': len(X_train),
            'status': 'completed'
        }
    
    def predict(self, data: Union[DataLoader, np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Make predictions using the sklearn model.
        
        Args:
            data: Either DataLoader, numpy array, or (X, y) tuple
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(data, DataLoader):
            X, _ = self._dataloader_to_arrays(data)
        elif isinstance(data, tuple) and len(data) == 2:
            # Direct (X, y) tuple from optimized path
            X, _ = data
        else:
            # Direct numpy array
            X = data
            
        return self.sklearn_model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        params = {
            'model_type': type(self.sklearn_model).__name__,
            'lag_window': self.lag_window
        }
        
        # Get sklearn model parameters
        if hasattr(self.sklearn_model, 'get_params'):
            sklearn_params = self.sklearn_model.get_params()
            params.update(sklearn_params)
            
        return params
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'sklearn_model': self.sklearn_model,
            'lag_window': self.lag_window,
            'is_fitted': self.is_fitted,
            'model_type': type(self.sklearn_model).__name__
        }
        
        # Use joblib for sklearn models (better than pickle)
        joblib.dump(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        save_dict = joblib.load(path)
        
        self.sklearn_model = save_dict['sklearn_model']
        self.lag_window = save_dict['lag_window']
        self.is_fitted = save_dict['is_fitted']
    
    def _dataloader_to_arrays(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataLoader to numpy arrays.
        Handles both sequence data and flat feature data.
        OPTIMIZED VERSION: Uses vectorized operations for better performance.
        """
        # Collect all batches first
        all_X = []
        all_y = []
        
        for batch_X, batch_y in dataloader:
            all_X.append(batch_X.numpy())
            all_y.append(batch_y.numpy())
        
        # Concatenate all batches
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        
        # Handle different input shapes
        if len(X_all.shape) == 3:
            # Sequence data: (batch_size, sequence_length, n_features)
            # Flatten to traditional format for sklearn using vectorized operations
            batch_size, seq_len, n_features = X_all.shape
            
            # Extract week numbers (latest in each sequence)
            weeks = X_all[:, -1, 0]  # Shape: (batch_size,)
            
            # Extract lag features (proficiency scores from all time steps)
            lags_all = X_all[:, :, 1]  # Shape: (batch_size, seq_len)
            
            # Handle lag window sizing vectorized
            if seq_len > self.lag_window:
                # Take last lag_window values
                lags_processed = lags_all[:, -self.lag_window:]
            elif seq_len < self.lag_window:
                # Pad with zeros at the beginning
                pad_width = ((0, 0), (self.lag_window - seq_len, 0))
                lags_processed = np.pad(lags_all, pad_width, mode='constant', constant_values=0)
            else:
                lags_processed = lags_all
            
            # Combine features: [week, lag1, lag2, ..., lagN] for each sample
            X_processed = np.column_stack([weeks, lags_processed])
                    
        else:
            # Already flat features: (batch_size, n_features)
            X_processed = X_all
        
        # Handle targets
        if len(y_all.shape) == 2 and y_all.shape[1] == 1:
            # Shape (batch_size, 1) -> (batch_size,)
            y_processed = y_all.flatten()
        else:
            y_processed = y_all
        
        return X_processed, y_processed


class LegacyFrameworkAdapter:
    """
    Adapter to use the old framework with sklearn models in the new architecture.
    Provides backwards compatibility.
    """
    
    def __init__(self, sklearn_model, lag_window: int = 5):
        # Import legacy framework
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'legacy'))
        
        try:
            from framework import TimeSeriesFramework
            self.legacy_framework = TimeSeriesFramework(sklearn_model, lag_window)
            self.available = True
        except ImportError:
            print("Warning: Legacy framework not available")
            self.available = False
    
    def cross_validate(self, data_path: str, n_splits: int = 5, test_size: int = 1):
        """Run cross-validation using legacy framework."""
        if not self.available:
            raise RuntimeError("Legacy framework not available")
        
        return self.legacy_framework.cross_validate(
            data_path=data_path,
            n_splits=n_splits,
            test_size=test_size
        ) 