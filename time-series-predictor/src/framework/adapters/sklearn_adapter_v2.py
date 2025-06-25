"""
Schema-driven sklearn adapter that eliminates hardcoded feature indices.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import joblib
from pathlib import Path

from ..core.base import TimeSeriesModel
from ..core.schema import DataSchema, FeatureExtractor


class SchemaBasedSKLearnAdapter(TimeSeriesModel):
    """
    Adapter to make sklearn-style models work with the framework using schema-based configuration.
    Eliminates hardcoded indices and column assumptions.
    """
    
    def __init__(self, sklearn_model, schema: DataSchema, lag_window: int = 5):
        """
        Args:
            sklearn_model: Any sklearn-compatible model with fit/predict methods
            schema: DataSchema defining the data structure
            lag_window: Number of lag features (for legacy compatibility)
        """
        self.sklearn_model = sklearn_model
        self.schema = schema
        self.feature_extractor = FeatureExtractor(schema)
        self.lag_window = lag_window
        self.is_fitted = False
        
        # Build feature mapping from schema
        self._build_feature_mapping()
        
    def _build_feature_mapping(self):
        """Build mapping of feature names to indices based on schema."""
        self.feature_map = {}
        
        # Map time column if it's in features
        if self.schema.time_column in self.schema.feature_columns:
            self.feature_map['time'] = self.schema.feature_columns.index(self.schema.time_column)
        else:
            self.feature_map['time'] = None
            
        # Map target column if it's in features (for lag features)
        if self.schema.target_column in self.schema.feature_columns:
            self.feature_map['target'] = self.schema.feature_columns.index(self.schema.target_column)
        else:
            self.feature_map['target'] = None
            
        # Map all feature names
        for i, feature in enumerate(self.schema.feature_columns):
            self.feature_map[feature] = i
        
    def fit(self, 
            train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the sklearn model using schema-based data handling.
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
            'feature_dim': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'status': 'completed'
        }
    
    def predict(self, data: Union[DataLoader, np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Make predictions using the sklearn model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(data, DataLoader):
            X, _ = self._dataloader_to_arrays(data)
        elif isinstance(data, tuple) and len(data) == 2:
            # Direct (X, y) tuple
            X, _ = data
        else:
            # Direct numpy array
            X = data
            
        return self.sklearn_model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        params = {
            'model_type': type(self.sklearn_model).__name__,
            'lag_window': self.lag_window,
            'schema': self.schema.to_config()
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
            'schema': self.schema.to_config(),
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
        self.schema = DataSchema.from_config(save_dict['schema'])
        self.lag_window = save_dict['lag_window']
        self.is_fitted = save_dict['is_fitted']
        
        # Rebuild mappings
        self.feature_extractor = FeatureExtractor(self.schema)
        self._build_feature_mapping()
    
    def _dataloader_to_arrays(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataLoader to numpy arrays using schema-based extraction.
        No hardcoded indices - everything is based on the schema.
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
            batch_size, seq_len, n_features = X_all.shape
            
            # Process based on schema configuration
            if self._should_create_lag_features():
                X_processed = self._create_lag_features(X_all)
            else:
                # Flatten sequences for traditional sklearn models
                X_processed = self._flatten_sequences(X_all)
                    
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
    
    def _should_create_lag_features(self) -> bool:
        """Determine if we should create lag features based on schema."""
        # Create lag features if:
        # 1. We have a time column in features
        # 2. We have a target column in features (for creating lags)
        return (self.feature_map.get('time') is not None and 
                self.feature_map.get('target') is not None)
    
    def _create_lag_features(self, X_all: np.ndarray) -> np.ndarray:
        """Create lag features using schema-based indexing."""
        batch_size, seq_len, n_features = X_all.shape
        
        # Get indices from schema
        time_idx = self.feature_map.get('time')
        target_idx = self.feature_map.get('target')
        
        if time_idx is None or target_idx is None:
            # Fallback to flattening if we can't create proper lag features
            return self._flatten_sequences(X_all)
        
        # Extract components using schema indices
        times = X_all[:, -1, time_idx]  # Latest time for each sequence
        
        # Extract lag values from target feature
        lag_values = X_all[:, :, target_idx]  # All time steps for target
        
        # Handle lag window sizing
        if seq_len > self.lag_window:
            # Take last lag_window values
            lags_processed = lag_values[:, -self.lag_window:]
        elif seq_len < self.lag_window:
            # Pad with zeros at the beginning
            pad_width = ((0, 0), (self.lag_window - seq_len, 0))
            lags_processed = np.pad(lag_values, pad_width, mode='constant', constant_values=0)
        else:
            lags_processed = lag_values
        
        # Combine features: [time, lag1, lag2, ..., lagN]
        X_processed = np.column_stack([times, lags_processed])
        
        # Add other features from the last time step if needed
        other_features = []
        for feature_name, idx in self.feature_map.items():
            if feature_name not in ['time', 'target'] and idx is not None:
                other_features.append(X_all[:, -1, idx])
        
        if other_features:
            X_processed = np.column_stack([X_processed] + other_features)
        
        return X_processed
    
    def _flatten_sequences(self, X_all: np.ndarray) -> np.ndarray:
        """Flatten sequence data for models that expect flat features."""
        batch_size, seq_len, n_features = X_all.shape
        
        # Simple flattening: use the last time step's features
        # This is suitable for models that don't explicitly handle sequences
        return X_all[:, -1, :]  # Shape: (batch_size, n_features)


class LegacyCompatibilityAdapter:
    """
    Provides backwards compatibility for existing code while using schema-based internals.
    """
    
    def __init__(self, sklearn_model, lag_window: int = 5, 
                 schema_name: str = 'legacy'):
        """
        Args:
            sklearn_model: sklearn model
            lag_window: number of lag features
            schema_name: name of predefined schema to use
        """
        # Import schema utilities
        from ..core.schema import get_schema
        
        # Get predefined schema
        schema = get_schema(schema_name)
        
        # Create schema-based adapter
        self.adapter = SchemaBasedSKLearnAdapter(
            sklearn_model=sklearn_model,
            schema=schema,
            lag_window=lag_window
        )
    
    def __getattr__(self, name):
        """Forward all calls to the schema-based adapter."""
        return getattr(self.adapter, name) 