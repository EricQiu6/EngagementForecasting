"""
Unified SKLearn adapter with optional schema support.
"""

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
    Unified adapter for sklearn-style models with optional schema-based configuration.
    Handles both legacy numpy arrays and DataLoader format.
    """
    
    def __init__(self, sklearn_model, schema: Optional['DataSchema'] = None, lag_window: int = 5):
        """
        Args:
            sklearn_model: Any sklearn-compatible model with fit/predict methods
            schema: Optional DataSchema for schema-based configuration
            lag_window: Number of lag features (for compatibility)
        """
        self.sklearn_model = sklearn_model
        self.schema = schema
        self.lag_window = lag_window
        self.is_fitted = False
        
        # Initialize feature extractor if schema is provided
        if self.schema:
            from ..core.schema import FeatureExtractor
            self.feature_extractor = FeatureExtractor(schema)
            self._build_feature_mapping()
        else:
            self.feature_extractor = None
            self.feature_map = {}
            
        # Pass feature metadata to models that support it
        self._pass_feature_metadata()
        
    def _build_feature_mapping(self):
        """Build mapping of feature names to indices based on schema."""
        self.feature_map = {}
        
        if not self.schema:
            return
            
        # Map all feature names to indices
        for i, feature in enumerate(self.schema.feature_columns):
            self.feature_map[feature] = i
            
        # Special mappings for time and target
        if self.schema.time_column in self.schema.feature_columns:
            self.feature_map['time'] = self.schema.feature_columns.index(self.schema.time_column)
        else:
            self.feature_map['time'] = None
            
        if self.schema.target_column in self.schema.feature_columns:
            self.feature_map['target'] = self.schema.feature_columns.index(self.schema.target_column)
        else:
            self.feature_map['target'] = None
        
    def get_feature_names(self):
        """Get names for all engineered features."""
        if not self.schema:
            return None
            
        feature_names = []
        
        # 1. Current values (from last timestep)
        for feat in self.schema.feature_columns:
            feature_names.append(f'current_{feat}')
        
        # 2. Lag features (based on _create_all_features order)
        lag_features = {
            'target': self.feature_map.get('target'),
            'minutes_per_week': self.feature_map.get('minutes_per_week'),
            'problems_solved': self.feature_map.get('problems_solved'),
            'total_opportunities': self.feature_map.get('total_opportunities'),
            'n_skills_measured': self.feature_map.get('n_skills_measured')
        }
        
        for feat_name, feat_idx in lag_features.items():
            if feat_idx is not None:
                for lag in range(self.lag_window):
                    feature_names.append(f'{feat_name}_lag{lag+1}')
        
        # 3. Change features
        change_features = ['avg_proficiency', 'minutes_per_week', 'problems_solved']
        for feat in change_features:
            if feat in self.feature_map:
                feature_names.append(f'{feat}_recent_change')
                feature_names.append(f'{feat}_avg_change')
        
        # 4. Statistical features
        if 'minutes_per_week' in self.feature_map:
            feature_names.extend([
                'minutes_mean', 'minutes_std', 'minutes_range', 'minutes_iqr'
            ])
        if 'problems_solved' in self.feature_map:
            feature_names.extend([
                'problems_mean', 'problems_sum', 'problems_std'
            ])
        if 'avg_proficiency' in self.feature_map:
            feature_names.append('proficiency_trend')
            feature_names.append('proficiency_acceleration')
            
        # 5. Interaction features
        if 'minutes_per_week' in self.feature_map and 'week_difficulty' in self.feature_map:
            feature_names.append('minutes_x_difficulty')
            
        # 6. Gap features
        if 'minutes_per_week' in self.feature_map:
            feature_names.extend([
                'has_recent_gap', 'weeks_since_last_gap', 'gap_count'
            ])
            
        return feature_names
    
    def _pass_feature_metadata(self):
        """Pass feature metadata to models that support it."""
        if hasattr(self.sklearn_model, 'set_feature_metadata'):
            feature_names = self.get_feature_names()
            
            # Build a mapping of feature types to indices
            feature_index_map = {}
            if feature_names:
                for i, name in enumerate(feature_names):
                    feature_index_map[name] = i
                    
            metadata = {
                'lag_window': self.lag_window,
                'target_name': self.schema.target_column if self.schema else None,
                'feature_names': feature_names,
                'feature_index_map': feature_index_map,
            }
            
            self.sklearn_model.set_feature_metadata(metadata)
        
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
            Training info dictionary
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
        
        # Add schema info if available
        if self.schema:
            params['schema'] = self.schema.__class__.__name__
            params['n_features'] = len(self.schema.feature_columns)
        
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
        
        # Save schema if available
        if self.schema:
            save_dict['schema'] = self.schema.to_config() if hasattr(self.schema, 'to_config') else None
        
        # Use joblib for sklearn models (better than pickle)
        joblib.dump(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        save_dict = joblib.load(path)
        
        self.sklearn_model = save_dict['sklearn_model']
        self.lag_window = save_dict['lag_window']
        self.is_fitted = save_dict['is_fitted']
        
        # Load schema if available
        if 'schema' in save_dict and save_dict['schema']:
            try:
                from ..core.schema import DataSchema, FeatureExtractor
                self.schema = DataSchema.from_config(save_dict['schema'])
                self.feature_extractor = FeatureExtractor(self.schema)
                self._build_feature_mapping()
            except:
                self.schema = None
                self.feature_extractor = None
    
    def _dataloader_to_arrays(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataLoader to numpy arrays.
        Uses schema-based extraction if schema is available, otherwise falls back to legacy behavior.
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
            if self.schema:
                # Use ALL features with schema-based extraction
                X_processed = self._create_all_features(X_all)
            else:
                print("Create valid schema")
                    
        else:
            # Already flat features: (batch_size, n_features)
            X_processed = X_all
            print("already flat features")
        
        # Handle targets
        if len(y_all.shape) == 2 and y_all.shape[1] == 1:
            # Shape (batch_size, 1) -> (batch_size,)
            y_processed = y_all.flatten()
        else:
            y_processed = y_all
        
        return X_processed, y_processed
    
    def _create_all_features(self, X_all: np.ndarray) -> np.ndarray:
        """
        Create features using ALL available data from the schema.
        This includes:
        - Current values of all features
        - Lag values for multiple important features (not just target)
        - Aggregated statistics over the sequence
        - Differences/changes between timesteps
        """
        batch_size, seq_len, n_features = X_all.shape
        feature_list = []
        
        # 1. Current values (from last timestep) for ALL features
        current_features = X_all[:, -1, :]  # Shape: (batch_size, n_features)
        feature_list.append(current_features)
        
        # 2. Lag features for MULTIPLE variables (not just target)
        # Define which features should have lags
        lag_features_to_create = {
            'target': self.feature_map.get('target'),
            'minutes_per_week': self.feature_map.get('minutes_per_week'),
            'problems_solved': self.feature_map.get('problems_solved'),
            'total_opportunities': self.feature_map.get('total_opportunities'),
            'n_skills_measured': self.feature_map.get('n_skills_measured')
        }
        
        # Create lags for each important feature
        for feature_name, feature_idx in lag_features_to_create.items():
            if feature_idx is not None:
                values = X_all[:, :, feature_idx]  # All timesteps for this feature
                
                # Get last lag_window values
                if seq_len > self.lag_window:
                    lags = values[:, -self.lag_window:]
                elif seq_len < self.lag_window:
                    # Pad with zeros
                    pad_width = ((0, 0), (self.lag_window - seq_len, 0))
                    lags = np.pad(values, pad_width, mode='constant', constant_values=0)
                else:
                    lags = values
                
                feature_list.append(lags)
        
        # 3. Differences/changes (first-order differences)
        # These capture trends and momentum
        diff_features = []
        
        # Changes in key metrics
        for feature_name, feature_idx in [
            ('avg_proficiency', self.feature_map.get('avg_proficiency')),
            ('minutes_per_week', self.feature_map.get('minutes_per_week')),
            ('problems_solved', self.feature_map.get('problems_solved'))
        ]:
            if feature_idx is not None:
                values = X_all[:, :, feature_idx]
                # Recent change (last timestep - previous)
                if seq_len > 1:
                    recent_change = values[:, -1] - values[:, -2]
                else:
                    recent_change = np.zeros(batch_size)
                diff_features.append(recent_change)
                
                # Average change over sequence
                if seq_len > 1:
                    all_changes = np.diff(values, axis=1)
                    avg_change = np.mean(all_changes, axis=1)
                else:
                    avg_change = np.zeros(batch_size)
                diff_features.append(avg_change)
        
        if diff_features:
            diff_array = np.column_stack(diff_features)
            feature_list.append(diff_array)
        
        # 4. Statistical features over the sequence
        stats_features = []
        
        # Minutes per week - engagement pattern
        if 'minutes_per_week' in self.feature_map:
            idx = self.feature_map['minutes_per_week']
            values = X_all[:, :, idx]
            stats_features.extend([
                np.mean(values, axis=1),  # Average engagement
                np.std(values, axis=1),   # Variability in engagement
                np.max(values, axis=1) - np.min(values, axis=1),  # Range
                np.percentile(values, 75, axis=1) - np.percentile(values, 25, axis=1)  # IQR
            ])
        
        # Problems solved - practice volume
        if 'problems_solved' in self.feature_map:
            idx = self.feature_map['problems_solved']
            values = X_all[:, :, idx]
            stats_features.extend([
                np.mean(values, axis=1),  # Average practice
                np.sum(values, axis=1),   # Total practice
                np.std(values, axis=1),   # Practice consistency
            ])
        
        # Average proficiency - performance trend
        if 'avg_proficiency' in self.feature_map:
            idx = self.feature_map['avg_proficiency']
            values = X_all[:, :, idx]
            # Calculate trend: slope of linear fit
            x = np.arange(seq_len)
            slopes = np.array([np.polyfit(x, values[i], 1)[0] for i in range(batch_size)])
            stats_features.append(slopes)  # Proficiency trend
            
            # Also add acceleration (second derivative)
            if seq_len > 2:
                accel = np.array([np.polyfit(x, values[i], 2)[0] * 2 for i in range(batch_size)])
                stats_features.append(accel)
        
        if stats_features:
            stats_array = np.column_stack(stats_features)
            feature_list.append(stats_array)
        
        # 5. Interaction features (optional, can be expensive)
        # For example: engagement * difficulty at recent timesteps
        interaction_features = []
        
        if 'minutes_per_week' in self.feature_map and 'week_difficulty' in self.feature_map:
            minutes_idx = self.feature_map['minutes_per_week']
            difficulty_idx = self.feature_map['week_difficulty']
            # Interaction at last timestep
            interaction = X_all[:, -1, minutes_idx] * X_all[:, -1, difficulty_idx]
            interaction_features.append(interaction)
        
        if interaction_features:
            interaction_array = np.column_stack(interaction_features)
            feature_list.append(interaction_array)
        
        # 6. GAP FEATURES - Learning gaps (periods with no activity)
        gap_features = []
        
        if 'minutes_per_week' in self.feature_map:
            minutes_idx = self.feature_map['minutes_per_week']
            minutes_values = X_all[:, :, minutes_idx]  # Shape: (batch_size, seq_len)
            
            # Identify gaps (weeks with 0 minutes)
            is_gap = (minutes_values == 0).astype(float)  # 1 where gap, 0 otherwise
            
            # Feature 1: has_recent_gap - Was there a gap in the last 3 weeks?
            recent_window = min(3, seq_len)
            has_recent_gap = np.any(is_gap[:, -recent_window:], axis=1).astype(float)
            gap_features.append(has_recent_gap)
            
            # Feature 2: weeks_since_last_gap - How many weeks since the last gap?
            weeks_since_gap = np.zeros(batch_size)
            for i in range(batch_size):
                gap_positions = np.where(is_gap[i] == 1)[0]
                if len(gap_positions) > 0:
                    # Find the most recent gap
                    last_gap_pos = gap_positions[-1]
                    weeks_since_gap[i] = seq_len - 1 - last_gap_pos
                else:
                    # No gaps in sequence
                    weeks_since_gap[i] = seq_len  # All weeks had activity
            gap_features.append(weeks_since_gap)
            
            # Feature 3: gap_count - Total number of gaps in the sequence
            gap_count = np.sum(is_gap, axis=1)
            gap_features.append(gap_count)
        
        if gap_features:
            gap_array = np.column_stack(gap_features)
            feature_list.append(gap_array)
        
        # 7. Combine all features
        X_processed = np.hstack(feature_list)
        
        return X_processed
    
    def _should_create_lag_features(self) -> bool:
        """Determine if we should create lag features based on schema."""
        # Create lag features if we have both time and target columns in features
        return (self.feature_map.get('time') is not None and 
                self.feature_map.get('target') is not None)
    
    def _create_lag_features_schema(self, X_all: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: This method only used time and target.
        Use _create_all_features instead for comprehensive feature extraction.
        """
        # Redirect to the new comprehensive method
        return self._create_all_features(X_all)


# Alias for backward compatibility
SchemaBasedSKLearnAdapter = SKLearnAdapter


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