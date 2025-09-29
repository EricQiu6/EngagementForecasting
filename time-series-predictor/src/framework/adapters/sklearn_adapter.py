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
            
        # 7. Class-level features
        if 'avg_proficiency' in self.feature_map:
            feature_names.extend([
                'performance_vs_class_mean_prof', 'class_percentile_rank_prof', 'class_improvement_trend_prof'
            ])
        
        if 'minutes_per_week' in self.feature_map:
            feature_names.extend([
                'performance_vs_class_mean_mins', 'class_percentile_rank_mins', 'class_improvement_trend_mins'
            ])
        
        # 8. Prior achievement features
        if 'avg_proficiency' in self.feature_map:
            feature_names.extend([
                'starting_ability_quartile', 'performance_consistency_score', 'learning_acceleration_capacity'
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
                # Handle student ID for different model types
                X_processed = self._handle_student_id_features(X_processed)
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
        
        # 7. CLASS-LEVEL FEATURES (for both avg_proficiency and minutes_per_week)
        class_features = []
        
        # CRITICAL: Only calculate class features if we have sufficient batch size
        # Small batches (< 10 samples) make class statistics meaningless and numerically unstable
        MIN_BATCH_SIZE_FOR_CLASS_FEATURES = 10
        
        if batch_size >= MIN_BATCH_SIZE_FOR_CLASS_FEATURES:
            # Get the full dataset to calculate class statistics
            # Note: This assumes we can access the full DataLoader dataset
            # For now, we'll calculate class stats from the current batch
            # In production, this would need access to the full training set
            
            # Class features for avg_proficiency
            if 'avg_proficiency' in self.feature_map:
                proficiency_idx = self.feature_map['avg_proficiency']
                current_proficiency = X_all[:, -1, proficiency_idx]  # Current week proficiency
                
                # Class mean proficiency (approximated from current batch)
                class_mean_proficiency = np.mean(current_proficiency)
                performance_vs_class_mean_prof = current_proficiency - class_mean_proficiency
                # Handle NaN/inf values
                performance_vs_class_mean_prof = np.nan_to_num(performance_vs_class_mean_prof, nan=0.0, posinf=0.0, neginf=0.0)
                class_features.append(performance_vs_class_mean_prof)
                
                # Class percentile rank for proficiency
                try:
                    from scipy.stats import rankdata
                    if len(current_proficiency) > 1:
                        percentile_rank_prof = rankdata(current_proficiency, method='average') / len(current_proficiency) * 100
                        percentile_rank_prof = np.nan_to_num(percentile_rank_prof, nan=50.0, posinf=100.0, neginf=0.0)
                    else:
                        percentile_rank_prof = np.full(batch_size, 50.0)  # Default to median
                    class_features.append(percentile_rank_prof)
                except:
                    # Fallback if scipy fails
                    class_features.append(np.full(batch_size, 50.0))
                
                # Class improvement trend for proficiency (using available sequence data)
                if seq_len >= 3:
                    try:
                        all_prof_values = X_all[:, :, proficiency_idx]  # All timesteps
                        batch_means_over_time = np.mean(all_prof_values, axis=0)  # Mean across students for each timestep
                        # Calculate trend slope with error handling
                        x_trend = np.arange(len(batch_means_over_time))
                        
                        # Check for valid data
                        if len(batch_means_over_time) >= 2 and not np.all(np.isnan(batch_means_over_time)):
                            # Remove NaN values
                            valid_mask = ~np.isnan(batch_means_over_time)
                            if np.sum(valid_mask) >= 2:
                                x_valid = x_trend[valid_mask]
                                y_valid = batch_means_over_time[valid_mask]
                                class_trend_prof = np.polyfit(x_valid, y_valid, 1)[0]
                                class_trend_prof = np.nan_to_num(class_trend_prof, nan=0.0, posinf=0.0, neginf=0.0)
                            else:
                                class_trend_prof = 0.0
                        else:
                            class_trend_prof = 0.0
                            
                        class_trend_prof_array = np.full(batch_size, class_trend_prof)
                        class_features.append(class_trend_prof_array)
                    except:
                        # Fallback to zero trend if calculation fails
                        class_features.append(np.zeros(batch_size))
                else:
                    class_features.append(np.zeros(batch_size))
            
            # Class features for minutes_per_week
            if 'minutes_per_week' in self.feature_map:
                minutes_idx = self.feature_map['minutes_per_week']
                current_minutes = X_all[:, -1, minutes_idx]  # Current week minutes
                
                # Class mean minutes (approximated from current batch)
                class_mean_minutes = np.mean(current_minutes)
                performance_vs_class_mean_mins = current_minutes - class_mean_minutes
                # Handle NaN/inf values
                performance_vs_class_mean_mins = np.nan_to_num(performance_vs_class_mean_mins, nan=0.0, posinf=0.0, neginf=0.0)
                class_features.append(performance_vs_class_mean_mins)
                
                # Class percentile rank for minutes
                try:
                    if len(current_minutes) > 1:
                        percentile_rank_mins = rankdata(current_minutes, method='average') / len(current_minutes) * 100
                        percentile_rank_mins = np.nan_to_num(percentile_rank_mins, nan=50.0, posinf=100.0, neginf=0.0)
                    else:
                        percentile_rank_mins = np.full(batch_size, 50.0)  # Default to median
                    class_features.append(percentile_rank_mins)
                except:
                    # Fallback if calculation fails
                    class_features.append(np.full(batch_size, 50.0))
                
                # Class improvement trend for minutes (using available sequence data)
                if seq_len >= 3:
                    try:
                        all_mins_values = X_all[:, :, minutes_idx]  # All timesteps
                        batch_means_over_time = np.mean(all_mins_values, axis=0)  # Mean across students for each timestep
                        # Calculate trend slope with error handling
                        x_trend = np.arange(len(batch_means_over_time))
                        
                        # Check for valid data
                        if len(batch_means_over_time) >= 2 and not np.all(np.isnan(batch_means_over_time)):
                            # Remove NaN values
                            valid_mask = ~np.isnan(batch_means_over_time)
                            if np.sum(valid_mask) >= 2:
                                x_valid = x_trend[valid_mask]
                                y_valid = batch_means_over_time[valid_mask]
                                class_trend_mins = np.polyfit(x_valid, y_valid, 1)[0]
                                class_trend_mins = np.nan_to_num(class_trend_mins, nan=0.0, posinf=0.0, neginf=0.0)
                            else:
                                class_trend_mins = 0.0
                        else:
                            class_trend_mins = 0.0
                            
                        class_trend_mins_array = np.full(batch_size, class_trend_mins)
                        class_features.append(class_trend_mins_array)
                    except:
                        # Fallback to zero trend if calculation fails
                        class_features.append(np.zeros(batch_size))
                else:
                    class_features.append(np.zeros(batch_size))
        
        else:
            # FALLBACK: For small batches, use default class feature values
            # This prevents numerical instability with tiny fold sizes
            n_class_features = 0
            
            if 'avg_proficiency' in self.feature_map:
                n_class_features += 3  # performance_vs_class_mean_prof, percentile_rank_prof, class_trend_prof
            
            if 'minutes_per_week' in self.feature_map:
                n_class_features += 3  # performance_vs_class_mean_mins, percentile_rank_mins, class_trend_mins
            
            if n_class_features > 0:
                # Create default class features (all zeros/neutral values)
                default_class_features = []
                
                if 'avg_proficiency' in self.feature_map:
                    default_class_features.append(np.zeros(batch_size))  # performance_vs_class_mean_prof
                    default_class_features.append(np.full(batch_size, 50.0))  # percentile_rank_prof (median)
                    default_class_features.append(np.zeros(batch_size))  # class_trend_prof
                
                if 'minutes_per_week' in self.feature_map:
                    default_class_features.append(np.zeros(batch_size))  # performance_vs_class_mean_mins
                    default_class_features.append(np.full(batch_size, 50.0))  # percentile_rank_mins (median)
                    default_class_features.append(np.zeros(batch_size))  # class_trend_mins
                
                class_features.extend(default_class_features)
        
        if class_features:
            class_array = np.column_stack(class_features)
            # Additional safety check for the entire class array
            class_array = np.nan_to_num(class_array, nan=0.0, posinf=0.0, neginf=0.0)
            feature_list.append(class_array)
        
        # 8. PRIOR ACHIEVEMENT FEATURES
        prior_achievement_features = []
        
        # 1. Starting ability quartile - careful handling of first 2 weeks
        if 'avg_proficiency' in self.feature_map and seq_len >= 2:
            proficiency_idx = self.feature_map['avg_proficiency']
            
            # Use first 2 weeks of data to establish baseline
            early_weeks = min(2, seq_len)
            early_proficiency = X_all[:, :early_weeks, proficiency_idx]  # First 1-2 weeks
            early_avg = np.mean(early_proficiency, axis=1)  # Average over early weeks
            
            # Handle NaN values in early_avg
            early_avg = np.nan_to_num(early_avg, nan=0.5)  # Default to middle proficiency
            
            # Calculate quartiles from the current batch (approximation)
            try:
                if len(early_avg) > 4:  # Need at least 4 samples for reliable quartiles
                    quartile_25 = np.percentile(early_avg, 25)
                    quartile_50 = np.percentile(early_avg, 50)
                    quartile_75 = np.percentile(early_avg, 75)
                else:
                    # Use fixed quartiles if too few samples
                    quartile_25 = 0.25
                    quartile_50 = 0.5
                    quartile_75 = 0.75
                
                # Assign quartile (1-4)
                starting_ability_quartile = np.ones(batch_size)
                starting_ability_quartile[early_avg > quartile_25] = 2
                starting_ability_quartile[early_avg > quartile_50] = 3
                starting_ability_quartile[early_avg > quartile_75] = 4
                
                prior_achievement_features.append(starting_ability_quartile)
            except:
                # Fallback to quartile 2 (average) if calculation fails
                prior_achievement_features.append(np.full(batch_size, 2.0))
        elif seq_len < 2:
            # For very early weeks, use current performance as proxy
            if 'avg_proficiency' in self.feature_map:
                proficiency_idx = self.feature_map['avg_proficiency']
                current_proficiency = X_all[:, -1, proficiency_idx]
                # Assign quartile 2 as default (average) for new students
                starting_ability_quartile = np.full(batch_size, 2.0)
                prior_achievement_features.append(starting_ability_quartile)
        
        # 2. Performance consistency score
        if 'avg_proficiency' in self.feature_map and seq_len >= 3:
            proficiency_idx = self.feature_map['avg_proficiency']
            all_proficiency = X_all[:, :, proficiency_idx]
            
            # Calculate standard deviation across time for each student
            proficiency_std = np.std(all_proficiency, axis=1)
            # Handle NaN and very small values
            proficiency_std = np.nan_to_num(proficiency_std, nan=0.1)  # Default std
            proficiency_std = np.maximum(proficiency_std, 1e-8)  # Avoid division by zero
            
            # Consistency score: higher is more consistent
            consistency_score = 1.0 / (1.0 + proficiency_std)
            consistency_score = np.nan_to_num(consistency_score, nan=0.5)  # Default consistency
            consistency_score = np.clip(consistency_score, 0.0, 1.0)  # Ensure valid range
            
            prior_achievement_features.append(consistency_score)
        elif seq_len < 3:
            # Default consistency for new students
            consistency_score = np.full(batch_size, 0.5)  # Medium consistency
            prior_achievement_features.append(consistency_score)
        
        # 3. Learning acceleration capacity - max improvement over any 3-week window
        if 'avg_proficiency' in self.feature_map and seq_len >= 3:
            proficiency_idx = self.feature_map['avg_proficiency']
            all_proficiency = X_all[:, :, proficiency_idx]
            
            max_acceleration = np.zeros(batch_size)
            for i in range(batch_size):
                try:
                    student_prof = all_proficiency[i]
                    # Handle NaN values
                    student_prof = np.nan_to_num(student_prof, nan=0.0)
                    
                    max_accel = 0.0
                    
                    # Check all possible 3-week windows
                    for window_start in range(len(student_prof) - 2):
                        window_improvement = student_prof[window_start + 2] - student_prof[window_start]
                        window_improvement = np.nan_to_num(window_improvement, nan=0.0)
                        max_accel = max(max_accel, window_improvement)
                    
                    max_acceleration[i] = max_accel
                except:
                    max_acceleration[i] = 0.0  # Fallback
            
            # Ensure valid range
            max_acceleration = np.nan_to_num(max_acceleration, nan=0.0)
            max_acceleration = np.clip(max_acceleration, -1.0, 1.0)  # Reasonable bounds
            
            prior_achievement_features.append(max_acceleration)
        elif seq_len < 3:
            # For early weeks, use current trend as proxy
            if 'avg_proficiency' in self.feature_map and seq_len >= 2:
                proficiency_idx = self.feature_map['avg_proficiency']
                values = X_all[:, :, proficiency_idx]
                # Simple improvement: last - first
                improvement = values[:, -1] - values[:, 0]
                improvement = np.nan_to_num(improvement, nan=0.0)
                improvement = np.clip(improvement, -1.0, 1.0)  # Reasonable bounds
                prior_achievement_features.append(improvement)
            else:
                # Default for very new students
                default_acceleration = np.zeros(batch_size)
                prior_achievement_features.append(default_acceleration)
        
        if prior_achievement_features:
            prior_achievement_array = np.column_stack(prior_achievement_features)
            # Additional safety check for the entire prior achievement array
            prior_achievement_array = np.nan_to_num(prior_achievement_array, nan=0.0, posinf=0.0, neginf=0.0)
            feature_list.append(prior_achievement_array)
        
        # 9. Combine all features
        X_processed = np.hstack(feature_list)
        
        return X_processed
    
    def _handle_student_id_features(self, X_processed: np.ndarray) -> np.ndarray:
        """
        Handle student ID features for different model types.
        
        For most sklearn models, we'll remove the raw student ID since they 
        already have student-derived features (student_ability, student_learning_rate).
        Mixed effects models will be handled by their special adapter.
        """
        if not self.schema or self.schema.student_id_strategy.strategy_type != 'universal':
            return X_processed
            
        # Check if student ID is the first feature (for universal schema)
        if (self.schema.feature_columns and 
            self.schema.feature_columns[0] == self.schema.student_column):
            
            # For regular sklearn models, remove the student ID column
            # since they already have student_ability and student_learning_rate
            # which capture the important student-specific information
            
            # Check the model type
            model_name = type(self.sklearn_model).__name__.lower()
            
            if any(term in model_name for term in ['mixed', 'hierarchical', 'multilevel']):
                # Keep student ID for mixed effects models (though they use special adapters)
                return X_processed
            else:
                # Remove student ID for regular sklearn models
                # They already have student_ability and student_learning_rate
                return X_processed[:, 1:]  # Remove first column (student ID)
        
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