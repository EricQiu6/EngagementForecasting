"""
Goal-Based Prediction Algorithm
==============================

A rule-based prediction algorithm that:
1. For windows < 9: Uses random goal (0-60) and formula: prediction = goal + (performance - goal) * 0.5
2. For windows >= 9: Predicts the 50th percentile (median) of the last 9 datapoints

This algorithm is designed to work with the schema-based framework where
the SKLearnAdapter creates engineered features from the time series windows.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class GoalBasedPredictor(BaseEstimator, RegressorMixin):
    """
    A rule-based predictor for goal achievement.
    
    The algorithm adapts based on the available window size:
    - If window < 9: Uses a random goal and adjusts based on last week's performance
    - If window >= 9: Uses the median of the last 9 datapoints
    
    This version is designed to work with the SKLearnAdapter which creates
    engineered features from the time series data.
    
    Parameters
    ----------
    random_min : float, default=0
        Minimum value for random goal generation
    random_max : float, default=60
        Maximum value for random goal generation
    adjustment_factor : float, default=0.5
        Factor for adjusting prediction based on performance-goal difference
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, 
                 random_min=0,
                 random_max=60,
                 adjustment_factor=0.5,
                 random_state=None):
        self.random_min = random_min
        self.random_max = random_max
        self.adjustment_factor = adjustment_factor
        self.random_state = random_state
        
        # Feature metadata from adapter
        self.metadata = None
        
    def set_feature_metadata(self, metadata):
        """Receive feature metadata from SKLearnAdapter."""
        self.metadata = metadata
        # print(f"GoalBasedPredictor received metadata: {metadata}")
        
    def fit(self, X, y):
        """
        Fit the model. This predictor doesn't need training but we store
        the random state for consistency.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features (engineered by SKLearnAdapter)
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Set random state if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Store training target statistics as fallback
        self.training_median_ = np.median(y) if len(y) > 0 else (self.random_min + self.random_max) / 2
        
        # Try to understand the feature structure
        # print(f"GoalBasedPredictor.fit: X shape = {X.shape}")
        # if self.metadata:
        #     print(f"  Lag window: {self.metadata.get('lag_window', 'unknown')}")
        #     print(f"  Feature names available: {self.metadata.get('feature_names') is not None}")
        #     if self.metadata.get('feature_names'):
        #         print(f"  Number of features: {len(self.metadata['feature_names'])}")
        #         # Find lag features for minutes_per_week
        #         lag_features = [i for i, name in enumerate(self.metadata['feature_names']) 
        #                        if 'minutes_per_week_lag' in name]
        #         print(f"  Found {len(lag_features)} lag features for minutes_per_week")
        
        return self
    
    def predict(self, X):
        """
        Make predictions based on the rule-based algorithm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for prediction (engineered by SKLearnAdapter).
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples)
        
        # Set random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state + 1000)
        
        # Determine the effective window size from metadata
        effective_window = self.metadata.get('lag_window', 5) if self.metadata else 5
        
        # Try to extract historical minutes_per_week values from features
        historical_values_indices = []
        
        if self.metadata and self.metadata.get('feature_names'):
            # Look for lag features of minutes_per_week
            feature_names = self.metadata['feature_names']
            
            # First, try to find current minutes_per_week
            current_idx = None
            for i, name in enumerate(feature_names):
                if name == 'current_minutes_per_week':
                    current_idx = i
                    break
            
            # Then find lag features
            lag_indices = []
            for lag in range(1, effective_window + 1):
                for i, name in enumerate(feature_names):
                    if name == f'minutes_per_week_lag{lag}':
                        lag_indices.append(i)
                        break
            
            # Combine current and lags to get historical sequence
            if current_idx is not None:
                historical_values_indices = [current_idx] + lag_indices
            else:
                historical_values_indices = lag_indices
        
        # print(f"GoalBasedPredictor.predict: Found {len(historical_values_indices)} historical value indices")
        
        for i in range(n_samples):
            # Extract historical values if we found the indices
            if historical_values_indices:
                # Get values in reverse chronological order (most recent first)
                historical_values = [X[i, idx] for idx in historical_values_indices if idx < n_features]
                # Filter out invalid values
                valid_values = [v for v in historical_values if v > 0 and np.isfinite(v)]
            else:
                # Fallback: assume last features might be lag values
                # This is a heuristic when we don't have metadata
                if effective_window < n_features:
                    historical_values = X[i, -effective_window:]
                    valid_values = [v for v in historical_values if v > 0 and np.isfinite(v)]
                else:
                    valid_values = []
            
            # Determine effective window size based on available valid data
            actual_window_size = len(valid_values)
            
            if actual_window_size < 9 or effective_window < 9:
                # Case 1: Window < 9 - use random goal and adjustment formula
                
                # Generate random goal
                goal = np.random.uniform(self.random_min, self.random_max)
                
                if valid_values:
                    # Get last week's performance (most recent valid value)
                    performance = valid_values[0]  # Most recent is first
                    
                    # Apply formula: prediction = goal + (performance - goal) * 0.5
                    predictions[i] = goal + (performance - goal) * self.adjustment_factor
                else:
                    # No valid data - just use the random goal
                    predictions[i] = goal
            else:
                # Case 2: Window >= 9 - use median of last 9 datapoints
                
                # Get the last 9 valid datapoints
                last_9_points = valid_values[:9]  # Take first 9 (most recent)
                
                if last_9_points:
                    # Calculate 50th percentile (median)
                    predictions[i] = np.percentile(last_9_points, 50)
                else:
                    # Fallback to training median
                    predictions[i] = self.training_median_
        
        return predictions
    
    def __str__(self):
        return (f"GoalBasedPredictor(random_range=[{self.random_min}, {self.random_max}], "
                f"adjustment_factor={self.adjustment_factor})") 