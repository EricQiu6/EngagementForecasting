import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AveragePredictor:
    """
    Simple baseline predictor that predicts the historical average.
    Compatible with SKLearnAdapter.
    """
    def __init__(self):
        self.mean_value = None
    
    def fit(self, X, y):
        """Fit by computing the mean of training targets"""
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X):
        """Predict the training mean for all samples"""
        if self.mean_value is None:
            raise ValueError("Must fit before predicting")
        return np.full(len(X), self.mean_value)
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {}


class DLinearWrapper:
    """
    Wrapper to make DLinear compatible with SKLearnAdapter.
    Uses the PyTorch implementation but handles feature extraction properly.
    """
    
    def __init__(self, seq_len: int = 5, kernel_size: int = 3):
        """
        Args:
            seq_len: Input sequence length (should match lag_window)
            kernel_size: Size of moving average kernel for trend extraction
        """
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.model = None
        self.metadata = None
        self.target_lag_indices = None
        self.device = torch.device('cpu')
        
    def set_feature_metadata(self, metadata):
        """Receive feature metadata from adapter."""
        self.metadata = metadata
        
    def fit(self, X, y):
        """
        Initialize the PyTorch model and identify target lag features.
        """
        # Find target lag indices from metadata
        self.target_lag_indices = []
        if self.metadata and 'feature_index_map' in self.metadata:
            target_name = self.metadata['target_name']
            feature_map = self.metadata['feature_index_map']
            
            # Get lag features for the target
            for lag in range(1, self.seq_len + 1):
                for possible_name in [f'{target_name}_lag{lag}', f'target_lag{lag}', f'minutes_per_week_lag{lag}']:
                    if possible_name in feature_map:
                        self.target_lag_indices.append(feature_map[possible_name])
                        break
        
        if not self.target_lag_indices:
            # Fallback: assume lags are in positions 9-13
            self.target_lag_indices = list(range(9, 9 + self.seq_len))
        
        # Initialize the original DLinear model
        from .DLinear import Model as DLinearModel
        
        # Create config for original DLinear
        class Config:
            pass
        
        config = Config()
        config.seq_len = len(self.target_lag_indices)
        config.pred_len = 1  # Single step prediction
        config.enc_in = 1    # Single channel (univariate)
        config.individual = False
        
        self.model = DLinearModel(config)
        self.model.to(self.device)
        
        return self
    
    def predict(self, X):
        """
        Make predictions by extracting the target lag sequence and passing through DLinear.
        """
        if self.model is None:
            raise ValueError("Must fit before predicting")
            
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Extract target lag sequences
        if self.target_lag_indices:
            # Shape: (batch_size, seq_len)
            lag_sequences = X_tensor[:, self.target_lag_indices]
        else:
            # Fallback
            lag_sequences = X_tensor[:, 9:9+self.seq_len]
            
        # Forward pass through DLinear
        with torch.no_grad():
            self.model.eval()
            # Original DLinear expects (batch, seq_len, channels)
            lag_sequences_3d = lag_sequences.unsqueeze(-1)  # Add channel dimension
            predictions = self.model(lag_sequences_3d)  # Returns (batch, pred_len=1, channels=1)
            
        return predictions.cpu().numpy().squeeze()
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {'seq_len': self.seq_len, 'kernel_size': self.kernel_size}





class NaiveForecast:
    """
    Naive forecasting baseline that uses the last observed value.
    For time series: uses the last value from each individual sequence.
    Compatible with SKLearnAdapter.
    """
    
    def __init__(self, seasonal_period: int = None):
        """
        Args:
            seasonal_period: If provided, uses seasonal naive (value from same season)
        """
        self.seasonal_period = seasonal_period
        self.global_last = None
        self.last_value_idx = None
        self.metadata = None
        
    def set_feature_metadata(self, metadata):
        """Receive feature metadata from adapter."""
        self.metadata = metadata
        
    def fit(self, X, y):
        """
        Identify which feature contains the most recent target value.
        """
        # Store global last value as fallback
        if len(y) > 0:
            self.global_last = y[-1]
        else:
            self.global_last = 0.0
            
        # Use metadata if available
        if self.metadata and 'feature_index_map' in self.metadata:
            # Find the most recent lag feature for the target
            target_name = self.metadata['target_name']
            feature_map = self.metadata['feature_index_map']
            
            # Look for target_lag1 or minutes_per_week_lag1 (most recent)
            possible_names = [
                f'{target_name}_lag1',
                'target_lag1',
                'minutes_per_week_lag1'  # fallback for time prediction
            ]
            
            for name in possible_names:
                if name in feature_map:
                    self.last_value_idx = feature_map[name]
                    break
        else:
            # Fallback to hardcoded position
            self.last_value_idx = 13
                        
        return self
    
    def predict(self, X):
        """
        Predict using last observed value from each sequence.
        """
        if self.global_last is None:
            raise ValueError("Must fit before predicting")
            
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        if self.last_value_idx is not None:
            # Use the identified lag feature
            for i in range(n_samples):
                last_value = X[i, self.last_value_idx]
                # Use the lag value if it's non-zero, otherwise use global
                if last_value > 0:
                    predictions[i] = last_value
                else:
                    predictions[i] = self.global_last
        else:
            # Fallback to global last value
            predictions[:] = self.global_last
            
        return predictions
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {'seasonal_period': self.seasonal_period}


class LinearTrend:
    """
    Linear trend baseline model.
    Fits a linear trend to the historical sequence of target values.
    Compatible with SKLearnAdapter.
    """
    
    def __init__(self):
        self.slopes = None
        self.intercepts = None
        self.fitted = False
        self.metadata = None
        self.global_slope = None
        self.global_intercept = None
        
    def set_feature_metadata(self, metadata):
        """Receive feature metadata from adapter."""
        self.metadata = metadata
        
    def fit(self, X, y):
        """
        Fit linear trend over the sequence of target values.
        For each sample, we fit a trend over its lag features.
        """
        n_samples = len(y)
        self.slopes = np.zeros(n_samples)
        self.intercepts = np.zeros(n_samples)
        
        # Get lag feature indices from metadata
        lag_indices = []
        if self.metadata and 'feature_index_map' in self.metadata:
            target_name = self.metadata['target_name']
            feature_map = self.metadata['feature_index_map']
            
            # Get all lag features for the target in order
            for lag in range(1, 6):  # lag1 to lag5
                for possible_name in [f'{target_name}_lag{lag}', f'target_lag{lag}', f'minutes_per_week_lag{lag}']:
                    if possible_name in feature_map:
                        lag_indices.append(feature_map[possible_name])
                        break
        
        if not lag_indices:
            # Fallback: assume lags are in positions 9-13
            lag_indices = list(range(9, 14))
            
        # Fit trend for each sample based on its lag sequence
        for i in range(n_samples):
            if lag_indices and all(idx < X.shape[1] for idx in lag_indices):
                # Extract lag values (historical sequence)
                lag_values = np.array([X[i, idx] for idx in lag_indices])
                
                # Filter out zero padding
                non_zero_lags = lag_values[lag_values > 0]
                
                if len(non_zero_lags) > 1:
                    # Fit linear trend over the sequence
                    x = np.arange(len(non_zero_lags))
                    coeffs = np.polyfit(x, non_zero_lags, 1)
                    self.slopes[i] = coeffs[0]
                    self.intercepts[i] = coeffs[1]
                else:
                    # Not enough data for trend
                    self.slopes[i] = 0.0
                    self.intercepts[i] = non_zero_lags[0] if len(non_zero_lags) > 0 else y[i]
            else:
                # No lag features available
                self.slopes[i] = 0.0
                self.intercepts[i] = y[i]
                
        # Also compute global trend as fallback
        self.global_slope = np.mean(self.slopes[self.slopes != 0])
        self.global_intercept = np.mean(self.intercepts)
        
        if np.isnan(self.global_slope):
            self.global_slope = 0.0
            
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict by extending the linear trend from each sequence.
        """
        if not self.fitted:
            raise ValueError("Must fit before predicting")
            
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        # Get lag feature indices from metadata
        lag_indices = []
        if self.metadata and 'feature_index_map' in self.metadata:
            target_name = self.metadata['target_name']
            feature_map = self.metadata['feature_index_map']
            
            # Get all lag features for the target in order
            for lag in range(1, 6):  # lag1 to lag5
                for possible_name in [f'{target_name}_lag{lag}', f'target_lag{lag}', f'minutes_per_week_lag{lag}']:
                    if possible_name in feature_map:
                        lag_indices.append(feature_map[possible_name])
                        break
        
        if not lag_indices:
            # Fallback: assume lags are in positions 9-13
            lag_indices = list(range(9, 14))
            
        for i in range(n_samples):
            if lag_indices and all(idx < X.shape[1] for idx in lag_indices):
                # Extract lag values
                lag_values = np.array([X[i, idx] for idx in lag_indices])
                non_zero_lags = lag_values[lag_values > 0]
                
                if len(non_zero_lags) > 1:
                    # Fit trend for this specific sequence
                    x = np.arange(len(non_zero_lags))
                    coeffs = np.polyfit(x, non_zero_lags, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    # Predict next value (extend trend by one step)
                    next_x = len(non_zero_lags)
                    predictions[i] = slope * next_x + intercept
                else:
                    # Use global trend
                    if len(non_zero_lags) > 0:
                        predictions[i] = non_zero_lags[-1] + self.global_slope
                    else:
                        predictions[i] = self.global_intercept
            else:
                # No features, use global average
                predictions[i] = self.global_intercept
                
        return predictions
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {} 