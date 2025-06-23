import numpy as np
import torch
import torch.nn as nn


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


class DLinearPyTorch(nn.Module):
    """
    PyTorch implementation of DLinear for time series forecasting.
    
    Based on "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    Decomposes time series into trend and seasonal components using moving average,
    then applies linear transformations to each component.
    """
    
    def __init__(self, seq_len: int = 5, kernel_size: int = 25):
        """
        Args:
            seq_len: Input sequence length
            kernel_size: Size of moving average kernel for trend extraction
        """
        super().__init__()
        self.seq_len = seq_len
        self.kernel_size = min(kernel_size, seq_len)  # Don't exceed sequence length
        
        # Linear layers for seasonal and trend components
        self.seasonal_linear = nn.Linear(seq_len, 1)
        self.trend_linear = nn.Linear(seq_len, 1)
        
    def _moving_average(self, x):
        """
        Compute moving average for trend extraction.
        x: shape (batch_size, seq_len)
        """
        batch_size, seq_len = x.shape
        
        # Create padding for valid convolution
        pad_size = self.kernel_size // 2
        x_padded = torch.nn.functional.pad(x, (pad_size, pad_size), mode='replicate')
        
        # Create moving average kernel
        kernel = torch.ones(self.kernel_size, device=x.device) / self.kernel_size
        kernel = kernel.view(1, 1, self.kernel_size)
        
        # Apply 1D convolution for moving average
        x_padded = x_padded.unsqueeze(1)  # Add channel dimension
        trend = torch.nn.functional.conv1d(x_padded, kernel, padding=0)
        trend = trend.squeeze(1)  # Remove channel dimension
        
        # Trim to original length
        if trend.shape[1] > seq_len:
            start = (trend.shape[1] - seq_len) // 2
            trend = trend[:, start:start + seq_len]
        
        return trend
    
    def forward(self, x):
        """
        Forward pass through DLinear.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Extract the proficiency scores (assuming last feature is the target)
        if len(x.shape) == 3:
            # Use the proficiency values (feature index 1)
            time_series = x[:, :, 1]  # (batch_size, seq_len)
        else:
            # Handle flattened input
            batch_size = x.shape[0]
            # Reconstruct sequence from flattened features
            # Assuming format: [week, lag1, lag2, ..., lagN]
            lags = x[:, 1:]  # Remove week, keep lags
            time_series = lags[:, :self.seq_len]  # Take first seq_len lags
        
        # Decomposition
        trend = self._moving_average(time_series)
        seasonal = time_series - trend
        
        # Apply linear transformations
        seasonal_output = self.seasonal_linear(seasonal)
        trend_output = self.trend_linear(trend)
        
        # Combine outputs
        output = seasonal_output + trend_output
        
        return output


class NaiveForecast:
    """
    Naive forecasting baseline that uses the last observed value.
    Compatible with SKLearnAdapter.
    """
    
    def __init__(self, seasonal_period: int = None):
        """
        Args:
            seasonal_period: If provided, uses seasonal naive (value from same season)
        """
        self.seasonal_period = seasonal_period
        self.last_values = None
        
    def fit(self, X, y):
        """Store last values for prediction"""
        # For seasonal naive, we'd need more sophisticated logic
        # For now, just store the last value
        if len(y) > 0:
            self.last_values = y[-1]
        else:
            self.last_values = 0.0
        return self
    
    def predict(self, X):
        """Predict using last observed value"""
        if self.last_values is None:
            raise ValueError("Must fit before predicting")
        return np.full(len(X), self.last_values)
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {'seasonal_period': self.seasonal_period}


class LinearTrend:
    """
    Linear trend baseline model.
    Fits a simple linear trend to the time series.
    Compatible with SKLearnAdapter.
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.fitted = False
        
    def fit(self, X, y):
        """Fit linear trend"""
        # Extract time information (assume first column is time/week)
        if len(X.shape) == 2 and X.shape[1] > 0:
            time_values = X[:, 0]
        else:
            # Use indices if no time column
            time_values = np.arange(len(y))
        
        # Fit linear regression y = slope * time + intercept
        n = len(time_values)
        if n > 1:
            sum_x = np.sum(time_values)
            sum_y = np.sum(y)
            sum_xy = np.sum(time_values * y)
            sum_x2 = np.sum(time_values ** 2)
            
            denominator = n * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                self.slope = (n * sum_xy - sum_x * sum_y) / denominator
                self.intercept = (sum_y - self.slope * sum_x) / n
            else:
                self.slope = 0.0
                self.intercept = np.mean(y)
        else:
            self.slope = 0.0
            self.intercept = y[0] if len(y) > 0 else 0.0
            
        self.fitted = True
        return self
    
    def predict(self, X):
        """Predict using linear trend"""
        if not self.fitted:
            raise ValueError("Must fit before predicting")
            
        # Extract time information
        if len(X.shape) == 2 and X.shape[1] > 0:
            time_values = X[:, 0]
        else:
            time_values = np.arange(len(X))
            
        return self.slope * time_values + self.intercept
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {} 