#!/usr/bin/env python3
"""
Zero-Inflated Models for Student Proficiency Prediction

These models handle the fact that:
1. ~33% of observations are zeros (no skills mastered)
2. The target is count data (number of skills)
3. There may be different processes for zero vs non-zero outcomes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ZeroInflatedPoissonModel(nn.Module):
    """
    Zero-Inflated Poisson model that separately models:
    1. Binary outcome: Will student master any skills? (logistic regression)
    2. Count outcome: If yes, how many skills? (Poisson regression)
    """
    
    def __init__(self, history_window: int = 5):
        super().__init__()
        self.history_window = history_window
        
        # Binary model (zero vs non-zero)
        self.zero_model = nn.Sequential(
            nn.Linear(2 + history_window * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        # Count model (Poisson parameter)
        self.count_model = nn.Sequential(
            nn.Linear(2 + history_window * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returns three values for training.
        
        Returns:
            - zero_prob: Probability of zero outcome
            - lambda_param: Poisson rate parameter
            - expected_value: Expected value combining both models
        """
        # Extract features (same as student ability model)
        features = self._extract_features(x)
        
        # Zero probability (sigmoid for probability)
        zero_logits = self.zero_model(features)
        zero_prob = torch.sigmoid(zero_logits)
        
        # Poisson rate (exp to ensure positive)
        lambda_logits = self.count_model(features)
        lambda_param = torch.exp(lambda_logits)
        
        # Expected value: P(zero) * 0 + P(non-zero) * E[Poisson]
        expected_value = (1 - zero_prob) * lambda_param
        
        return zero_prob.squeeze(), lambda_param.squeeze(), expected_value.squeeze()
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from sequence data."""
        batch_size, seq_len, n_features = x.shape
        
        # Extract student features
        student_ability = x[:, -1, 7:8]
        student_learning_rate = x[:, -1, 8:9]
        
        # Extract historical data
        if seq_len > self.history_window:
            past_performance = x[:, -self.history_window-1:-1, 4]
            past_difficulty = x[:, -self.history_window-1:-1, 6]
        else:
            past_performance = x[:, :-1, 4]
            past_difficulty = x[:, :-1, 6]
            
            # Pad if needed
            if past_performance.shape[1] < self.history_window:
                pad_size = self.history_window - past_performance.shape[1]
                past_performance = torch.cat([
                    torch.zeros(batch_size, pad_size, device=x.device),
                    past_performance
                ], dim=1)
                past_difficulty = torch.cat([
                    torch.zeros(batch_size, pad_size, device=x.device),
                    past_difficulty
                ], dim=1)
        
        # Concatenate all features
        features = torch.cat([
            student_ability,
            student_learning_rate,
            past_performance,
            past_difficulty
        ], dim=1)
        
        return features


class ZeroInflatedPoissonLoss(nn.Module):
    """Custom loss for zero-inflated Poisson model."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, zero_prob: torch.Tensor, lambda_param: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for ZIP model.
        
        Args:
            zero_prob: Probability of zero outcome
            lambda_param: Poisson rate parameter
            target: Actual count values
        """
        # Clamp lambda to avoid numerical issues
        lambda_param = torch.clamp(lambda_param, min=1e-6, max=50)
        
        # Separate zero and non-zero cases
        is_zero = (target == 0).float()
        
        # Log likelihood for zero observations
        # P(Y=0) = p + (1-p) * exp(-lambda)
        poisson_zero_prob = torch.exp(-lambda_param)
        zero_ll = is_zero * torch.log(zero_prob + (1 - zero_prob) * poisson_zero_prob + 1e-8)
        
        # Log likelihood for non-zero observations
        # P(Y=k) = (1-p) * Poisson(k; lambda)
        # Log Poisson PMF: k * log(lambda) - lambda - log(k!)
        k = target
        log_factorial_k = torch.lgamma(k + 1)  # log(k!)
        poisson_ll = k * torch.log(lambda_param + 1e-8) - lambda_param - log_factorial_k
        nonzero_ll = (1 - is_zero) * (torch.log(1 - zero_prob + 1e-8) + poisson_ll)
        
        # Total negative log likelihood
        return -(zero_ll + nonzero_ll).mean()


class ZeroInflatedPoissonAdapter(nn.Module):
    """Adapter to make ZIP model compatible with standard regression interface."""
    
    def __init__(self, history_window: int = 5):
        super().__init__()
        self.model = ZeroInflatedPoissonModel(history_window)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return only expected value for compatibility."""
        _, _, expected = self.model(x)
        return expected.unsqueeze(1)


class ImprovedStudentModel(nn.Module):
    """
    Improved model with better feature engineering and architecture.
    """
    
    def __init__(self, history_window: int = 5):
        super().__init__()
        self.history_window = history_window
        
        # More sophisticated feature extraction
        # Features: ability(1) + rate(1) + perf(h) + diff(h) + momentum(h) + variance(1)
        feature_size = 2 + history_window * 3 + 1
        
        self.feature_net = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced features."""
        features = self._extract_enhanced_features(x)
        output = self.feature_net(features)
        
        # Apply ReLU to ensure non-negative predictions
        return F.relu(output)
    
    def _extract_enhanced_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract enhanced features including momentum and variance."""
        batch_size, seq_len, n_features = x.shape
        
        # Basic features
        student_ability = x[:, -1, 7:8]
        student_learning_rate = x[:, -1, 8:9]
        
        # Historical data
        if seq_len > self.history_window:
            past_performance = x[:, -self.history_window-1:-1, 4]
            past_difficulty = x[:, -self.history_window-1:-1, 6]
        else:
            past_performance = x[:, :-1, 4]
            past_difficulty = x[:, :-1, 6]
            
            # Pad if needed
            if past_performance.shape[1] < self.history_window:
                pad_size = self.history_window - past_performance.shape[1]
                past_performance = torch.cat([
                    torch.zeros(batch_size, pad_size, device=x.device),
                    past_performance
                ], dim=1)
                past_difficulty = torch.cat([
                    torch.zeros(batch_size, pad_size, device=x.device),
                    past_difficulty
                ], dim=1)
        
        # Calculate momentum (difference between consecutive values)
        momentum = torch.zeros_like(past_performance)
        if past_performance.shape[1] > 1:
            momentum[:, 1:] = past_performance[:, 1:] - past_performance[:, :-1]
        
        # Calculate variance/stability
        perf_std = past_performance.std(dim=1, keepdim=True)
        
        # Concatenate all features
        features = torch.cat([
            student_ability,
            student_learning_rate,
            past_performance,
            past_difficulty,
            momentum,
            perf_std
        ], dim=1)
        
        return features 