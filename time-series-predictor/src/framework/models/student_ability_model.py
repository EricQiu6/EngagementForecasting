#!/usr/bin/env python3
"""
Student Ability Model

Implements the formulation:
ŷ_{i,t} = α + β_a * a_i + β_l * l_i + Σ_{j=1}^{h}(β_y^{(j)} * y_{i,t-j} + β_d^{(j)} * d_{i,t-j}) + ε_{i,t}

Where:
- a_i: student ability (from data)
- l_i: student learning rate (from data)
- y_{i,t-j}: past performance (avg_proficiency)
- d_{i,t-j}: past week difficulty
- h: history window size
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.schema import DataSchema


class StudentAbilityLinearModel(nn.Module):
    """
    Schema-aware linear regression model incorporating student ability and learning rate.
    """
    
    def __init__(self, history_window: int = 5, schema: Optional['DataSchema'] = None):
        """
        Args:
            history_window: Number of past weeks to consider
            schema: DataSchema defining feature layout (optional for backward compatibility)
        """
        super().__init__()
        self.history_window = history_window
        self.schema = schema
        
        # Build feature indices from schema if provided
        if schema is not None:
            self.feature_indices = schema.get_feature_indices()
            # Validate required features exist
            required_features = ['student_ability', 'student_learning_rate', 'avg_proficiency', 'week_difficulty']
            missing_features = [f for f in required_features if f not in self.feature_indices]
            if missing_features:
                raise ValueError(f"Schema missing required features: {missing_features}")
        else:
            # Fallback to hardcoded indices for backward compatibility
            self.feature_indices = None
        
        # Model parameters
        self.alpha = nn.Parameter(torch.zeros(1))  # Intercept
        self.beta_a = nn.Parameter(torch.zeros(1))  # Coefficient for ability
        self.beta_l = nn.Parameter(torch.zeros(1))  # Coefficient for learning rate
        
        # Coefficients for past performance (one for each lag)
        self.beta_y = nn.Parameter(torch.zeros(history_window))
        
        # Coefficients for past difficulty (one for each lag)
        self.beta_d = nn.Parameter(torch.zeros(history_window))
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters with small random values."""
        nn.init.normal_(self.alpha, mean=0.0, std=0.1)
        nn.init.normal_(self.beta_a, mean=0.0, std=0.1)
        nn.init.normal_(self.beta_l, mean=0.0, std=0.1)
        nn.init.normal_(self.beta_y, mean=0.0, std=0.1)
        nn.init.normal_(self.beta_d, mean=0.0, std=0.1)
    
    def _get_feature_index(self, feature_name: str, fallback_index: int) -> int:
        """Get feature index from schema or use fallback."""
        if self.feature_indices is not None:
            return self.feature_indices[feature_name]
        else:
            return fallback_index
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Schema-aware forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Features are extracted based on schema or fallback to legacy indices
               
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Get feature indices (schema-based or fallback)
        ability_idx = self._get_feature_index('student_ability', 6)  # Updated fallback
        learning_rate_idx = self._get_feature_index('student_learning_rate', 7)  # Updated fallback
        performance_idx = self._get_feature_index('avg_proficiency', 3)  # Updated fallback
        difficulty_idx = self._get_feature_index('week_difficulty', 5)  # Updated fallback
        
        # Validate indices are within bounds
        max_idx = max(ability_idx, learning_rate_idx, performance_idx, difficulty_idx)
        if max_idx >= n_features:
            raise IndexError(
                f"Feature index {max_idx} out of bounds for tensor with {n_features} features. "
                f"Required indices: ability={ability_idx}, learning_rate={learning_rate_idx}, "
                f"performance={performance_idx}, difficulty={difficulty_idx}"
            )
        
        # Extract the latest values for student ability and learning rate
        # These should be constant across the sequence for each student
        student_ability = x[:, -1, ability_idx]
        student_learning_rate = x[:, -1, learning_rate_idx]
        
        # Extract historical performance and difficulty
        # We need the past h values (not including current timestep)
        if seq_len > self.history_window:
            # Take the last history_window values before the current timestep
            past_performance = x[:, -self.history_window-1:-1, performance_idx]
            past_difficulty = x[:, -self.history_window-1:-1, difficulty_idx]
        else:
            # Pad with zeros if we don't have enough history
            past_performance = x[:, :-1, performance_idx]  # All but last timestep
            past_difficulty = x[:, :-1, difficulty_idx]
            
            # Pad to history_window size
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
        
        # Compute the prediction
        # ŷ = α + β_a * a_i + β_l * l_i + Σ(β_y * y_past + β_d * d_past)
        prediction = self.alpha
        prediction = prediction + self.beta_a * student_ability
        prediction = prediction + self.beta_l * student_learning_rate
        
        # Add contributions from past performance and difficulty
        # Note: We reverse the order so beta_y[0] corresponds to t-1, beta_y[1] to t-2, etc.
        past_performance = past_performance.flip(dims=[1])  # Most recent first
        past_difficulty = past_difficulty.flip(dims=[1])
        
        performance_contribution = torch.sum(self.beta_y * past_performance, dim=1)
        difficulty_contribution = torch.sum(self.beta_d * past_difficulty, dim=1)
        
        prediction = prediction + performance_contribution + difficulty_contribution
        
        return prediction.unsqueeze(1)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients for interpretation."""
        with torch.no_grad():
            coeffs = {
                'alpha': self.alpha.item(),
                'beta_ability': self.beta_a.item(),
                'beta_learning_rate': self.beta_l.item(),
            }
            
            # Add lag coefficients
            for i in range(self.history_window):
                coeffs[f'beta_performance_lag{i+1}'] = self.beta_y[i].item()
                coeffs[f'beta_difficulty_lag{i+1}'] = self.beta_d[i].item()
                
        return coeffs


class StudentAbilityNeuralModel(nn.Module):
    """
    Schema-aware neural network version that can capture non-linear interactions.
    """
    
    def __init__(self, history_window: int = 5, hidden_size: int = 32, schema: Optional['DataSchema'] = None):
        """
        Args:
            history_window: Number of past weeks to consider
            hidden_size: Size of hidden layers
            schema: DataSchema defining feature layout (optional for backward compatibility)
        """
        super().__init__()
        self.history_window = history_window
        self.schema = schema
        
        # Build feature indices from schema if provided
        if schema is not None:
            self.feature_indices = schema.get_feature_indices()
            # Validate required features exist
            required_features = ['student_ability', 'student_learning_rate', 'avg_proficiency', 'week_difficulty']
            missing_features = [f for f in required_features if f not in self.feature_indices]
            if missing_features:
                raise ValueError(f"Schema missing required features: {missing_features}")
        else:
            # Fallback to hardcoded indices for backward compatibility
            self.feature_indices = None
        
        # Input size: ability + learning_rate + history_window * (performance + difficulty)
        input_size = 2 + history_window * 2
        
        # Neural network layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def _get_feature_index(self, feature_name: str, fallback_index: int) -> int:
        """Get feature index from schema or use fallback."""
        if self.feature_indices is not None:
            return self.feature_indices[feature_name]
        else:
            return fallback_index
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Schema-aware forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Features are extracted based on schema or fallback to legacy indices
               
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Get feature indices (schema-based or fallback)
        ability_idx = self._get_feature_index('student_ability', 6)  # Updated fallback
        learning_rate_idx = self._get_feature_index('student_learning_rate', 7)  # Updated fallback
        performance_idx = self._get_feature_index('avg_proficiency', 3)  # Updated fallback
        difficulty_idx = self._get_feature_index('week_difficulty', 5)  # Updated fallback
        
        # Validate indices are within bounds
        max_idx = max(ability_idx, learning_rate_idx, performance_idx, difficulty_idx)
        if max_idx >= n_features:
            raise IndexError(
                f"Feature index {max_idx} out of bounds for tensor with {n_features} features. "
                f"Required indices: ability={ability_idx}, learning_rate={learning_rate_idx}, "
                f"performance={performance_idx}, difficulty={difficulty_idx}"
            )
        
        # Extract features
        student_ability = x[:, -1, ability_idx:ability_idx+1]  # Keep dimension
        student_learning_rate = x[:, -1, learning_rate_idx:learning_rate_idx+1]
        
        # Extract historical performance and difficulty
        if seq_len > self.history_window:
            past_performance = x[:, -self.history_window-1:-1, performance_idx]
            past_difficulty = x[:, -self.history_window-1:-1, difficulty_idx]
        else:
            past_performance = x[:, :-1, performance_idx]
            past_difficulty = x[:, :-1, difficulty_idx]
            
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
        
        # Pass through neural network
        return self.layers(features) 