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
from typing import Dict, Any, Optional, Tuple


class StudentAbilityLinearModel(nn.Module):
    """
    Linear regression model incorporating student ability and learning rate.
    """
    
    def __init__(self, history_window: int = 5):
        """
        Args:
            history_window: Number of past weeks to consider
        """
        super().__init__()
        self.history_window = history_window
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Features expected in order:
               [0]: week_id (numeric)
               [1]: minutes_per_week
               [2]: problems_solved
               [3]: total_opportunities
               [4]: avg_proficiency (target variable)
               [5]: n_skills_measured
               [6]: week_difficulty
               [7]: student_ability
               [8]: student_learning_rate
               
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Extract the latest values for student ability and learning rate
        # These should be constant across the sequence for each student
        student_ability = x[:, -1, 7]  # Last timestep, ability feature
        student_learning_rate = x[:, -1, 8]  # Last timestep, learning rate feature
        
        # Extract historical performance and difficulty
        # We need the past h values (not including current timestep)
        if seq_len > self.history_window:
            # Take the last history_window values before the current timestep
            past_performance = x[:, -self.history_window-1:-1, 4]  # avg_proficiency
            past_difficulty = x[:, -self.history_window-1:-1, 6]   # week_difficulty
        else:
            # Pad with zeros if we don't have enough history
            past_performance = x[:, :-1, 4]  # All but last timestep
            past_difficulty = x[:, :-1, 6]
            
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
    Neural network version that can capture non-linear interactions.
    """
    
    def __init__(self, history_window: int = 5, hidden_size: int = 32):
        """
        Args:
            history_window: Number of past weeks to consider
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.history_window = history_window
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Same feature order as linear model
               
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Extract features
        student_ability = x[:, -1, 7:8]  # Keep dimension
        student_learning_rate = x[:, -1, 8:9]
        
        # Extract historical performance and difficulty
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
        
        # Pass through neural network
        return self.layers(features) 