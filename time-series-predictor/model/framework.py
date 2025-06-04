#!/usr/bin/env python3
"""
Minimal train and evaluation framework for time series predictors
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to Python path so we can import data_processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from data_processing import split_chronologically

class TimeSeriesFramework:
    """
    Minimal framework to plug any time series predictor
    """
    
    def __init__(self, predictor):
        """
        Args:
            predictor: Any object with fit(X, y) and predict(X) methods
        """
        self.predictor = predictor
        self.is_fitted = False
    
    def train(self, data_path='data/data_tidied.csv', K=3):
        """
        Train the predictor on chronologically split data
        
        Args:
            data_path: Path to tidied data
            K: Number of test weeks per student
        """
        # Get split data
        split_result = split_chronologically(data_path=data_path, K=K)
        train_data = split_result['train_data']
        
        # Create features and target
        X_train, y_train = self._prepare_features(train_data)
        
        # Train predictor
        self.predictor.fit(X_train, y_train)
        self.is_fitted = True
        
        return {'train_samples': len(X_train)}
    
    def evaluate(self, data_path='data_tidied.csv', K=3):
        """
        Evaluate the predictor on test data
        
        Args:
            data_path: Path to tidied data  
            K: Number of test weeks per student
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Must train predictor first")
            
        # Get split data
        split_result = split_chronologically(data_path=data_path, K=K)
        test_data = split_result['test_data']
        
        # Create features and target
        X_test, y_test = self._prepare_features(test_data)
        
        # Make predictions
        y_pred = self.predictor.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return metrics
    
    def _prepare_features(self, data):
        """
        Prepare features and target from data
        
        Args:
            data: DataFrame with columns [name, week, proficient]
            
        Returns:
            X, y: Features and target arrays
        """
        features = []
        targets = []
        
        for student in data['name'].unique():
            student_data = data[data['name'] == student].sort_values('week')
            
            # Create lag features for each student
            for i in range(2, len(student_data)):  # Start from index 2 for lag features
                row_features = [
                    student_data.iloc[i]['week'],  # Current week
                    student_data.iloc[i-1]['proficient'],  # Lag 1
                    student_data.iloc[i-2]['proficient']   # Lag 2
                ]
                target = student_data.iloc[i]['proficient']
                
                features.append(row_features)
                targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Metrics
        """
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # SMAPE (symmetric mean absolute percentage error)
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'smape': smape,
            'test_samples': len(y_true)
        } 