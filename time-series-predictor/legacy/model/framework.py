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

from data_processing import create_time_series_splits, get_fold_data, split_chronologically

class TimeSeriesFramework:
    """
    Minimal framework to plug any time series predictor
    """
    
    def __init__(self, predictor, lag_window=2):
        """
        Args:
            predictor: sklearn-compatible model with fit/predict methods
            lag_window: Number of lag features (default AR(2))
        """
        self.predictor = predictor
        self.lag_window = lag_window
        self.is_fitted = False
    
    def cross_validate(self, data_path='data/data_tidied.csv', n_splits=5, test_size=1):
        """
        Perform time series cross-validation
        
        Args:
            data_path: Path to data
            n_splits: Number of CV folds
            test_size: Weeks in test set per fold (default 1)
            
        Returns:
            dict: Mean and std metrics across folds
        """
        print(f"=== TIME SERIES CROSS-VALIDATION ===")
        print(f"Folds: {n_splits}, Test size: {test_size} week(s)")
        print()
        
        # Get CV folds
        global_timeline, splits = create_time_series_splits(data_path, n_splits, test_size)
        
        fold_metrics = []
        total_test_samples = 0
        
        for i, (train_indices, test_indices) in enumerate(splits):
            print(f"Fold {i+1}/{n_splits}:")
            
            # Get fold data
            train_data, test_data = get_fold_data(global_timeline, train_indices, test_indices)
            
            # Train and evaluate on this fold
            self.train_fold(train_data)
            metrics = self.evaluate_fold(train_data, test_data)
            
            fold_metrics.append(metrics)
            total_test_samples += metrics['test_samples']
            
            print(f"  MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, SMAPE: {metrics['smape']:.1f}%")
        
        print()
        
        # Aggregate results across folds
        mae_scores = [m['mae'] for m in fold_metrics]
        rmse_scores = [m['rmse'] for m in fold_metrics]
        smape_scores = [m['smape'] for m in fold_metrics]
        
        results = {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'smape_mean': np.mean(smape_scores),
            'smape_std': np.std(smape_scores),
            'total_test_samples': total_test_samples,
            'n_folds': n_splits
        }
        
        return results
    
    def train_fold(self, train_data):
        """Train model on one CV fold"""
        X_train, y_train = self._prepare_features(train_data)
        self.predictor.fit(X_train, y_train)
        self.is_fitted = True
    
    def evaluate_fold(self, train_data, test_data):
        """Evaluate model on one CV fold with proper boundary handling"""
        if not self.is_fitted:
            raise ValueError("Must train on fold first")
        
        # Use boundary-aware feature preparation
        X_test, y_test = self._prepare_features_with_boundary(train_data, test_data)
        
        if len(X_test) == 0:
            # No valid test samples (e.g., insufficient lag data)
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'smape': float('nan'),
                'test_samples': 0
            }
        
        # Make predictions
        y_pred = self.predictor.predict(X_test)
        
        # Calculate metrics
        return self._calculate_metrics(y_test, y_pred)
    
    def _prepare_features_with_boundary(self, train_data, test_data):
        """
        Prepare test features with proper train-test boundary handling
        
        Use training data for initial test lag features
        """
        features = []
        targets = []
        
        for student in test_data['name'].unique():
            # Get student's training and test data
            student_train = train_data[train_data['name'] == student].sort_values('week')
            student_test = test_data[test_data['name'] == student].sort_values('week')
            
            if len(student_train) == 0:
                continue  # Skip students with no training data
            
            # Combine train+test for continuous timeline
            student_complete = pd.concat([student_train, student_test]).sort_values('week').reset_index(drop=True)
            
            # Find test week positions in complete timeline
            test_weeks = set(student_test['week'])
            
            # Create features for test weeks only
            for i in range(self.lag_window, len(student_complete)):
                current_week = student_complete.iloc[i]['week']
                
                # Only create features for test weeks
                if current_week in test_weeks:
                    row_features = [current_week]  # Start with week
                    
                    # Add lag features (can come from training data)
                    for lag in range(1, self.lag_window + 1):
                        lag_idx = i - lag
                        if lag_idx >= 0:
                            row_features.append(student_complete.iloc[lag_idx]['proficient'])
                        else:
                            # Not enough history - skip this sample
                            break
                    
                    # Only add if we have all required lag features
                    if len(row_features) == self.lag_window + 1:
                        features.append(row_features)
                        targets.append(student_complete.iloc[i]['proficient'])
        
        return np.array(features), np.array(targets)
    
    # Legacy methods for backward compatibility
    def train(self, data_path='data/data_tidied.csv', K=3):
        """
        DEPRECATED: Use cross_validate() for proper evaluation
        Train the predictor on chronologically split data
        
        Args:
            data_path: Path to tidied data
            K: Number of test weeks per student
        """
        print("WARNING: train() is deprecated. Use cross_validate() for proper evaluation.")
        
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
        print("WARNING: evaluate() is deprecated. Use cross_validate() for proper evaluation.")
        
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
        Original feature preparation (for single dataset)
        
        Args:
            data: DataFrame with columns [name, week, proficient]
            
        Returns:
            X, y: Features and target arrays
            dimension of X: (n_samples, lag_window + 1)
            dimension of y: (n_samples,)
        """
        features = []
        targets = []
        
        for student in data['name'].unique():
            student_data = data[data['name'] == student].sort_values('week')
            
            # Create lag features for each student
            for i in range(self.lag_window, len(student_data)):
                row_features = [student_data.iloc[i]['week']]  # Current week
                
                # Add lag features
                for lag in range(1, self.lag_window + 1):
                    row_features.append(student_data.iloc[i-lag]['proficient'])
                
                target = student_data.iloc[i]['proficient']
                
                features.append(row_features)
                targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
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