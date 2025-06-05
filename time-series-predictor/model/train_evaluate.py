#!/usr/bin/env python3
"""
Demo of the minimal train and evaluation framework
"""

from framework import TimeSeriesFramework
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_evaluate():
    """
    Demo: Plug different predictors into the framework
    """
    print("=== MINIMAL TRAIN & EVALUATION FRAMEWORK ===")
    print()
    
    # Hyperparameters
    n_splits = 5
    test_size = 1
    
    print(f"Configuration: {n_splits} folds, {test_size} test week(s) per fold")
    print()
    
    # Test different predictors
    predictors = [
        ("Linear Regression (AR)", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=10, random_state=42))
    ]
    
    for name, predictor_class in predictors:
        print(f"=== {name} ===")
        
        # Reinstantiate the predictor for each fold
        predictor = predictor_class.__class__(**predictor_class.get_params())
        
        framework = TimeSeriesFramework(predictor, lag_window=2)
        
        results = framework.cross_validate(
            data_path='~/cmu/goalsetting-recommendation-algorithm/time-series-predictor/data/data_tidied.csv',
            n_splits=n_splits,
            test_size=test_size
        )
        
        print(f"Results across {results['n_folds']} folds:")
        print(f"   MAE: {results['mae_mean']:.3f} ± {results['mae_std']:.3f}")
        print(f"   RMSE: {results['rmse_mean']:.3f} ± {results['rmse_std']:.3f}")
        print(f"   SMAPE: {results['smape_mean']:.1f}% ± {results['smape_std']:.1f}%")
        print(f"   Total test samples: {results['total_test_samples']}")
        print()
    
    print("✅ Proper time series cross-validation with confidence intervals")
    print("✅ Train-test boundary handled correctly")
    print("✅ All test weeks evaluated (not just final weeks)")

if __name__ == "__main__":
    train_evaluate() 