#!/usr/bin/env python3
"""
Demo of the minimal train and evaluation framework
"""

from model.framework import TimeSeriesFramework
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def demo():
    """
    Demo: Plug different predictors into the framework
    """
    print("=== MINIMAL TRAIN & EVALUATION FRAMEWORK ===")
    print()
    
    # Example 1: Linear Regression (AR model)
    print("1. Linear Regression (AR model):")
    lr_predictor = LinearRegression()
    framework_lr = TimeSeriesFramework(lr_predictor)
    
    train_result = framework_lr.train()
    print(f"   Trained on {train_result['train_samples']} samples")
    
    eval_result = framework_lr.evaluate()
    print(f"   MAE: {eval_result['mae']:.3f}")
    print(f"   RMSE: {eval_result['rmse']:.3f}")
    print(f"   SMAPE: {eval_result['smape']:.1f}%")
    print()
    
    # Example 2: Random Forest
    print("2. Random Forest:")
    rf_predictor = RandomForestRegressor(n_estimators=10, random_state=42)
    framework_rf = TimeSeriesFramework(rf_predictor)
    
    train_result = framework_rf.train()
    print(f"   Trained on {train_result['train_samples']} samples")
    
    eval_result = framework_rf.evaluate()
    print(f"   MAE: {eval_result['mae']:.3f}")
    print(f"   RMSE: {eval_result['rmse']:.3f}")
    print(f"   SMAPE: {eval_result['smape']:.1f}%")
    print()
    
    print("✅ Framework allows plugging any predictor with fit/predict methods")

if __name__ == "__main__":
    demo() 