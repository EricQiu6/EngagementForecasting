"""
Reproduce results from gap_features_window_analysis_report.md
=============================================================

This script attempts to reproduce the exact conditions that led to the 
reported results:
- Window size: 5
- Models: Lasso (7.453), Random Forest (7.487), XGBoost Linear (7.614)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator


def test_window_5_models():
    """Test models with window=5 as reported."""
    
    print("Reproducing Window=5 Results from Report")
    print("=" * 60)
    
    # Use the exact schema from the report
    schema = get_schema('time_goal_extended')
    
    # Create dataset with window=5
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,  # Window = 5 as in report
        validate_data=False
    )
    
    print(f"Dataset: {len(dataset)} sequences (should be ~2,910)")
    print(f"Target: {schema.target_column}")
    
    # Define models exactly as in the report
    models = {
        'lasso': {
            'model': Lasso(alpha=0.1, max_iter=2000),
            'expected_mae': 7.453
        },
        'random_forest': {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=3,  # Shallow trees as mentioned
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'expected_mae': 7.487
        },
        'xgb_linear': {
            'model': xgb.XGBRegressor(
                booster='gblinear',
                n_estimators=200,
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'expected_mae': 7.614
        }
    }
    
    # Test with different CV settings
    cv_configs = [
        {'n_splits': 3, 'test_size': 1, 'name': '3-fold, 1 week test'},
        {'n_splits': 5, 'test_size': 1, 'name': '5-fold, 1 week test'},
        {'n_splits': 5, 'test_size': 2, 'name': '5-fold, 2 week test'},
    ]
    
    results = {}
    
    for cv_config in cv_configs:
        print(f"\n\nTesting with {cv_config['name']}:")
        print("-" * 50)
        
        for model_name, model_info in models.items():
            print(f"\n{model_name}...", end='', flush=True)
            
            # Create adapter with lag_window=5 to match sequence_length
            adapter = SKLearnAdapter(
                sklearn_model=model_info['model'],
                schema=schema,
                lag_window=5  # Match sequence_length
            )
            
            # Run cross-validation
            cv = CrossValidator(adapter, dataset)
            cv_results = cv.cross_validate(
                n_splits=cv_config['n_splits'],
                test_size=cv_config['test_size']
            )
            
            mae = cv_results['mae_mean']
            mae_std = cv_results['mae_std']
            
            print(f" MAE={mae:.3f}±{mae_std:.3f} (expected: {model_info['expected_mae']})")
            
            # Store results
            key = f"{model_name}_{cv_config['name']}"
            results[key] = {
                'mae': mae,
                'mae_std': mae_std,
                'expected': model_info['expected_mae'],
                'difference': mae - model_info['expected_mae']
            }
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY: Differences from reported values")
    print("=" * 60)
    
    for key, res in results.items():
        diff_str = f"+{res['difference']:.3f}" if res['difference'] > 0 else f"{res['difference']:.3f}"
        print(f"{key:<40} MAE={res['mae']:.3f} (diff: {diff_str})")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: abs(x[1]['difference']))
    print(f"\nClosest to report: {best_config[0]}")


def test_feature_differences():
    """Check if feature engineering differences might explain the gap."""
    
    print("\n\n" + "=" * 60)
    print("Checking Feature Engineering Differences")
    print("=" * 60)
    
    schema = get_schema('time_goal_extended')
    
    # Create a small sample to inspect features
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Get a sample batch
    sample_x, sample_y = dataset[0]
    print(f"\nRaw sequence shape: {sample_x.shape}")
    print(f"Features in sequence: {schema.feature_columns}")
    
    # Create adapter to see processed features
    dummy_model = Lasso(alpha=0.1)
    adapter = SKLearnAdapter(
        sklearn_model=dummy_model,
        schema=schema,
        lag_window=5
    )
    
    # Get feature names
    feature_names = adapter.get_feature_names()
    if feature_names:
        print(f"\nTotal engineered features: {len(feature_names)}")
        print("\nFeature categories:")
        print(f"- Current values: {sum(1 for f in feature_names if f.startswith('current_'))}")
        print(f"- Lag features: {sum(1 for f in feature_names if '_lag' in f)}")
        print(f"- Change features: {sum(1 for f in feature_names if '_change' in f)}")
        print(f"- Statistical features: {sum(1 for f in feature_names if any(s in f for s in ['_mean', '_std', '_sum', '_trend']))}")
        print(f"- Gap features: {sum(1 for f in feature_names if 'gap' in f)}")


def main():
    """Run all reproduction tests."""
    
    print("🔍 Attempting to reproduce gap_features_window_analysis_report.md results\n")
    
    # Test 1: Try to match window=5 results
    test_window_5_models()
    
    # Test 2: Check feature engineering
    test_feature_differences()
    
    print("\n\nPossible reasons for differences:")
    print("1. Cross-validation randomness (different train/test splits)")
    print("2. Feature engineering implementation details")
    print("3. Data preprocessing differences")
    print("4. Random seeds not fully controlling sklearn models")
    print("5. The report might have used a different evaluation script")


if __name__ == "__main__":
    main() 