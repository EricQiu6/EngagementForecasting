"""
Test XGBoost Properly with Various Configurations
================================================

XGBoost should perform well on tabular data. Let's test different configurations
to find out why it's underperforming.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator


def test_xgboost_configurations():
    """Test various XGBoost configurations to find optimal settings."""
    
    print("Testing XGBoost Configurations")
    print("=" * 60)
    
    # Setup data with window=5 (as in the report)
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    print(f"Dataset: {len(dataset)} sequences")
    print(f"Using 3-fold CV with 1-week test (same as report)\n")
    
    # Different XGBoost configurations to test
    xgb_configs = {
        # 1. Original configuration (tree-based)
        'xgb_original': {
            'model': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'description': 'Original tree-based config'
        },
        
        # 2. Shallow trees (like Random Forest)
        'xgb_shallow': {
            'model': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=3,  # Shallow like RF
                learning_rate=0.1,
                min_child_weight=10,  # Regularization
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'description': 'Shallow trees (depth=3)'
        },
        
        # 3. More boosting rounds with slower learning
        'xgb_slow_learn': {
            'model': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,  # Slower learning
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'description': 'More rounds, slower learning'
        },
        
        # 4. Regularized version
        'xgb_regularized': {
            'model': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=1.0,  # L1 regularization
                reg_lambda=2.0,  # L2 regularization
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42
            ),
            'description': 'Heavy regularization'
        },
        
        # 5. Linear booster (already tested)
        'xgb_linear': {
            'model': xgb.XGBRegressor(
                booster='gblinear',
                n_estimators=200,
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'description': 'Linear booster (reference)'
        },
        
        # 6. Dart booster (dropout)
        'xgb_dart': {
            'model': xgb.XGBRegressor(
                booster='dart',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                rate_drop=0.1,
                skip_drop=0.5,
                random_state=42
            ),
            'description': 'DART (with dropout)'
        },
        
        # 7. XGBoost with scaling (sometimes helps)
        'xgb_scaled': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ))
            ]),
            'description': 'XGBoost with feature scaling'
        },
        
        # 8. Tuned based on typical best practices
        'xgb_tuned': {
            'model': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                gamma=0.1,  # Minimum loss reduction
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.5,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1
            ),
            'description': 'Carefully tuned parameters'
        }
    }
    
    # Test each configuration
    results = {}
    
    for name, config in xgb_configs.items():
        print(f"\n{name}: {config['description']}...", end='', flush=True)
        
        adapter = SKLearnAdapter(
            sklearn_model=config['model'],
            schema=schema,
            lag_window=5
        )
        
        cv = CrossValidator(adapter, dataset)
        cv_results = cv.cross_validate(
            n_splits=3,  # Same as report
            test_size=1   # Same as report
        )
        
        mae = cv_results['mae_mean']
        mae_std = cv_results['mae_std']
        
        results[name] = {
            'mae': mae,
            'mae_std': mae_std,
            'description': config['description']
        }
        
        print(f" MAE={mae:.3f}±{mae_std:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("XGBOOST CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Sort by MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
    
    print(f"\n{'Rank':<5} {'Configuration':<20} {'MAE':<12} {'Description':<30}")
    print("-" * 70)
    
    for i, (name, res) in enumerate(sorted_results):
        mae_str = f"{res['mae']:.3f}±{res['mae_std']:.3f}"
        print(f"{i+1:<5} {name:<20} {mae_str:<12} {res['description']:<30}")
    
    # Compare to baselines
    print("\n\nCOMPARISON TO REPORT BASELINES:")
    print("-" * 50)
    print(f"Lasso (from report):        MAE = 7.453")
    print(f"Random Forest (from report): MAE = 7.487")
    print(f"XGBoost Linear (report):     MAE = 7.614")
    print(f"Best XGBoost (this test):    MAE = {sorted_results[0][1]['mae']:.3f}")
    
    # Analysis
    print("\n\nANALYSIS:")
    best_config = sorted_results[0][0]
    if sorted_results[0][1]['mae'] < 7.5:
        print("✅ XGBoost CAN be competitive! The issue was hyperparameters.")
    else:
        print("❌ Even with tuning, XGBoost struggles on this dataset.")
        
    print(f"\nBest configuration: {best_config}")
    print(f"Key insight: {sorted_results[0][1]['description']}")


def test_feature_importance():
    """Check XGBoost feature importance to understand what it's learning."""
    
    print("\n\n" + "=" * 60)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Use the best configuration from above
    schema = get_schema('time_goal_extended')
    
    # Create a simple XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Create adapter
    adapter = SKLearnAdapter(
        sklearn_model=model,
        schema=schema,
        lag_window=5
    )
    
    # Get feature names
    feature_names = adapter.get_feature_names()
    
    # Load and prepare a sample of data for fitting
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Get first fold for training
    splits = dataset.get_splits(n_splits=3, test_size=1)
    train_indices, _ = splits[0]
    
    # Create subset
    from torch.utils.data import Subset, DataLoader
    train_subset = Subset(dataset, train_indices[:1000])  # Use first 1000 for speed
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    
    # Fit the model
    adapter.fit(train_loader)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],  # Handle mismatch
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print("-" * 50)
    for i, row in importance_df.head(15).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    # Analyze by feature type
    print("\n\nImportance by Feature Type:")
    print("-" * 50)
    
    feature_types = {
        'current': [],
        'lag': [],
        'change': [],
        'statistical': [],
        'gap': [],
        'other': []
    }
    
    for _, row in importance_df.iterrows():
        feat = row['feature']
        imp = row['importance']
        
        if feat.startswith('current_'):
            feature_types['current'].append(imp)
        elif '_lag' in feat:
            feature_types['lag'].append(imp)
        elif '_change' in feat:
            feature_types['change'].append(imp)
        elif any(s in feat for s in ['_mean', '_std', '_sum', '_trend', '_iqr']):
            feature_types['statistical'].append(imp)
        elif 'gap' in feat:
            feature_types['gap'].append(imp)
        else:
            feature_types['other'].append(imp)
    
    for feat_type, importances in feature_types.items():
        if importances:
            total_imp = sum(importances)
            avg_imp = np.mean(importances)
            print(f"{feat_type:<15} Total: {total_imp:.3f}, Avg: {avg_imp:.4f}, Count: {len(importances)}")


def main():
    """Run all XGBoost tests."""
    print("🔍 Investigating XGBoost Performance\n")
    
    # Test 1: Various configurations
    test_xgboost_configurations()
    
    # Test 2: Feature importance
    test_feature_importance()
    
    print("\n\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 