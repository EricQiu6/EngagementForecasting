"""
Test Improvements with Same CV Settings as Report
================================================

Compare original models vs our improvements using the exact same
cross-validation settings that produced the reported results.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb

from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator


def test_improvements():
    """Test improvements with same CV as report."""
    
    print("Testing Improvements with Report's CV Settings")
    print("=" * 60)
    print("CV: 3 folds, 1 week test window")
    print("Window size: 5\n")
    
    # Setup
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Original models (from report)
    original_models = {
        'Original Lasso': Lasso(alpha=0.1, max_iter=2000),
        'Original RF (shallow)': RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        'Original XGB Linear': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=200,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
    }
    
    # Our improved models
    improved_models = {
        'Improved RF (deeper)': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,  # Deeper trees
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'Lasso + Feature Selection': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'Lasso + Log Transform': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.05, max_iter=3000))
            ]),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        'XGB Linear Improved': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=500,  # More iterations
            learning_rate=0.05,  # Slower learning
            reg_alpha=0.5,  # More L1
            reg_lambda=2.0,  # More L2
            random_state=42
        )
    }
    
    # Test all models
    all_models = {**original_models, **improved_models}
    results = {}
    
    for name, model in all_models.items():
        print(f"\n{name}...", end='', flush=True)
        
        adapter = SKLearnAdapter(
            sklearn_model=model,
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
        
        results[name] = {'mae': mae, 'mae_std': mae_std}
        print(f" MAE={mae:.3f}±{mae_std:.3f}")
    
    # Analysis
    print("\n\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    # Sort by MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
    
    print(f"\n{'Rank':<5} {'Model':<30} {'MAE':<12}")
    print("-" * 50)
    
    for i, (name, res) in enumerate(sorted_results):
        mae_str = f"{res['mae']:.3f}±{res['mae_std']:.3f}"
        print(f"{i+1:<5} {name:<30} {mae_str:<12}")
    
    # Compare original vs improved
    print("\n\nIMPROVEMENT ANALYSIS:")
    print("-" * 50)
    
    # Lasso comparison
    orig_lasso = results['Original Lasso']['mae']
    improved_lasso = min(
        results.get('Lasso + Feature Selection', {'mae': 999})['mae'],
        results.get('Lasso + Log Transform', {'mae': 999})['mae']
    )
    print(f"Lasso: {orig_lasso:.3f} → {improved_lasso:.3f} "
          f"({'✓ improved' if improved_lasso < orig_lasso else '✗ no improvement'})")
    
    # RF comparison
    orig_rf = results['Original RF (shallow)']['mae']
    improved_rf = min(
        results.get('Improved RF (deeper)', {'mae': 999})['mae'],
        results.get('Extra Trees', {'mae': 999})['mae']
    )
    print(f"Random Forest: {orig_rf:.3f} → {improved_rf:.3f} "
          f"({'✓ improved' if improved_rf < orig_rf else '✗ no improvement'})")
    
    # XGB comparison
    orig_xgb = results['Original XGB Linear']['mae']
    improved_xgb = results.get('XGB Linear Improved', {'mae': 999})['mae']
    print(f"XGBoost: {orig_xgb:.3f} → {improved_xgb:.3f} "
          f"({'✓ improved' if improved_xgb < orig_xgb else '✗ no improvement'})")


def main():
    """Run the comparison."""
    print("🔬 Testing if our improvements actually help...\n")
    test_improvements()


if __name__ == "__main__":
    main() 