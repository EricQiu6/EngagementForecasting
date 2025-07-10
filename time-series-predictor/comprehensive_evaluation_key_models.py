"""
Simplified Comprehensive Evaluation - Testing Key Improvements
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters import SchemaBasedSKLearnAdapter
from src.framework.core.base import CrossValidator

# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb

# Baselines
from src.framework.models.baselines import AveragePredictor, NaiveForecast


def create_key_models():
    """Create the most important improved models."""
    
    models = {}
    
    # 1. Baselines
    models['baseline_average'] = {
        'model': AveragePredictor(),
        'description': 'Historical average',
        'category': 'baseline'
    }
    
    # 2. Standard Lasso
    models['lasso_standard'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Standard Lasso',
        'category': 'linear'
    }
    
    # 3. Improved Random Forest
    models['rf_improved'] = {
        'model': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'description': 'RF with deeper trees',
        'category': 'ensemble'
    }
    
    # 4. XGBoost Linear
    models['xgb_linear'] = {
        'model': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=500,
            learning_rate=0.05,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42
        ),
        'description': 'XGBoost linear improved',
        'category': 'ensemble'
    }
    
    # 5. Lasso with feature selection
    models['lasso_selected'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Lasso with top 30 features',
        'category': 'feature_selection'
    }
    
    # 6. Lasso with log transformation
    models['lasso_log'] = {
        'model': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.05, max_iter=3000))
            ]),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        'description': 'Lasso with log(1+y)',
        'category': 'transformed'
    }
    
    return models


def run_evaluation(window_size=8):
    """Run evaluation with improved methodology."""
    
    print(f"\nEvaluating with window size: {window_size}")
    print("=" * 60)
    
    # Configuration
    config = {
        'sequence_length': window_size,
        'n_splits': 10,  # More folds
        'test_size': 2,  # 2 weeks test
        'schema': 'time_goal_extended'
    }
    
    # Setup data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    schema = get_schema(config['schema'])
    
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=config['sequence_length'],
        validate_data=False
    )
    
    print(f"Dataset: {len(dataset)} sequences")
    
    # Get models
    models = create_key_models()
    
    # Run evaluation
    results = {}
    
    for name, model_config in models.items():
        print(f"\n{name}...", end='', flush=True)
        start_time = time.time()
        
        try:
            adapter = SchemaBasedSKLearnAdapter(
                sklearn_model=model_config['model'],
                schema=schema,
                lag_window=config['sequence_length']
            )
            
            cv = CrossValidator(adapter, dataset)
            cv_results = cv.cross_validate(
                n_splits=config['n_splits'],
                test_size=config['test_size']
            )
            
            results[name] = {
                'mae': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'category': model_config['category'],
                'time': time.time() - start_time
            }
            
            print(f" MAE={cv_results['mae_mean']:.3f}")
            
        except Exception as e:
            print(f" Failed: {str(e)}")
    
    return results


def main():
    """Main function."""
    
    print("Testing key model improvements...")
    
    # Test window size 8
    results = run_evaluation(window_size=8)
    
    # Show results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Sort by MAE
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'mae' in v],
        key=lambda x: x[1]['mae']
    )
    
    print(f"\n{'Model':<25} {'MAE':<10} {'Category':<15}")
    print("-"*50)
    
    for name, res in sorted_results:
        print(f"{name:<25} {res['mae']:.3f}±{res['mae_std']:.3f} {res['category']:<15}")
    
    # Show improvement
    if sorted_results:
        baseline = next((r for n, r in sorted_results if 'baseline' in n), None)
        best = sorted_results[0][1]
        
        if baseline:
            improvement = (baseline['mae'] - best['mae']) / baseline['mae'] * 100
            print(f"\nImprovement over baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()
