"""
Test recommended models based on data analysis insights.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, skipping")

from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator


def create_recommended_models():
    """Create models recommended based on data analysis."""
    
    models = {
        # 1. Regularized Linear Models (for multicollinearity)
        'ridge': {
            'model': Ridge(alpha=1.0),
            'description': 'Ridge with L2 regularization',
            'category': 'linear'
        },
        
        'lasso': {
            'model': Lasso(alpha=0.1, max_iter=2000),
            'description': 'Lasso with L1 regularization (feature selection)',
            'category': 'linear'
        },
        
        'elastic_net': {
            'model': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'description': 'ElasticNet combining L1 and L2',
            'category': 'linear'
        },
        
        # 2. Robust Linear Models (for outliers)
        'huber': {
            'model': HuberRegressor(epsilon=1.35, max_iter=200),
            'description': 'Huber regression (robust to outliers)',
            'category': 'linear'
        },
        
        'ransac': {
            'model': RANSACRegressor(min_samples=0.5, max_trials=100, random_state=42),
            'description': 'RANSAC (very robust to outliers)',
            'category': 'linear'
        },
        
        # 3. Linear SVM (captures linear patterns with margin)
        'linear_svr': {
            'model': LinearSVR(epsilon=0.1, C=1.0, max_iter=2000, random_state=42),
            'description': 'Linear Support Vector Regression',
            'category': 'linear'
        },
        
        # 4. Tree ensembles with limited depth (prevent overfitting)
        'rf_shallow': {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=3,  # Very shallow trees
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'description': 'Random Forest with shallow trees',
            'category': 'ensemble'
        },
        
        'extra_trees': {
            'model': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'description': 'Extra Trees (more random than RF)',
            'category': 'ensemble'
        },
        
        # 5. Gradient Boosting with linear base learners
        'xgb_linear': {
            'model': xgb.XGBRegressor(
                booster='gblinear',
                n_estimators=200,
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'description': 'XGBoost with linear booster',
            'category': 'ensemble'
        }
    }
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        models['lgbm_low_leaves'] = {
            'model': LGBMRegressor(
                n_estimators=200,
                num_leaves=8,  # Very few leaves
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'description': 'LightGBM with limited complexity',
            'category': 'ensemble'
        }
    
    return models


def test_feature_selection():
    """Test if feature selection improves XGBoost performance."""
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.pipeline import Pipeline
    
    models = {
        'xgb_select_k_best': {
            'model': Pipeline([
                ('selector', SelectKBest(f_regression, k=20)),
                ('model', xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42))
            ]),
            'description': 'XGBoost with top 20 features (F-statistic)',
            'category': 'feature_selection'
        },
        
        'xgb_mutual_info': {
            'model': Pipeline([
                ('selector', SelectKBest(mutual_info_regression, k=20)),
                ('model', xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42))
            ]),
            'description': 'XGBoost with top 20 features (Mutual Information)',
            'category': 'feature_selection'
        }
    }
    
    return models


def main():
    """Run evaluation of recommended models."""
    print("Testing Recommended Models")
    print("=" * 60)
    
    # Setup
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=15,  # Extended to 15
        validate_data=False
    )
    
    # Get all models
    models = create_recommended_models()
    models.update(test_feature_selection())
    
    # Run quick evaluation (fewer folds for speed)
    results = {}
    
    for name, config in models.items():
        print(f"\nTesting {name}...")
        try:
            adapter = SKLearnAdapter(
                sklearn_model=config['model'],
                schema=schema,
                lag_window=15  # Extended to 15
            )
            
            cv = CrossValidator(adapter, dataset)
            cv_results = cv.cross_validate(n_splits=3, test_size=1)
            
            results[name] = {
                'mae': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'category': config['category'],
                'description': config['description']
            }
            
            print(f"✓ {name}: MAE = {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
            
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by MAE)")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
    for name, res in sorted_results:
        print(f"{name:20} MAE: {res['mae']:6.3f} ± {res['mae_std']:.3f}  ({res['category']})")
    
    # Category summary
    print("\nBest by category:")
    categories = {}
    for name, res in results.items():
        cat = res['category']
        if cat not in categories or res['mae'] < categories[cat][1]:
            categories[cat] = (name, res['mae'])
    
    for cat, (name, mae) in categories.items():
        print(f"  {cat:20} {name:20} MAE: {mae:.3f}")


if __name__ == "__main__":
    main() 