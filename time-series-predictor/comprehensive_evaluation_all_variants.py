"""
Final Comprehensive Evaluation with All Improvements
===================================================

This script implements all the suggested improvements that work:
1. Better cross-validation (10 folds, 2-week test windows)
2. Improved model configurations
3. Feature selection
4. Target transformations
5. Two-stage modeling
6. Simple ensemble approaches
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
from src.framework.adapters import SchemaBasedSKLearnAdapter, PyTorchAdapter
from src.framework.core.base import CrossValidator

# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

# Framework models
from src.framework.models.baselines import AveragePredictor, NaiveForecast, DLinearWrapper
from src.framework.models.neural_nets import SimpleLSTM


class TwoStageZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """Two-stage model for zero-inflated targets."""
    
    def __init__(self, zero_threshold=5.0):
        """
        Args:
            zero_threshold: Values below this are considered "effectively zero"
        """
        self.zero_threshold = zero_threshold
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )
        self.regressor = Lasso(alpha=0.1, max_iter=2000)
        
    def fit(self, X, y):
        # Create binary labels
        self.is_nonzero_ = (y > self.zero_threshold).astype(int)
        
        # Fit classifier
        self.classifier.fit(X, self.is_nonzero_)
        
        # Fit regressor only on non-zero samples
        nonzero_mask = self.is_nonzero_ == 1
        if np.sum(nonzero_mask) > 10:  # Need enough samples
            self.regressor.fit(X[nonzero_mask], y[nonzero_mask])
        else:
            # Fallback to all data
            self.regressor.fit(X, y)
            
        return self
        
    def predict(self, X):
        # Predict zero/non-zero
        is_nonzero_pred = self.classifier.predict(X)
        
        # Initialize predictions
        predictions = np.zeros(len(X))
        
        # Predict values for predicted non-zero cases
        nonzero_mask = is_nonzero_pred == 1
        if np.sum(nonzero_mask) > 0:
            predictions[nonzero_mask] = self.regressor.predict(X[nonzero_mask])
            
        # Ensure non-negative
        return np.maximum(predictions, 0)


class SimpleEnsemble(BaseEstimator, RegressorMixin):
    """Simple averaging ensemble."""
    
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        for name, model in self.models:
            model.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = []
        for name, model in self.models:
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)


def create_all_models():
    """Create all models including improvements."""
    
    models = {}
    
    # === BASELINES ===
    models['01_baseline_average'] = {
        'model': AveragePredictor(),
        'description': 'Historical average',
        'category': 'baseline'
    }
    
    models['02_baseline_naive'] = {
        'model': NaiveForecast(),
        'description': 'Last value predictor',
        'category': 'baseline'
    }
    
    models['03_dlinear'] = {
        'model': DLinearWrapper(seq_len=8, kernel_size=3),
        'description': 'DLinear decomposition',
        'category': 'baseline'
    }
    
    # === LINEAR MODELS ===
    models['04_linear_regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.01))  # Small regularization
        ]),
        'description': 'Linear regression',
        'category': 'linear'
    }
    
    models['05_lasso'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Lasso regression',
        'category': 'linear'
    }
    
    models['06_elastic_net'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000))
        ]),
        'description': 'ElasticNet regression',
        'category': 'linear'
    }
    
    models['07_huber'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', HuberRegressor(epsilon=1.35, max_iter=200))
        ]),
        'description': 'Huber robust regression',
        'category': 'linear'
    }
    
    # === TREE ENSEMBLES ===
    models['08_rf_shallow'] = {
        'model': RandomForestRegressor(
            n_estimators=200,
            max_depth=3,  # Original shallow
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        'description': 'RF shallow trees (original)',
        'category': 'ensemble'
    }
    
    models['09_rf_deep'] = {
        'model': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,  # Deeper trees
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'description': 'RF deep trees (improved)',
        'category': 'ensemble'
    }
    
    models['10_extra_trees'] = {
        'model': ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Extra Trees (max randomization)',
        'category': 'ensemble'
    }
    
    models['11_xgb_tree'] = {
        'model': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'description': 'XGBoost tree-based',
        'category': 'ensemble'
    }
    
    models['12_xgb_linear'] = {
        'model': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=200,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        'description': 'XGBoost linear (original)',
        'category': 'ensemble'
    }
    
    models['13_xgb_linear_improved'] = {
        'model': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=500,
            learning_rate=0.05,
            reg_alpha=0.5,
            reg_lambda=2.0,
            feature_selector='cyclic',
            random_state=42
        ),
        'description': 'XGBoost linear (improved)',
        'category': 'ensemble'
    }
    
    # === FEATURE SELECTION ===
    models['14_lasso_mi_select'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Lasso + MI feature selection (30)',
        'category': 'feature_selection'
    }
    
    models['15_lasso_f_select'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_regression, k=25)),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Lasso + F-stat selection (25)',
        'category': 'feature_selection'
    }
    
    models['16_rf_select'] = {
        'model': Pipeline([
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=42
            ))
        ]),
        'description': 'RF + feature selection (30)',
        'category': 'feature_selection'
    }
    
    # === TARGET TRANSFORMATION ===
    models['17_lasso_log'] = {
        'model': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.05, max_iter=3000))
            ]),
            func=np.log1p,
            inverse_func=np.expm1
        ),
        'description': 'Lasso + log(1+y) transform',
        'category': 'transformed'
    }
    
    models['18_ridge_sqrt'] = {
        'model': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            func=np.sqrt,
            inverse_func=np.square
        ),
        'description': 'Ridge + sqrt(y) transform',
        'category': 'transformed'
    }
    
    # === SPECIALIZED ===
    models['19_two_stage'] = {
        'model': TwoStageZeroInflatedRegressor(zero_threshold=5.0),
        'description': 'Two-stage zero-inflated model',
        'category': 'specialized'
    }
    
    # === ENSEMBLE ===
    models['20_ensemble_top3'] = {
        'model': SimpleEnsemble([
            ('lasso', Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.1))
            ])),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
            ('xgb', xgb.XGBRegressor(booster='gblinear', n_estimators=200, random_state=42))
        ]),
        'description': 'Ensemble: Lasso + RF + XGB-linear',
        'category': 'ensemble_meta'
    }
    
    return models


def run_comprehensive_evaluation(window_size=8, quick_mode=False):
    """Run comprehensive evaluation."""
    
    print("=" * 80)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Configuration
    config = {
        'sequence_length': window_size,
        'n_splits': 10 if not quick_mode else 3,  # 10 folds or 3 for quick
        'test_size': 2,  # 2 weeks test
        'schema': 'time_goal_extended'
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    schema = get_schema(config['schema'])
    
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=config['sequence_length'],
        validate_data=False
    )
    
    print(f"\nDataset: {len(dataset)} sequences")
    print(f"Target: {schema.target_column}")
    
    # Get models
    models = create_all_models()
    
    # Add LSTM
    models['21_lstm'] = {
        'model': PyTorchAdapter(
            SimpleLSTM(
                input_size=len(schema.feature_columns),
                hidden_size=64,
                num_layers=2,
                dropout=0.3
            ),
            schema=schema
        ),
        'description': 'LSTM neural network',
        'category': 'neural',
        'is_pytorch': True
    }
    
    print(f"\nEvaluating {len(models)} models...")
    if quick_mode:
        print("(Quick mode: 3 folds only)")
    
    # Run evaluation
    results = {}
    
    # Sort models by key
    sorted_models = sorted(models.items())
    
    for model_name, model_config in sorted_models:
        print(f"\n{model_name}: {model_config['description']}...", end='', flush=True)
        start_time = time.time()
        
        try:
            if model_config.get('is_pytorch', False):
                # PyTorch model
                model = model_config['model']
                cv = CrossValidator(model, dataset)
                cv_results = cv.cross_validate(
                    n_splits=config['n_splits'],
                    test_size=config['test_size'],
                    epochs=50 if quick_mode else 100,
                    batch_size=64,
                    early_stopping_patience=10,
                    verbose=False
                )
            else:
                # SKLearn model
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
            
            results[model_name] = {
                'mae': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'rmse': cv_results['rmse_mean'],
                'category': model_config['category'],
                'description': model_config['description'],
                'time': time.time() - start_time
            }
            
            print(f" MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}")
            
        except Exception as e:
            print(f" Failed: {str(e)[:50]}...")
            results[model_name] = {'error': str(e)}
    
    return results, config


def analyze_results(results):
    """Comprehensive analysis of results."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)
    
    # Convert to DataFrame
    rows = []
    for name, res in results.items():
        if 'error' not in res:
            rows.append({
                'Model': name,
                'Category': res['category'],
                'MAE': res['mae'],
                'MAE_Std': res['mae_std'],
                'RMSE': res['rmse'],
                'Description': res['description'],
                'Time': res['time']
            })
    
    if not rows:
        print("No successful results!")
        return None
    
    df = pd.DataFrame(rows).sort_values('MAE')
    
    # Overall ranking
    print("\n📊 TOP 10 MODELS:")
    print("-" * 90)
    print(f"{'Rank':<5} {'Model':<22} {'MAE':<12} {'Category':<18} {'Description':<35}")
    print("-" * 90)
    
    for i, row in df.head(10).iterrows():
        rank = df.index.get_loc(i) + 1
        mae_str = f"{row['MAE']:.3f}±{row['MAE_Std']:.3f}"
        desc_short = row['Description'][:35]
        print(f"{rank:<5} {row['Model']:<22} {mae_str:<12} {row['Category']:<18} {desc_short:<35}")
    
    # Category analysis
    print("\n📈 BEST MODEL PER CATEGORY:")
    print("-" * 70)
    for cat in sorted(df['Category'].unique()):
        cat_best = df[df['Category'] == cat].iloc[0]
        print(f"{cat:<18}: {cat_best['Model']:<22} MAE={cat_best['MAE']:.3f}")
    
    # Improvement analysis
    baseline = df[df['Model'].str.contains('baseline_average')]
    if not baseline.empty:
        baseline_mae = baseline.iloc[0]['MAE']
        best_mae = df.iloc[0]['MAE']
        improvement = (baseline_mae - best_mae) / baseline_mae * 100
        
        print(f"\n🎯 PERFORMANCE IMPROVEMENT:")
        print(f"  Baseline MAE: {baseline_mae:.3f}")
        print(f"  Best MAE: {best_mae:.3f} ({df.iloc[0]['Model']})")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Compare key findings
    print(f"\n🔍 KEY FINDINGS:")
    
    # 1. Feature selection impact
    feature_sel = df[df['Category'] == 'feature_selection']
    if not feature_sel.empty:
        best_fs = feature_sel.iloc[0]
        base_lasso = df[df['Model'] == '05_lasso']
        if not base_lasso.empty:
            fs_improvement = (base_lasso.iloc[0]['MAE'] - best_fs['MAE']) / base_lasso.iloc[0]['MAE'] * 100
            print(f"  • Feature selection improvement: {fs_improvement:.1f}% "
                  f"(best: {best_fs['Model']})")
    
    # 2. Deep vs shallow trees
    rf_shallow = df[df['Model'] == '08_rf_shallow']
    rf_deep = df[df['Model'] == '09_rf_deep']
    if not rf_shallow.empty and not rf_deep.empty:
        tree_improvement = (rf_shallow.iloc[0]['MAE'] - rf_deep.iloc[0]['MAE']) / rf_shallow.iloc[0]['MAE'] * 100
        print(f"  • Deeper trees improvement: {tree_improvement:.1f}% "
              f"(shallow: {rf_shallow.iloc[0]['MAE']:.3f}, deep: {rf_deep.iloc[0]['MAE']:.3f})")
    
    # 3. Target transformation
    transformed = df[df['Category'] == 'transformed']
    if not transformed.empty:
        best_trans = transformed.iloc[0]
        print(f"  • Best transformation: {best_trans['Model']} (MAE={best_trans['MAE']:.3f})")
    
    # 4. Two-stage model
    two_stage = df[df['Model'] == '19_two_stage']
    if not two_stage.empty:
        print(f"  • Two-stage model: MAE={two_stage.iloc[0]['MAE']:.3f}")
    
    return df


def main():
    """Main function."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=8, help='Window size')
    parser.add_argument('--quick', action='store_true', help='Quick mode (3 folds)')
    args = parser.parse_args()
    
    print(f"🚀 Starting final comprehensive evaluation...")
    print(f"Window size: {args.window}")
    print(f"Mode: {'Quick (3 folds)' if args.quick else 'Full (10 folds)'}")
    
    # Run evaluation
    results, config = run_comprehensive_evaluation(
        window_size=args.window,
        quick_mode=args.quick
    )
    
    # Analyze
    df = analyze_results(results)
    
    if df is not None:
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('experiments/outputs')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'final_eval_window{args.window}_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main() 