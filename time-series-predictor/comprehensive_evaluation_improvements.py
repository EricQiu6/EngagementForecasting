"""
Comprehensive Evaluation of Time Series Prediction Algorithms - Improved Version
===============================================================================

This script implements improved evaluation methodology with:
1. Better cross-validation (more folds, larger test windows)
2. Target transformation handling
3. Feature selection
4. Ensemble methods
5. Two-stage modeling for zero-inflation
6. Optimized hyperparameters

Models evaluated:
1. Trivial baselines (average, last value)
2. DLinear (temporal decomposition)
3. Linear models (with/without regularization)
4. Tree ensembles (optimized Random Forest, XGBoost)
5. Neural networks (MLP, LSTM)
6. Ensemble models
7. Two-stage models for zero-inflation
8. Feature selection variants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Schema-based framework imports
from src.framework.core.schema import get_schema, DataSchema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters import SchemaBasedSKLearnAdapter, PyTorchAdapter
from src.framework.core.base import CrossValidator, MetricsCalculator

# Model imports
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
import xgboost as xgb

# Framework models
from src.framework.models.baselines import AveragePredictor, NaiveForecast, DLinearWrapper
from src.framework.models.neural_nets import SimpleLSTM

# For two-stage model
from sklearn.ensemble import RandomForestClassifier


class TargetTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for target variable."""
    
    def __init__(self, method='log1p'):
        self.method = method
        self.scaler = None
        
    def fit(self, y):
        if self.method == 'log1p':
            # No fitting needed for log1p
            pass
        elif self.method == 'sqrt':
            # No fitting needed for sqrt
            pass
        elif self.method == 'power':
            self.scaler = PowerTransformer()
            self.scaler.fit(y.reshape(-1, 1))
        return self
        
    def transform(self, y):
        if self.method == 'log1p':
            return np.log1p(y)
        elif self.method == 'sqrt':
            return np.sqrt(y)
        elif self.method == 'power':
            return self.scaler.transform(y.reshape(-1, 1)).flatten()
        else:
            return y
            
    def inverse_transform(self, y):
        if self.method == 'log1p':
            return np.expm1(y)
        elif self.method == 'sqrt':
            return np.square(y)
        elif self.method == 'power':
            return self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        else:
            return y


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Two-stage model for zero-inflated targets."""
    
    def __init__(self, classifier=None, regressor=None, zero_threshold=1.0):
        """
        Args:
            classifier: Model to predict zero vs non-zero
            regressor: Model to predict non-zero values
            zero_threshold: Values below this are considered "zero"
        """
        self.classifier = classifier or RandomForestClassifier(n_estimators=100, max_depth=5)
        self.regressor = regressor or Lasso(alpha=0.1)
        self.zero_threshold = zero_threshold
        
    def fit(self, X, y):
        # Create binary labels
        is_nonzero = (y > self.zero_threshold).astype(int)
        
        # Fit classifier
        self.classifier.fit(X, is_nonzero)
        
        # Fit regressor only on non-zero samples
        nonzero_mask = is_nonzero == 1
        if np.sum(nonzero_mask) > 0:
            self.regressor.fit(X[nonzero_mask], y[nonzero_mask])
        else:
            # Fallback if no non-zero samples
            self.regressor.fit(X, y)
            
        return self
        
    def predict(self, X):
        # Predict zero/non-zero
        is_nonzero = self.classifier.predict(X)
        
        # Initialize predictions
        predictions = np.zeros(len(X))
        
        # Predict values for non-zero cases
        nonzero_mask = is_nonzero == 1
        if np.sum(nonzero_mask) > 0:
            predictions[nonzero_mask] = self.regressor.predict(X[nonzero_mask])
            
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
        
    def get_params(self, deep=True):
        return {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'zero_threshold': self.zero_threshold
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_improved_models():
    """Create all models with improved configurations."""
    
    models = {}
    
    # Flag to skip problematic models for now
    skip_problematic = True
    
    # 1. Baselines
    models['01_average_predictor'] = {
        'model': AveragePredictor(),
        'description': 'Historical average baseline',
        'category': 'baseline'
    }
    
    models['02_last_value'] = {
        'model': NaiveForecast(),
        'description': 'Last value (naive) baseline',
        'category': 'baseline'
    }
    
    # 3. DLinear
    models['03_dlinear'] = {
        'model': DLinearWrapper(seq_len=5, kernel_size=3),
        'description': 'DLinear temporal decomposition',
        'category': 'time_series'
    }
    
    # 4. Linear Regression (with better preprocessing)
    models['04_linear_regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'description': 'Linear regression with scaling',
        'category': 'linear'
    }
    
    # 5. MLP (optimized)
    models['05_mlp'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42
            ))
        ]),
        'description': 'Optimized MLP with 3 layers',
        'category': 'neural'
    }
    
    # 6. LSTM - will be added separately due to PyTorch adapter needs
    
    # 7. Random Forest (improved)
    models['07_rf_improved'] = {
        'model': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,  # Deeper than before
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Random Forest with deeper trees',
        'category': 'ensemble'
    }
    
    # 8. XGBoost (standard)
    models['08_xgboost'] = {
        'model': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'description': 'XGBoost gradient boosting',
        'category': 'ensemble'
    }
    
    # 9. Ensemble model
    models['09_ensemble'] = {
        'model': VotingRegressor([
            ('lasso', Lasso(alpha=0.1)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
            ('xgb_linear', xgb.XGBRegressor(booster='gblinear', n_estimators=200, random_state=42))
        ]),
        'description': 'Ensemble of Lasso + RF + XGBoost-linear',
        'category': 'ensemble'
    }
    
    # 10. Two-stage model
    models['10_two_stage'] = {
        'model': TwoStageRegressor(
            classifier=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            regressor=Lasso(alpha=0.1),
            zero_threshold=5.0  # Consider < 5 minutes as "effectively zero"
        ),
        'description': 'Two-stage model for zero-inflation',
        'category': 'specialized'
    }
    
    # 11. More robust Random Forest
    models['11_rf_extra_robust'] = {
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
        'description': 'Extra Trees with more randomization',
        'category': 'ensemble'
    }
    
    # 12. More robust XGBoost (linear booster)
    models['12_xgb_linear_robust'] = {
        'model': xgb.XGBRegressor(
            booster='gblinear',
            n_estimators=500,
            learning_rate=0.05,
            reg_alpha=0.5,
            reg_lambda=2.0,
            feature_selector='cyclic',
            random_state=42
        ),
        'description': 'XGBoost linear with strong regularization',
        'category': 'ensemble'
    }
    
    # 13+ Feature selection versions
    
    # 13. Lasso with feature selection
    models['13_lasso_selected'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', Lasso(alpha=0.1, max_iter=2000))
        ]),
        'description': 'Lasso with top 30 features (mutual info)',
        'category': 'feature_selection'
    }
    
    # 14. Random Forest with feature selection
    models['14_rf_selected'] = {
        'model': Pipeline([
            ('selector', SelectKBest(f_regression, k=25)),
            ('model', RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=42
            ))
        ]),
        'description': 'Random Forest with top 25 features (F-statistic)',
        'category': 'feature_selection'
    }
    
    # 15. XGBoost with feature selection
    models['15_xgb_selected'] = {
        'model': Pipeline([
            ('selector', SelectKBest(mutual_info_regression, k=30)),
            ('model', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            ))
        ]),
        'description': 'XGBoost with top 30 features',
        'category': 'feature_selection'
    }
    
    # 16. Ridge with target transformation
    models['16_ridge_transformed'] = {
        'model': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            transformer=TargetTransformer(method='log1p')
        ),
        'description': 'Ridge with log-transformed target',
        'category': 'transformed'
    }
    
    # 17. Lasso with target transformation
    models['17_lasso_transformed'] = {
        'model': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.05, max_iter=3000))
            ]),
            transformer=TargetTransformer(method='sqrt')
        ),
        'description': 'Lasso with sqrt-transformed target',
        'category': 'transformed'
    }
    
    return models


def run_improved_evaluation(schema_name='time_goal_extended', window_size=8):
    """Run evaluation with improved methodology."""
    
    print("=" * 80)
    print("COMPREHENSIVE TIME SERIES EVALUATION - IMPROVED VERSION")
    print("=" * 80)
    
    # Configuration with improvements
    evaluation_config = {
        'sequence_length': window_size,  # Configurable window
        'n_splits': 10,  # More folds as suggested
        'test_size': 2,  # 2 weeks test as suggested
        'min_samples_per_student': 10,
        'validation_strategy': 'time_series_cv'
    }
    
    print(f"\nImproved Evaluation Configuration:")
    for key, value in evaluation_config.items():
        print(f"  {key}: {value}")
    
    # Setup data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    schema = get_schema(schema_name)
    
    print(f"\n🎯 Target variable: {schema.target_column}")
    
    # Create dataset
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=evaluation_config['sequence_length'],
        validate_data=False
    )
    
    print(f"Dataset created: {len(dataset)} sequences")
    
    # Get models
    models = create_improved_models()
    
    # Add LSTM separately (needs PyTorch adapter)
    models['06_lstm'] = {
        'model': PyTorchAdapter(
            SimpleLSTM(
                input_size=len(schema.feature_columns),
                hidden_size=64,
                num_layers=2,
                dropout=0.3
            ),
            schema=schema
        ),
        'description': 'LSTM with 2 layers',
        'category': 'neural',
        'is_pytorch': True
    }
    
    print(f"\nModels to evaluate: {len(models)}")
    
    # Sort models by key for consistent ordering
    sorted_models = sorted(models.items(), key=lambda x: x[0])
    
    for name, config in sorted_models:
        print(f"  {name}: {config['description']} ({config['category']})")
    
    # Run evaluation
    results = {}
    
    print(f"\n" + "=" * 80)
    print("RUNNING EVALUATIONS")
    print("=" * 80)
    
    for model_name, model_config in sorted_models:
        print(f"\n🔄 Evaluating {model_name}: {model_config['description']}...")
        start_time = time.time()
        
        try:
            if model_config.get('is_pytorch', False):
                # PyTorch model
                model = model_config['model']
                cv = CrossValidator(model, dataset)
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size'],
                    epochs=100,
                    batch_size=64,
                    early_stopping_patience=10,
                    verbose=False
                )
            else:
                # SKLearn model
                model = SchemaBasedSKLearnAdapter(
                    sklearn_model=model_config['model'],
                    schema=schema,
                    lag_window=evaluation_config['sequence_length']
                )
                
                cv = CrossValidator(model, dataset)
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size']
                )
            
            # Store results
            results[model_name] = {
                **cv_results,
                'category': model_config['category'],
                'description': model_config['description'],
                'training_time': time.time() - start_time
            }
            
            print(f"✅ {model_name}: MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}, "
                  f"RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}, "
                  f"Time={results[model_name]['training_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'error': str(e),
                'category': model_config['category'],
                'description': model_config['description']
            }
    
    return results, evaluation_config


# Import for transformed target regression
from sklearn.compose import TransformedTargetRegressor


def analyze_improved_results(results, config):
    """Analyze and visualize improved results."""
    
    print(f"\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful results to analyze!")
        return None
    
    # Create results DataFrame
    results_data = []
    for name, result in successful_results.items():
        results_data.append({
            'Model': name,
            'Category': result['category'],
            'MAE': result['mae_mean'],
            'MAE_Std': result['mae_std'],
            'RMSE': result['rmse_mean'],
            'RMSE_Std': result['rmse_std'],
            'SMAPE': result['smape_mean'],
            'Time': result['training_time']
        })
    
    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('MAE')
    
    # Print top 10 models
    print(f"\n📊 TOP 10 MODELS (by MAE):")
    print("-" * 90)
    print(f"{'Rank':<5} {'Model':<25} {'Category':<15} {'MAE':<15} {'RMSE':<15} {'Time(s)':<10}")
    print("-" * 90)
    
    for i, row in df_results.head(10).iterrows():
        mae_str = f"{row['MAE']:.3f}±{row['MAE_Std']:.3f}"
        rmse_str = f"{row['RMSE']:.3f}±{row['RMSE_Std']:.3f}"
        print(f"{df_results.index.get_loc(i)+1:<5} {row['Model']:<25} {row['Category']:<15} "
              f"{mae_str:<15} {rmse_str:<15} {row['Time']:.1f}")
    
    # Category comparison
    print(f"\n📈 BEST MODEL PER CATEGORY:")
    best_per_category = df_results.groupby('Category').first()
    for cat, row in best_per_category.iterrows():
        print(f"  {cat:<15}: {row['Model']:<25} MAE={row['MAE']:.3f}")
    
    # Improvement analysis
    baseline_mae = df_results[df_results['Model'] == '01_average_predictor']['MAE'].values[0]
    best_mae = df_results.iloc[0]['MAE']
    improvement = (baseline_mae - best_mae) / baseline_mae * 100
    
    print(f"\n🎯 IMPROVEMENT OVER BASELINE:")
    print(f"  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Best MAE: {best_mae:.3f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    return df_results


def main():
    """Main evaluation function."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Improved comprehensive evaluation')
    parser.add_argument('--window', type=int, default=8,
                        help='Window size for sequences (default: 8)')
    parser.add_argument('--schema', type=str, default='time_goal_extended',
                        help='Schema name to use')
    args = parser.parse_args()
    
    print(f"🚀 Starting improved comprehensive evaluation...")
    print(f"Window size: {args.window}")
    print(f"Schema: {args.schema}")
    
    try:
        # Run evaluation
        results, config = run_improved_evaluation(
            schema_name=args.schema,
            window_size=args.window
        )
        
        # Analyze results
        df_results = analyze_improved_results(results, config)
        
        # Save results
        if df_results is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('experiments/outputs')
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            results_path = output_dir / f'improved_evaluation_results_{timestamp}.json'
            with open(results_path, 'w') as f:
                json_results = {k: v for k, v in results.items() if 'error' not in v}
                json.dump({
                    'timestamp': timestamp,
                    'config': config,
                    'results': json_results
                }, f, indent=2, default=str)
            
            # Save summary CSV
            summary_path = output_dir / f'improved_evaluation_summary_{timestamp}.csv'
            df_results.to_csv(summary_path, index=False)
            
            print(f"\n📁 Results saved:")
            print(f"  Detailed: {results_path}")
            print(f"  Summary: {summary_path}")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 