"""
Comprehensive Evaluation of Time Series Prediction Algorithms
=============================================================

This script evaluates multiple algorithms using the new schema-based framework
with realistic evaluation settings including sufficient windows, appropriate
number of folds, and settings that allow for reasonable performance potential.

Algorithms evaluated:
1. Trivial baselines (averaging, last value)
2. Classical ML (Random Forest, SVM, etc.)
3. Simple neural networks (MLP, LSTM)
4. Proposed student ability model

Uses proper time series cross-validation with realistic settings.
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
from src.framework.adapters import SchemaBasedSKLearnAdapter
from src.framework.core.base import CrossValidator, MetricsCalculator

# Mixed effects imports (optional)
try:
    from src.framework.adapters.mixed_effects_sklearn_adapter import (
        MixedEffectsSKLearnWrapper,
        MixedEffectsSKLearnAdapter
    )
    HAS_MIXED_EFFECTS = True
except ImportError:
    HAS_MIXED_EFFECTS = False
    print("Mixed effects models not available")

# Algorithm imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor

# Import existing baseline models from our framework
from src.framework.models.baselines import AveragePredictor, NaiveForecast, LinearTrend, DLinearWrapper
from src.framework.models.neural_nets import SimpleLSTM
from src.framework.adapters.pytorch_adapter import PyTorchAdapter

# Import the new goal-based predictor
from src.framework.models.goal_based_predictor import GoalBasedPredictor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")


class TrivialBaselines:
    """Trivial baseline algorithms for comparison."""
    
    @staticmethod
    def create_averaging_model():
        """Simple averaging baseline."""
        return DummyRegressor(strategy='mean')
    
    @staticmethod  
    def create_last_value_model():
        """Last value baseline."""
        class LastValuePredictor:
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                # For sequence data, use the last value of the target feature
                if len(X.shape) == 2 and X.shape[1] > 1:
                    # Assume last column is the target lag
                    return X[:, -1]
                else:
                    # Fallback to mean
                    return np.full(len(X), np.mean(X) if len(X) > 0 else 0.0)
        
        return LastValuePredictor()


# Remove duplicate StudentAbilityModel - we'll use the existing one from the models directory


def create_algorithm_configs(schema=None):
    """Create configurations for verified algorithms only."""
    
    algorithms = {
        # # Verified baselines
        # 'average_predictor': {
        #     'model': AveragePredictor(),
        #     'description': 'Historical average predictor',
        #     'category': 'baseline'
        # },
        
        # 'naive_forecast': {
        #     'model': NaiveForecast(),
        #     'description': 'Last value predictor',
        #     'category': 'baseline'
        # },
        
        # Goal-based predictors
        'goal_based_predictor': {
            'model': GoalBasedPredictor(random_state=42),
            'description': 'Goal-based predictor (random first 8, then median)',
            'category': 'baseline'
        },
        
        # 'adaptive_goal_predictor': {
        #     'model': AdaptiveGoalPredictor(random_state=42),
        #     'description': 'Adaptive goal-based predictor',
        #     'category': 'baseline'
        # },
        
        # Classical ML
        # 'linear_regression': {
        #     'model': LinearRegression(),
        #     'description': 'Simple linear regression',
        #     'category': 'classical'
        # },
        
        # 'ridge': {
        #     'model': Ridge(alpha=1.0),
        #     'description': 'Ridge regression',
        #     'category': 'classical'
        # },
        
        'lasso': {
            'model': Lasso(alpha=0.1),
            'description': 'Lasso regression',
            'category': 'classical'
        },
        
        # 'random_forest': {
        #     'model': RandomForestRegressor(
        #         n_estimators=100,
        #         max_depth=10,
        #         random_state=42
        #     ),
        #     'description': 'Random Forest',
        #     'category': 'ensemble'
        # },
        
        # Time series specific
        # 'dlinear': {
        #     'model': DLinearWrapper(seq_len=15, kernel_size=3),
        #     'description': 'DLinear decomposition model',
        #     'category': 'time_series'
        # },
        
        # Neural networks with SKLearnAdapter
        # 'mlp': {
        #     'model': MLPRegressor(
        #         hidden_layer_sizes=(64, 32),
        #         activation='relu',
        #         solver='adam',
        #         alpha=0.01,
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.1,
        #         n_iter_no_change=10,
        #         random_state=42
        #     ),
        #     'description': 'Multi-Layer Perceptron',
        #     'category': 'neural'
        # },
    }
    
    # Add XGBoost if available
    # if HAS_XGBOOST:
    #     algorithms['xgboost'] = {
    #         'model': xgb.XGBRegressor(
    #             n_estimators=100,
    #             max_depth=6,
    #             learning_rate=0.1,
    #             random_state=42
    #         ),
    #         'description': 'XGBoost Gradient Boosting',
    #         'category': 'ensemble'
    #     }
    
    # Add Mixed Effects models
    # if HAS_MIXED_EFFECTS:
    #     # Comment out for now - the current wrapper has a different interface
    #     """
    #     # Vanilla Mixed Effects
    #     algorithms['mixed_effects_vanilla'] = {
    #         'model': MixedEffectsSKLearnWrapper(
    #             model_type='vanilla',
    #             re_formula='1'
    #         ),
    #         'description': 'Mixed Effects (vanilla)',
    #         'category': 'mixed_effects',
    #         'requires_student_id': True
    #     }
    #     
    #     # LASSO + Mixed Effects
    #     algorithms['mixed_effects_lasso'] = {
    #         'model': MixedEffectsSKLearnWrapper(
    #             model_type='lasso',
    #             re_formula='1',
    #             lasso_alpha=None  # Auto-select via CV
    #         ),
    #         'description': 'Mixed Effects (LASSO)',
    #         'category': 'mixed_effects',
    #         'requires_student_id': True
    #     }
    #     
    #     # Ensemble Mixed Effects
    #     algorithms['mixed_effects_ensemble'] = {
    #         'model': MixedEffectsSKLearnWrapper(
    #             model_type='ensemble',
    #             re_formula='1'
    #         ),
    #         'description': 'Mixed Effects (ensemble)',
    #         'category': 'mixed_effects',
    #         'requires_student_id': True
    #     }
    #     """
    #     
    #     # Use the actual interface for now
    #     # Get target column from schema if available
    #     target_col = schema.target_column if schema else 'minutes_per_week'
    #     
    #     algorithms['mixed_effects_baseline'] = {
    #         'model': MixedEffectsSKLearnWrapper(
    #             target_col=target_col,
    #             n_lags=5,
    #             use_simple_baseline=True
    #         ),
    #         'description': 'Mixed Effects (baseline)',
    #         'category': 'mixed_effects',
    #         'requires_student_id': False  # This version doesn't need explicit student IDs
    #     }
    #     
    #     print("✅ Mixed effects models added to evaluation")
    # else:
    #     print("⚠️  Mixed effects models not available")
    
    # Note: LSTM requires PyTorchAdapter, so we'll add it separately
    # It needs different handling in the evaluation loop
    
    return algorithms


def setup_evaluation_data(schema_name=None):
    """Setup data with realistic evaluation settings."""
    
    # Check available data files
    data_files = {
        'student_week_rolling': '../data-analysis/student_week_aggregations_rolling_new.csv'
    }
    
    available_data = {}
    for name, path in data_files.items():
        if Path(path).exists():
            available_data[name] = path
            print(f"✅ Found {name} data: {path}")
        else:
            print(f"❌ Missing {name} data: {path}")
    
    if not available_data:
        raise FileNotFoundError("No evaluation data found")
    
    # Use the first available dataset
    data_name, data_path = list(available_data.items())[0]
    
    # Load and analyze data
    df = pd.read_csv(data_path)
    print(f"\nDataset: {data_name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Use provided schema or determine appropriate one
    if schema_name:
        print(f"Using provided schema: {schema_name}")
    elif 'anon_student_id' in df.columns and 'week_id' in df.columns and 'avg_proficiency' in df.columns:
        # Check if we have extended features for student ability models
        extended_features = ['student_ability', 'student_learning_rate', 'week_difficulty']
        has_extended = all(col in df.columns for col in extended_features)
        
        if has_extended:
            schema_name = 'extended'
            print(f"Using extended schema (includes student ability features)")
        else:
            schema_name = 'student_week'
            print(f"Using student_week schema")
    else:
        # Try to infer schema from columns
        possible_student_cols = [col for col in df.columns if 'student' in col.lower() or 'id' in col.lower()]
        possible_time_cols = [col for col in df.columns if 'week' in col.lower() or 'time' in col.lower()]
        possible_target_cols = [col for col in df.columns if 'profic' in col.lower() or 'score' in col.lower()]
        
        if possible_student_cols and possible_time_cols and possible_target_cols:
            # Create custom schema
            schema = DataSchema(
                student_column=possible_student_cols[0],
                time_column=possible_time_cols[0],
                target_column=possible_target_cols[0],
                feature_columns=[possible_time_cols[0], possible_target_cols[0]]
            )
            schema_name = 'custom'
            print(f"Using custom schema: {schema.to_config()}")
        else:
            raise ValueError(f"Cannot determine schema for data with columns: {df.columns}")
    
    return data_path, schema_name


def add_student_ability_models_if_possible(algorithms, schema, dataset):
    """
    Add student ability models if the data supports them.
    """
    # Check if we have the required features for student ability model
    required_features = ['student_ability', 'student_learning_rate', 'week_difficulty']
    available_features = set(schema.feature_columns)
    
    missing_features = [f for f in required_features if f not in available_features]
    
    if missing_features:
        print(f"\n⚠️  Cannot use student ability models - missing features: {missing_features}")
        print(f"Available features: {schema.feature_columns}")
        print("Skipping student ability models.")
        return algorithms
    
    print(f"\n✅ Data supports student ability models - adding them to evaluation")
    
    try:
        # Try to import the PyTorch adapter
        from src.framework.adapters.pytorch_adapter_v2 import create_student_ability_model
        
        # Add student ability models
        algorithms['student_ability_linear'] = {
            'model': create_student_ability_model(
                schema=schema, 
                model_type='linear',
                epochs=50,  # Reduced for faster evaluation
                learning_rate=0.01
            ),
            'description': 'Student ability linear model (PyTorch)',
            'category': 'proposed'
        }
        
        algorithms['student_ability_neural'] = {
            'model': create_student_ability_model(
                schema=schema,
                model_type='neural',
                epochs=50,
                learning_rate=0.01,
                hidden_size=32
            ),
            'description': 'Student ability neural model (PyTorch)',
            'category': 'proposed'
        }
        
        print("Added student ability models successfully")
        
    except ImportError as e:
        print(f"Could not import student ability models: {e}")
        print("Skipping student ability models.")
    except Exception as e:
        print(f"Error creating student ability models: {e}")
        print("Skipping student ability models.")
    
    return algorithms


def run_comprehensive_evaluation(schema_name=None):
    """Run comprehensive evaluation with realistic settings."""
    
    print("=" * 80)
    print("COMPREHENSIVE TIME SERIES ALGORITHM EVALUATION")
    print("=" * 80)
    
    # Setup data
    data_path, schema_name = setup_evaluation_data(schema_name)
    
    if isinstance(schema_name, str):
        # Get predefined schema by name
        schema = get_schema(schema_name)
    else:
        # Use custom schema created in setup
        schema = schema_name  # This would be the DataSchema object
    
    # Realistic evaluation settings
    evaluation_config = {
        'sequence_length': 15,  # Extended history window
        'n_splits': 5,          # Appropriate number of folds
        'test_size': 1,         # Test on 1 week
        'min_samples_per_student': 10,  # Minimum data per student
        'validation_strategy': 'time_series_cv'
    }
    
    print(f"\nEvaluation Configuration:")
    for key, value in evaluation_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Target variable: {schema.target_column}")
    
    # Create dataset with realistic settings
    print(f"\nCreating dataset...")
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=evaluation_config['sequence_length'],
        validate_data=False  # Allow validation warnings without failing
    )
    
    print(f"Dataset created: {len(dataset)} sequences")
    
    # Get algorithms
    algorithms = create_algorithm_configs(schema)
    
    # Add LSTM back to evaluation
    algorithms['lstm'] = {
        'model': PyTorchAdapter(
            SimpleLSTM(
                input_size=len(schema.feature_columns),
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            ),
            schema=schema
        ),
        'description': 'LSTM with temporal modeling',
        'category': 'neural',
        'is_pytorch': True
    }
    
    print(f"\nAlgorithms to evaluate: {len(algorithms)}")
    for name, config in algorithms.items():
        print(f"  {name}: {config['description']} ({config['category']})")
    
    # Run evaluation
    results = {}
    
    print(f"\n" + "=" * 80)
    print("RUNNING EVALUATIONS")
    print("=" * 80)
    
    for algo_name, algo_config in algorithms.items():
        print(f"\n🔄 Evaluating {algo_name}...")
        start_time = time.time()
        
        try:
            # Check if this is a PyTorch model
            if algo_config.get('is_pytorch', False):
                # PyTorch model - already has adapter
                model = algo_config['model']
                cv = CrossValidator(model, dataset)
                # Use fewer epochs for faster evaluation
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size'],
                    epochs=50,
                    batch_size=32,
                    early_stopping_patience=5,
                    verbose=False
                )
            else:
                # Check if this is a mixed effects model
                if algo_config.get('requires_student_id', False):
                    # Mixed effects model - needs special adapter
                    from src.framework.adapters.mixed_effects_sklearn_adapter import MixedEffectsSKLearnAdapter
                    model = MixedEffectsSKLearnAdapter(
                        mixed_effects_model=algo_config['model'],
                        schema=schema,
                        lag_window=evaluation_config['sequence_length']
                    )
                else:
                    # Create schema-based adapter for sklearn models
                    model = SchemaBasedSKLearnAdapter(
                        sklearn_model=algo_config['model'],
                        schema=schema,
                        lag_window=evaluation_config['sequence_length']
                    )
                
                # Run cross-validation
                cv = CrossValidator(model, dataset)
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size']
                )
            
            # Store results
            results[algo_name] = {
                **cv_results,
                'category': algo_config['category'],
                'description': algo_config['description'],
                'training_time': time.time() - start_time
            }
            
            print(f"✅ {algo_name}: MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}, "
                  f"RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}, "
                  f"Time={results[algo_name]['training_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ {algo_name} failed: {str(e)}")
            results[algo_name] = {
                'error': str(e),
                'category': algo_config['category'],
                'description': algo_config['description']
            }
    
    return results, evaluation_config


def analyze_results(results, config):
    """Analyze results and create summary."""
    
    print(f"\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    
    if failed_results:
        print(f"\n❌ Failed algorithms ({len(failed_results)}):")
        for name, result in failed_results.items():
            print(f"  {name}: {result['error']}")
    
    if not successful_results:
        print("No successful results to analyze!")
        return None
    
    print(f"\n✅ Successful algorithms ({len(successful_results)}):")
    
    # Create results DataFrame
    results_data = []
    for name, result in successful_results.items():
        results_data.append({
            'Algorithm': name,
            'Category': result['category'],
            'MAE_Mean': result['mae_mean'],
            'MAE_Std': result['mae_std'],
            'RMSE_Mean': result['rmse_mean'],
            'RMSE_Std': result['rmse_std'],
            'SMAPE_Mean': result['smape_mean'],
            'SMAPE_Std': result['smape_std'],
            'Training_Time': result['training_time'],
            'Description': result['description']
        })
    
    df_results = pd.DataFrame(results_data)
    
    # Sort by MAE performance
    df_results = df_results.sort_values('MAE_Mean')
    
    # Print ranking
    print(f"\n📊 ALGORITHM RANKING (by MAE):")
    print("-" * 100)
    print(f"{'Rank':<4} {'Algorithm':<20} {'Category':<10} {'MAE':<12} {'RMSE':<12} {'SMAPE':<12} {'Time(s)':<8}")
    print("-" * 100)
    
    for i, row in df_results.iterrows():
        mae_str = f"{row['MAE_Mean']:.3f}±{row['MAE_Std']:.3f}"
        rmse_str = f"{row['RMSE_Mean']:.3f}±{row['RMSE_Std']:.3f}"
        smape_str = f"{row['SMAPE_Mean']:.1f}±{row['SMAPE_Std']:.1f}%"
        print(f"{df_results.index.get_loc(i)+1:<4} {row['Algorithm']:<20} {row['Category']:<10} "
              f"{mae_str:<12} {rmse_str:<12} {smape_str:<12} {row['Training_Time']:.1f}<8")
    
    # Category analysis
    print(f"\n📈 PERFORMANCE BY CATEGORY:")
    category_stats = df_results.groupby('Category').agg({
        'MAE_Mean': ['mean', 'min', 'max'],
        'RMSE_Mean': ['mean', 'min', 'max'],
        'Training_Time': ['mean', 'sum']
    }).round(3)
    
    print(category_stats)
    
    # Save results
    save_results(results, df_results, config)
    
    return df_results


def save_results(results, df_results, config):
    """Save evaluation results to files."""
    
    output_dir = Path('experiments/outputs')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results as JSON
    results_path = output_dir / f'evaluation_results_{timestamp}.json'
    try:
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.floating, np.float32, np.float64, np.int32, np.int64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.astype(float).tolist()
                elif isinstance(obj, list):
                    return [convert_numpy(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                else:
                    return obj
            
            json_results = convert_numpy(results)
            
            json.dump({
                'timestamp': timestamp,
                'config': config,
                'results': json_results
            }, f, indent=2, default=str)  # Use default=str as fallback
    except Exception as e:
        print(f"Warning: Could not save JSON results: {e}")
        # Still save CSV which is more important
    
    # Save summary as CSV
    summary_path = output_dir / f'evaluation_summary_{timestamp}.csv'
    df_results.to_csv(summary_path, index=False)
    
    print(f"📁 Results saved:")
    print(f"  Detailed: {results_path}")
    print(f"  Summary: {summary_path}")


def main():
    """Main evaluation function."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive time series algorithm evaluation')
    parser.add_argument('--schema', type=str, default=None,
                        help='Schema name to use (e.g., legacy, student_week, extended, time_goal, time_goal_extended)')
    parser.add_argument('--window-analysis', action='store_true',
                        help='Run window size analysis (5 to 30 weeks)')
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive time series algorithm evaluation...")
    print("Using the new schema-based framework with realistic settings.")
    
    try:
        if args.window_analysis:
            # Run window size analysis
            print("\n📊 Running window size analysis...")
            run_window_size_analysis(args.schema)
        else:
            # Run standard evaluation
            results, config = run_comprehensive_evaluation(args.schema)
            
            # Analyze results
            df_results = analyze_results(results, config)
            
            print(f"\n" + "=" * 80)
            print("EVALUATION COMPLETE!")
            print("=" * 80)
            
            if df_results is not None and len(df_results) > 0:
                best_algorithm = df_results.iloc[0]
                print(f"\n🏆 BEST PERFORMING ALGORITHM:")
                print(f"  Name: {best_algorithm['Algorithm']}")
                print(f"  Category: {best_algorithm['Category']}")
                print(f"  MAE: {best_algorithm['MAE_Mean']:.3f} ± {best_algorithm['MAE_Std']:.3f}")
                print(f"  RMSE: {best_algorithm['RMSE_Mean']:.3f} ± {best_algorithm['RMSE_Std']:.3f}")
                print(f"  Training Time: {best_algorithm['Training_Time']:.1f}s")
                print(f"  Description: {best_algorithm['Description']}")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_window_size_analysis(schema_name=None):
    """
    Run comprehensive window size analysis from 5 to 30 weeks.
    Analyzes model performance, consistency, learning curves, and stability.
    """
    from scipy import stats
    from sklearn.metrics import r2_score
    
    print("=" * 80)
    print("WINDOW SIZE ANALYSIS")
    print("=" * 80)
    
    # Setup data
    data_path, schema_name = setup_evaluation_data(schema_name)
    
    if isinstance(schema_name, str):
        schema = get_schema(schema_name)
    else:
        schema = schema_name
    
    # Window sizes to test
    window_sizes = [5, 10, 15, 20, 25]  # Removed 1 to avoid numerical issues
    
    # Initialize storage for results
    all_results = []
    feature_importance_results = []
    
    # Get algorithms once
    algorithms_template = create_algorithm_configs(schema)
    
    print(f"\nTesting {len(window_sizes)} window sizes with {len(algorithms_template)} algorithms")
    print(f"Total evaluations: {len(window_sizes) * len(algorithms_template)}")
    
    # Run evaluation for each window size
    for window_idx, window_size in enumerate(window_sizes):
        print(f"\n{'='*60}")
        print(f"Window Size: {window_size} ({window_idx+1}/{len(window_sizes)})")
        print(f"{'='*60}")
        
        # Create dataset with current window size
        try:
            dataset = SchemaBasedTimeSeriesDataset(
                data_path=data_path,
                schema=schema,
                sequence_length=window_size,
                validate_data=False
            )
            
            print(f"Dataset created: {len(dataset)} sequences")
            
            # Get fresh algorithms for this window (needed for models like DLinear that use seq_len)
            algorithms = create_algorithm_configs(schema)
            
            # Update DLinear for current window size
            if 'dlinear' in algorithms:
                algorithms['dlinear']['model'] = DLinearWrapper(seq_len=window_size, kernel_size=3)
            
            # Add LSTM back to evaluation
            algorithms['lstm'] = {
                'model': PyTorchAdapter(
                    SimpleLSTM(
                        input_size=len(schema.feature_columns),
                        hidden_size=64,
                        num_layers=2,
                        dropout=0.2
                    ),
                    schema=schema
                ),
                'description': 'LSTM with temporal modeling',
                'category': 'neural',
                'is_pytorch': True
            }
            
            # Run evaluation for each algorithm
            for algo_name, algo_config in algorithms.items():
                print(f"  Evaluating {algo_name}...", end='', flush=True)
                start_time = time.time()
                
                try:
                    # Create adapter based on model type
                    if algo_config.get('is_pytorch', False):
                        model = algo_config['model']
                        cv = CrossValidator(model, dataset)
                        cv_results = cv.cross_validate(
                            n_splits=5,
                            test_size=1,
                            epochs=30,  # Reduced for faster evaluation
                            batch_size=32,
                            early_stopping_patience=3,
                            verbose=False
                        )
                    else:
                        # Check if this is a mixed effects model
                        if algo_config.get('requires_student_id', False):
                            from src.framework.adapters.mixed_effects_sklearn_adapter import MixedEffectsSKLearnAdapter
                            model = MixedEffectsSKLearnAdapter(
                                mixed_effects_model=algo_config['model'],
                                schema=schema,
                                lag_window=window_size
                            )
                        else:
                            model = SchemaBasedSKLearnAdapter(
                                sklearn_model=algo_config['model'],
                                schema=schema,
                                lag_window=window_size
                            )
                        
                        cv = CrossValidator(model, dataset)
                        cv_results = cv.cross_validate(n_splits=5, test_size=1)
                    
                    # Calculate R² if predictions are available
                    r2_mean = None
                    if 'predictions' in cv_results and 'actuals' in cv_results:
                        try:
                            # Calculate R² for each fold
                            r2_scores = []
                            for pred, actual in zip(cv_results['predictions'], cv_results['actuals']):
                                if len(pred) > 0 and len(actual) > 0:
                                    r2 = r2_score(actual, pred)
                                    r2_scores.append(r2)
                            if r2_scores:
                                r2_mean = np.mean(r2_scores)
                        except:
                            r2_mean = None
                    
                    # Store results
                    result_entry = {
                        'window_size': window_size,
                        'model': algo_name,
                        'category': algo_config['category'],
                        'mae_mean': cv_results['mae_mean'],
                        'mae_std': cv_results['mae_std'],
                        'rmse_mean': cv_results['rmse_mean'],
                        'rmse_std': cv_results['rmse_std'],
                        'smape_mean': cv_results['smape_mean'],
                        'smape_std': cv_results['smape_std'],
                        'r2_mean': r2_mean,
                        'training_time': time.time() - start_time,
                        'description': algo_config['description']
                    }
                    
                    # Extract feature importance if available
                    if hasattr(algo_config['model'], 'feature_importances_'):
                        try:
                            importances = algo_config['model'].feature_importances_
                            feature_names = model.get_feature_names() if hasattr(model, 'get_feature_names') else None
                            
                            if feature_names and len(feature_names) == len(importances):
                                for feat_idx, (feat_name, importance) in enumerate(zip(feature_names, importances)):
                                    feature_importance_results.append({
                                        'window_size': window_size,
                                        'model': algo_name,
                                        'feature': feat_name,
                                        'importance': importance,
                                        'feature_rank': feat_idx
                                    })
                        except:
                            pass
                    
                    all_results.append(result_entry)
                    print(f" ✓ MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}")
                    
                except Exception as e:
                    print(f" ✗ Failed: {str(e)}")
                    
        except Exception as e:
            print(f"Failed to create dataset for window size {window_size}: {str(e)}")
            continue
    
    # Convert to DataFrames
    df_results = pd.DataFrame(all_results)
    df_feature_importance = pd.DataFrame(feature_importance_results) if feature_importance_results else None
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80)
    
    # 1. Find optimal window size for each model
    optimal_windows = df_results.groupby('model')['mae_mean'].idxmin()
    df_optimal = df_results.loc[optimal_windows][['model', 'window_size', 'mae_mean']].rename(
        columns={'window_size': 'optimal_window', 'mae_mean': 'optimal_mae'}
    )
    
    # 2. Calculate consistency metrics (rank correlation between consecutive windows)
    rank_correlations = []
    for i in range(len(window_sizes) - 1):
        w1, w2 = window_sizes[i], window_sizes[i+1]
        df_w1 = df_results[df_results['window_size'] == w1].sort_values('mae_mean')
        df_w2 = df_results[df_results['window_size'] == w2].sort_values('mae_mean')
        
        if len(df_w1) > 1 and len(df_w2) > 1:
            # Get ranks
            rank_w1 = {model: idx+1 for idx, model in enumerate(df_w1['model'])}
            rank_w2 = {model: idx+1 for idx, model in enumerate(df_w2['model'])}
            
            # Calculate Spearman correlation
            common_models = set(rank_w1.keys()) & set(rank_w2.keys())
            if len(common_models) > 1:
                ranks1 = [rank_w1[m] for m in common_models]
                ranks2 = [rank_w2[m] for m in common_models]
                corr, _ = stats.spearmanr(ranks1, ranks2)
                rank_correlations.append(corr)
    
    avg_rank_correlation = np.mean(rank_correlations) if rank_correlations else 0
    print(f"\nAverage rank correlation between consecutive windows: {avg_rank_correlation:.3f}")
    
    # 3. Learning curve analysis
    learning_curves = {}
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model].sort_values('window_size')
        if len(model_data) > 5:
            # Fit exponential decay: MAE = a * exp(-b * window) + c
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            try:
                x = model_data['window_size'].values
                y = model_data['mae_mean'].values
                popt, _ = curve_fit(exp_decay, x, y, p0=[1, 0.1, np.min(y)], maxfev=5000)
                
                # Calculate improvement rate
                improvement_5_to_15 = (y[0] - y[10]) / y[0] if len(y) > 10 else 0
                improvement_15_to_30 = (y[10] - y[-1]) / y[10] if len(y) > 10 else 0
                
                learning_curves[model] = {
                    'decay_rate': popt[1],
                    'asymptote': popt[2],
                    'improvement_5_to_15': improvement_5_to_15,
                    'improvement_15_to_30': improvement_15_to_30
                }
            except:
                learning_curves[model] = {
                    'decay_rate': 0,
                    'asymptote': np.min(model_data['mae_mean']),
                    'improvement_5_to_15': 0,
                    'improvement_15_to_30': 0
                }
    
    # 4. Stability analysis
    stability_metrics = {}
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model]
        mae_values = model_data['mae_mean'].values
        
        # Coefficient of variation
        cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
        
        # Range normalized by mean
        normalized_range = (np.max(mae_values) - np.min(mae_values)) / np.mean(mae_values)
        
        stability_metrics[model] = {
            'cv': cv,
            'normalized_range': normalized_range,
            'mae_std_across_windows': np.std(mae_values)
        }
    
    # Create summary DataFrames
    df_window_summary = df_results.groupby('window_size').agg({
        'mae_mean': ['min', 'max', 'mean', 'std']
    }).round(3)
    df_window_summary.columns = ['best_mae', 'worst_mae', 'avg_mae', 'mae_std']
    
    # Add best model for each window
    best_models = df_results.groupby('window_size').apply(
        lambda x: x.loc[x['mae_mean'].idxmin(), 'model']
    )
    df_window_summary['best_model'] = best_models
    df_window_summary['mae_spread'] = df_window_summary['worst_mae'] - df_window_summary['best_mae']
    df_window_summary['cv_across_models'] = df_window_summary['mae_std'] / df_window_summary['avg_mae']
    
    # Model consistency analysis
    model_consistency = []
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model]
        
        # Calculate average rank
        ranks = []
        for window in window_sizes:
            window_data = df_results[df_results['window_size'] == window].sort_values('mae_mean')
            if model in window_data['model'].values:
                rank = window_data['model'].tolist().index(model) + 1
                ranks.append(rank)
        
        if ranks:
            optimal_data = df_optimal[df_optimal['model'] == model]
            
            consistency_entry = {
                'model': model,
                'avg_rank': np.mean(ranks),
                'rank_std': np.std(ranks),
                'times_top1': sum(r == 1 for r in ranks),
                'times_top3': sum(r <= 3 for r in ranks),
                'best_window': int(optimal_data['optimal_window'].values[0]) if len(optimal_data) > 0 else None,
                'best_mae': float(optimal_data['optimal_mae'].values[0]) if len(optimal_data) > 0 else None,
                'worst_window': int(model_data.loc[model_data['mae_mean'].idxmax(), 'window_size']),
                'worst_mae': float(model_data['mae_mean'].max()),
                'cv': stability_metrics[model]['cv'],
                'learning_rate': learning_curves.get(model, {}).get('decay_rate', 0),
                'asymptotic_mae': learning_curves.get(model, {}).get('asymptote', None)
            }
            model_consistency.append(consistency_entry)
    
    df_model_consistency = pd.DataFrame(model_consistency).sort_values('avg_rank')
    
    # Save results
    output_dir = Path('experiments/outputs/window_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSVs
    df_results.to_csv(output_dir / f'window_performance_detailed_{timestamp}.csv', index=False)
    df_window_summary.to_csv(output_dir / f'window_performance_summary_{timestamp}.csv')
    df_model_consistency.to_csv(output_dir / f'model_consistency_analysis_{timestamp}.csv', index=False)
    
    if df_feature_importance is not None and len(df_feature_importance) > 0:
        df_feature_importance.to_csv(output_dir / f'feature_importance_{timestamp}.csv', index=False)
    
    # Create visualizations
    create_window_analysis_plots(df_results, df_model_consistency, output_dir, timestamp)
    
    # Print summary
    print("\n📊 WINDOW ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n🏆 Best Overall Performance:")
    best_overall = df_results.loc[df_results['mae_mean'].idxmin()]
    print(f"  Model: {best_overall['model']}")
    print(f"  Window Size: {best_overall['window_size']}")
    print(f"  MAE: {best_overall['mae_mean']:.3f} ± {best_overall['mae_std']:.3f}")
    
    print(f"\n📈 Most Consistent Models (lowest rank std):")
    for _, row in df_model_consistency.head(3).iterrows():
        print(f"  {row['model']}: avg rank {row['avg_rank']:.1f} ± {row['rank_std']:.2f}")
    
    print(f"\n🎯 Optimal Window Sizes by Model:")
    for _, row in df_model_consistency.head(5).iterrows():
        print(f"  {row['model']}: window {row['best_window']} (MAE={row['best_mae']:.3f})")
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return df_results, df_model_consistency, df_window_summary


def create_window_analysis_plots(df_results, df_model_consistency, output_dir, timestamp):
    """Create visualizations for window size analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Bar plots for selected windows
    selected_windows = [5, 10, 15, 20, 25]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, window in enumerate(selected_windows):
        ax = axes[idx]
        window_data = df_results[df_results['window_size'] == window].sort_values('mae_mean')
        
        # Create bar plot
        bars = ax.bar(range(len(window_data)), window_data['mae_mean'], 
                      yerr=window_data['mae_std'], capsize=5, alpha=0.7)
        
        # Color by category
        colors = {'baseline': 'blue', 'classical': 'green', 'ensemble': 'orange', 
                 'neural': 'red', 'time_series': 'purple', 'mixed_effects': 'brown'}
        for bar, cat in zip(bars, window_data['category']):
            bar.set_color(colors.get(cat, 'gray'))
        
        ax.set_xticks(range(len(window_data)))
        ax.set_xticklabels(window_data['model'], rotation=45, ha='right')
        ax.set_ylabel('MAE')
        ax.set_title(f'Window Size = {window}')
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot since we only have 5 windows
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'mae_by_window_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning curves for all models
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model].sort_values('window_size')
        ax.plot(model_data['window_size'], model_data['mae_mean'], 
               marker='o', label=model, alpha=0.7)
    
    ax.set_xlabel('Window Size (weeks)')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Learning Curves: MAE vs Window Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'learning_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of model rankings
    # Create ranking matrix
    models = df_results['model'].unique()
    windows = sorted(df_results['window_size'].unique())
    ranking_matrix = np.zeros((len(models), len(windows)))
    
    for j, window in enumerate(windows):
        window_data = df_results[df_results['window_size'] == window].sort_values('mae_mean')
        for i, model in enumerate(models):
            if model in window_data['model'].values:
                rank = window_data['model'].tolist().index(model) + 1
                ranking_matrix[i, j] = rank
            else:
                ranking_matrix[i, j] = len(models) + 1  # Worst rank if missing
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use custom colormap - lower ranks (better) are darker
    cmap = plt.cm.RdYlGn_r  # Reversed so green is good (low rank)
    im = ax.imshow(ranking_matrix, cmap=cmap, aspect='auto', vmin=1, vmax=len(models))
    
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Model')
    ax.set_title('Model Rankings Across Window Sizes (lower is better)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank')
    
    # Add text annotations for top 3 models in each window
    for j in range(len(windows)):
        window_ranks = ranking_matrix[:, j]
        top_3_indices = np.argsort(window_ranks)[:3]
        for idx in top_3_indices:
            if window_ranks[idx] <= 3:
                ax.text(j, idx, f'{int(window_ranks[idx])}', 
                       ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'ranking_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Stability vs Performance scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use model consistency data
    scatter = ax.scatter(df_model_consistency['best_mae'], 
                        df_model_consistency['cv'],
                        s=100 + df_model_consistency['times_top3'] * 20,  # Size by consistency
                        alpha=0.6)
    
    # Add labels
    for _, row in df_model_consistency.iterrows():
        ax.annotate(row['model'], 
                   (row['best_mae'], row['cv']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    ax.set_xlabel('Best MAE (lower is better)')
    ax.set_ylabel('Coefficient of Variation (lower is more stable)')
    ax.set_title('Model Performance vs Stability\n(bubble size = times in top 3)')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    median_mae = df_model_consistency['best_mae'].median()
    median_cv = df_model_consistency['cv'].median()
    ax.axvline(median_mae, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(median_cv, color='gray', linestyle='--', alpha=0.5)
    
    # Label quadrants
    ax.text(0.02, 0.98, 'Stable but\nPoor', transform=ax.transAxes, 
            va='top', ha='left', alpha=0.5, fontsize=10)
    ax.text(0.98, 0.98, 'Unstable and\nPoor', transform=ax.transAxes, 
            va='top', ha='right', alpha=0.5, fontsize=10)
    ax.text(0.02, 0.02, 'Stable and\nGood', transform=ax.transAxes, 
            va='bottom', ha='left', alpha=0.5, fontsize=10, weight='bold')
    ax.text(0.98, 0.02, 'Good but\nUnstable', transform=ax.transAxes, 
            va='bottom', ha='right', alpha=0.5, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'stability_vs_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
 