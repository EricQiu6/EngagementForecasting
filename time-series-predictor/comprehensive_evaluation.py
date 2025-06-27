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

# Algorithm imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor

# Import existing baseline models from our framework
from src.framework.models.baselines import AveragePredictor, NaiveForecast, LinearTrend

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


def create_algorithm_configs():
    """Create configurations for all algorithms to evaluate."""
    
    algorithms = {
        # Trivial baselines (using our framework's baseline models)
        'average_predictor': {
            'model': AveragePredictor(),
            'description': 'Historical average predictor',
            'category': 'trivial'
        },
        
        'naive_forecast': {
            'model': NaiveForecast(),
            'description': 'Naive forecast (last value)',
            'category': 'trivial'
        },
        
        'linear_trend': {
            'model': LinearTrend(),
            'description': 'Simple linear trend model',
            'category': 'trivial'
        },
        
        # Sklearn baselines for comparison
        'mean_baseline': {
            'model': TrivialBaselines.create_averaging_model(),
            'description': 'Always predicts the mean value (sklearn)',
            'category': 'trivial'
        },
        
        'last_value': {
            'model': TrivialBaselines.create_last_value_model(),
            'description': 'Predicts the last observed value (custom)',
            'category': 'trivial'
        },
        
        # Classical ML algorithms
        'linear_regression': {
            'model': LinearRegression(),
            'description': 'Simple linear regression',
            'category': 'classical'
        },
        
        'ridge_regression': {
            'model': Ridge(alpha=1.0),
            'description': 'Ridge regression with L2 regularization',
            'category': 'classical'
        },
        
        'lasso_regression': {
            'model': Lasso(alpha=0.1),
            'description': 'Lasso regression with L1 regularization',
            'category': 'classical'
        },
        
        'elastic_net': {
            'model': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'description': 'Elastic Net with L1 and L2 regularization',
            'category': 'classical'
        },
        
        'random_forest': {
            'model': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'description': 'Random Forest with 100 trees',
            'category': 'ensemble'
        },
        
        'gradient_boosting': {
            'model': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'description': 'Gradient Boosting Machine',
            'category': 'ensemble'
        },
        
        'svr_rbf': {
            'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'description': 'Support Vector Regression with RBF kernel',
            'category': 'classical'
        },
        
        'mlp_small': {
            'model': MLPRegressor(
                hidden_layer_sizes=(50,),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=500,
                random_state=42
            ),
            'description': 'Small Multi-Layer Perceptron (1 hidden layer, 50 units)',
            'category': 'neural'
        },
        
        'mlp_medium': {
            'model': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=500,
                random_state=42
            ),
            'description': 'Medium Multi-Layer Perceptron (2 hidden layers)',
            'category': 'neural'
        },
        
        # Note: Student ability models will be added separately if data supports them
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        algorithms['xgboost'] = {
            'model': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'description': 'XGBoost Gradient Boosting',
            'category': 'ensemble'
        }
    
    return algorithms


def setup_evaluation_data():
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
    
    # Determine appropriate schema
    if 'name' in df.columns and 'week' in df.columns and 'proficient' in df.columns:
        schema_name = 'legacy'
        print(f"Using legacy schema")
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


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with realistic settings."""
    
    print("=" * 80)
    print("COMPREHENSIVE TIME SERIES ALGORITHM EVALUATION")
    print("=" * 80)
    
    # Setup data
    data_path, schema_name = setup_evaluation_data()
    
    if schema_name == 'legacy':
        schema = get_schema('legacy')
    elif schema_name == 'student_week':
        schema = get_schema('student_week')
    elif schema_name == 'extended':
        schema = get_schema('extended')
    else:
        # Use custom schema created in setup
        schema = schema_name  # This would be the DataSchema object
    
    # Realistic evaluation settings
    evaluation_config = {
        'sequence_length': 8,  # Sufficient history
        'n_splits': 5,         # Appropriate number of folds
        'test_size': 2,        # Test on 2 weeks
        'min_samples_per_student': 10,  # Minimum data per student
        'validation_strategy': 'time_series_cv'
    }
    
    print(f"\nEvaluation Configuration:")
    for key, value in evaluation_config.items():
        print(f"  {key}: {value}")
    
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
    algorithms = create_algorithm_configs()
    
    # Check if we can add student ability models
    algorithms = add_student_ability_models_if_possible(algorithms, schema, dataset)
    
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
            # Check if this is already a schema-based model (like PyTorch models)
            if hasattr(algo_config['model'], 'fit') and hasattr(algo_config['model'], 'predict') and hasattr(algo_config['model'], 'schema'):
                # Already a schema-based model (e.g., PyTorch adapter)
                model = algo_config['model']
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
    
    print("🚀 Starting comprehensive time series algorithm evaluation...")
    print("Using the new schema-based framework with realistic settings.")
    
    try:
        # Run evaluation
        results, config = run_comprehensive_evaluation()
        
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


if __name__ == "__main__":
    main()
 