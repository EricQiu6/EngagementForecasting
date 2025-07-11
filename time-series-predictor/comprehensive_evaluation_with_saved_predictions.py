"""
Comprehensive Evaluation with Saved Predictions
==============================================

This script runs the same evaluation as comprehensive_evaluation.py but saves
all predictions and metadata for later analysis. This enables:
1. Faster iteration on plotting and analysis
2. Bootstrapping and significance testing
3. Detailed error analysis
4. Feature importance extraction
"""

import pandas as pd
import numpy as np
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Framework imports
from src.framework.core.schema import get_schema, DataSchema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters import SchemaBasedSKLearnAdapter, PyTorchAdapter
from src.framework.core.base import CrossValidator, MetricsCalculator

# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor

# Framework models
from src.framework.models.baselines import MeanPredictor, NaiveForecast, MedianPredictor, MedianPredictorNoZeros, MeanPredictorNoZeros
from src.framework.models.neural_nets import SimpleLSTM, create_model

# Try to import optional dependencies
try:
    from src.framework.models.goal_based_predictor import GoalBasedPredictor
    HAS_GOAL_BASED = True
except ImportError:
    HAS_GOAL_BASED = False
    print("Goal-based predictor not available")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from src.framework.adapters.mixed_effects_sklearn_adapter import SchemaAwareMixedEffectsAdapter
    HAS_MIXED_EFFECTS = True
except ImportError:
    HAS_MIXED_EFFECTS = False
    print("Mixed effects models not available")


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
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


class PredictionSaver:
    """Saves predictions and metadata during cross-validation."""
    
    def __init__(self, output_dir: str = "evaluation_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_fold_predictions(self, model_name: str, fold_idx: int, 
                            y_true: np.ndarray, y_pred: np.ndarray,
                            indices: List[int], metadata: Dict[str, Any] = None):
        """Save predictions for a single fold."""
        fold_data = {
            'model_name': model_name,
            'fold_idx': fold_idx,
            'y_true': convert_numpy(y_true),
            'y_pred': convert_numpy(y_pred),
            'indices': indices,
            'metadata': convert_numpy(metadata or {})
        }
        
        # Save to model-specific directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        filename = model_dir / f"fold_{fold_idx}_predictions.json"
        with open(filename, 'w') as f:
            json.dump(fold_data, f, indent=2)
            
    def save_model_summary(self, model_name: str, cv_results: Dict[str, Any], 
                         model_config: Dict[str, Any], training_time: float):
        """Save overall model results and configuration."""
        # Create serializable version of model_config
        serializable_config = {}
        for key, value in model_config.items():
            if key == 'model':
                # Store model type name instead of the object
                if value is not None:
                    serializable_config[key] = type(value).__name__
                else:
                    serializable_config[key] = None
            else:
                serializable_config[key] = convert_numpy(value)
        
        summary = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'cv_results': convert_numpy(cv_results),
            'model_config': serializable_config,
            'training_time': training_time
        }
        
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    def save_evaluation_config(self, config: Dict[str, Any]):
        """Save the overall evaluation configuration."""
        with open(self.output_dir / 'evaluation_config.json', 'w') as f:
            json.dump(convert_numpy(config), f, indent=2)


class ExtendedCrossValidator(CrossValidator):
    """Extended cross-validator that saves predictions."""
    
    def __init__(self, model, dataset, prediction_saver: PredictionSaver = None, model_name: str = ""):
        super().__init__(model, dataset)
        self.prediction_saver = prediction_saver
        self.model_name = model_name
        
    def cross_validate(self, n_splits: int = 5, test_size: int = 1, **fit_kwargs) -> Dict[str, Any]:
        """Perform cross-validation with prediction saving."""
        fold_results = []
        splits = self.dataset.get_splits(n_splits, test_size)
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"  Fold {fold_idx + 1}/{n_splits}")
            
            # Get fold data
            train_data = self._get_fold_data(train_indices)
            val_data = self._get_fold_data(val_indices)
            
            # Train model
            history = self.model.fit(train_data, val_data, **fit_kwargs)
            
            # Evaluate
            y_pred = self.model.predict(val_data)
            y_true = self._extract_targets(val_data)
            
            # Save predictions if saver provided
            if self.prediction_saver:
                self.prediction_saver.save_fold_predictions(
                    self.model_name, fold_idx, y_true, y_pred, val_indices
                )
            
            metrics = MetricsCalculator.calculate_metrics(y_true, y_pred)
            metrics['fold'] = fold_idx
            fold_results.append(metrics)
            
        return self._aggregate_results(fold_results)


def create_all_models() -> Dict[str, Dict[str, Any]]:
    """Create all models to evaluate."""
    models = {}
    
    # 1. Trivial Baselines
    models['average_all'] = {
        'model': MeanPredictor(),
        'category': 'baseline',
        'description': 'Simple average of all training values'
    }
    
    models['naive_forecast'] = {
        'model': NaiveForecast(),
        'category': 'baseline',
        'description': 'Last observed value'
    }
    
    models['median_all'] = {
        'model': MedianPredictor(),
        'category': 'baseline',
        'description': 'Median of all training values'
    }
    
    models['median_no_zeros'] = {
        'model': MedianPredictorNoZeros(),
        'category': 'baseline',
        'description': 'Median excluding zero values'
    }
    
    models['mean_no_zeros'] = {
        'model': MeanPredictorNoZeros(),
        'category': 'baseline',
        'description': 'Mean excluding zero values'
    }
    
    # 2. Goal-based predictors (if available)
    if HAS_GOAL_BASED:
        models['adams_baseline_50'] = {
            'model': GoalBasedPredictor(prediction_percentile=50, adjustment_factor=0.5, random_state=42),
            'category': 'goal_based',
            'description': 'Goal-based predictor with 50th percentile'
        }
        
        models['adams_baseline_60'] = {
            'model': GoalBasedPredictor(prediction_percentile=60, adjustment_factor=0.5, random_state=42),
            'category': 'goal_based',
            'description': 'Goal-based predictor with 60th percentile'
        }
        
        models['adams_baseline_70'] = {
            'model': GoalBasedPredictor(prediction_percentile=70, adjustment_factor=0.5, random_state=42),
            'category': 'goal_based',
            'description': 'Goal-based predictor with 70th percentile'
        }
    
    # 3. Linear Models
    models['linear_regression'] = {
        'model': LinearRegression(),
        'category': 'linear',
        'description': 'Standard linear regression'
    }
    
    models['ridge'] = {
        'model': Ridge(alpha=1.0),
        'category': 'linear',
        'description': 'Ridge regression (L2 regularization)'
    }
    
    models['lasso'] = {
        'model': Lasso(alpha=0.1, max_iter=5000),
        'category': 'linear',
        'description': 'Lasso regression (L1 regularization)'
    }
    
    # 4. Tree-based Models
    models['random_forest'] = {
        'model': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),
        'category': 'tree',
        'description': 'Random Forest with tuned hyperparameters'
    }
    
    # 5. XGBoost (if available)
    if HAS_XGBOOST:
        models['xgboost'] = {
            'model': XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'category': 'tree',
            'description': 'XGBoost with tuned hyperparameters'
        }
    
    # 6. Neural Network Models
    models['mlp'] = {
        'model': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ),
        'category': 'neural',
        'description': 'Multi-Layer Perceptron'
    }
    
    models['lstm'] = {
        'model': None,  # Will be created during evaluation with correct input size
        'category': 'neural',
        'description': 'LSTM with temporal modeling',
        'is_pytorch': True,
        'lstm_config': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
    }
    
    # 7. Mixed Effects Models (if available)
    if HAS_MIXED_EFFECTS:
        models['mixed_effects'] = {
            'model': None,  # SchemaAwareMixedEffectsAdapter is both model and adapter
            'category': 'mixed_effects',
            'description': 'Schema-aware mixed effects with student effects',
            'requires_student_id': True
        }
    
    return models


def run_evaluation_with_predictions(
    schema_name: str = 'time_goal_extended',
    window_size: int = 8,
    target_type: str = 'minutes_per_week',
    save_predictions: bool = True
):
    """Run comprehensive evaluation with prediction saving."""
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION WITH SAVED PREDICTIONS")
    print(f"{'='*80}")
    
    # Setup configuration
    evaluation_config = {
        'schema_name': schema_name,
        'window_size': window_size,
        'target_type': target_type,
        'n_splits': 5,
        'test_size': 1,
        'timestamp': datetime.now().isoformat()
    }
    
    # Load data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists or update the path.")
        return None
    
    # Select schema based on target type
    if target_type == 'minutes_per_week':
        schema = get_schema('time_goal_extended')
    else:  # avg_proficiency
        schema = get_schema('extended')
    
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=window_size,
        validate_data=False
    )
    
    print(f"\nDataset loaded:")
    print(f"  - Total sequences: {len(dataset)}")
    print(f"  - Window size: {window_size}")
    print(f"  - Target: {target_type}")
    print(f"  - Schema: {schema_name}")
    
    # Create prediction saver
    saver = PredictionSaver(f"evaluation_outputs/{target_type}_window{window_size}")
    saver.save_evaluation_config(evaluation_config)
    
    # Get all models
    models = create_all_models()
    
    # Run evaluation
    results = {}
    
    print(f"\nEvaluating {len(models)} models...")
    print("="*80)
    
    for model_name, model_config in models.items():
        print(f"\n🔄 {model_name}: {model_config['description']}...")
        start_time = time.time()
        
        try:
            # Create appropriate adapter
            if model_config.get('requires_student_id', False):
                # Mixed effects model
                if HAS_MIXED_EFFECTS:
                    adapter = SchemaAwareMixedEffectsAdapter(
                        sklearn_model=None,  # Not used
                        schema=schema,
                        lag_window=window_size,
                        target_col='minutes_per_week'
                    )
                else:
                    print("❌ Mixed effects not available, skipping...")
                    continue
            elif model_config.get('is_pytorch', False):
                # PyTorch model (LSTM) - create dynamically with correct input size
                if model_name == 'lstm':
                    lstm_config = model_config['lstm_config']
                    lstm_model = SimpleLSTM(
                        input_size=len(schema.feature_columns),
                        hidden_size=lstm_config['hidden_size'],
                        num_layers=lstm_config['num_layers'],
                        dropout=lstm_config['dropout']
                    )
                    adapter = PyTorchAdapter(lstm_model, schema=schema)
                else:
                    # Other PyTorch models
                    model = model_config['model']
                    model.schema = schema
                    adapter = model
            else:
                # Standard sklearn model
                adapter = SchemaBasedSKLearnAdapter(
                    sklearn_model=model_config['model'],
                    schema=schema,
                    lag_window=window_size
                )
            
            # Create extended cross-validator
            cv = ExtendedCrossValidator(
                adapter, dataset, 
                prediction_saver=saver if save_predictions else None,
                model_name=model_name
            )
            
            # Run cross-validation with appropriate parameters
            if model_config.get('is_pytorch', False):
                # PyTorch models need different training parameters
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size'],
                    epochs=50,
                    batch_size=32,
                    early_stopping_patience=5,
                    verbose=False
                )
            else:
                # Standard models
                cv_results = cv.cross_validate(
                    n_splits=evaluation_config['n_splits'],
                    test_size=evaluation_config['test_size']
                )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save model summary
            if save_predictions:
                saver.save_model_summary(model_name, cv_results, model_config, training_time)
            
            # Store results
            results[model_name] = {
                **cv_results,
                'category': model_config['category'],
                'description': model_config['description'],
                'training_time': training_time
            }
            
            print(f"✅ Completed: MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}, Time={training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Save overall results
    overall_results = {
        'evaluation_config': evaluation_config,
        'model_results': results,
        'summary_statistics': calculate_summary_statistics(results)
    }
    
    results_file = saver.output_dir / 'overall_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(overall_results), f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {saver.output_dir}")
    print_summary(results)
    
    return results


def calculate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics across all models."""
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return {}
    
    mae_values = [v['mae_mean'] for v in valid_results.values()]
    
    return {
        'n_models': len(results),
        'n_successful': len(valid_results),
        'best_model': min(valid_results.items(), key=lambda x: x[1]['mae_mean'])[0],
        'worst_model': max(valid_results.items(), key=lambda x: x[1]['mae_mean'])[0],
        'mae_range': [min(mae_values), max(mae_values)],
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values)
    }


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    # Sort by MAE
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'mae_mean' in v],
        key=lambda x: x[1]['mae_mean']
    )
    
    print("\nModel Performance Ranking (by MAE):")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<30} {'Category':<15} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{i:<5} {model_name:<30} {result['category']:<15} "
              f"{result['mae_mean']:<10.3f} {result['rmse_mean']:<10.3f}")


if __name__ == "__main__":
    # Run evaluation for minutes_per_week with different window sizes
    for window_size in [8]:  # Start with window size 8
        print(f"\n\n{'#'*80}")
        print(f"# WINDOW SIZE: {window_size}")
        print(f"{'#'*80}")
        
        run_evaluation_with_predictions(
            schema_name='time_goal_extended',
            window_size=window_size,
            target_type='minutes_per_week',
            save_predictions=True
        ) 