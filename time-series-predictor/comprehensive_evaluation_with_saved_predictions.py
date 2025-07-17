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
import itertools
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


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

class ExperimentConfig:
    """Configuration class for comprehensive evaluation experiments."""
    
    def __init__(self):
        # Dataset configurations
        self.dataset_directory = '../data-analysis/'
        self.available_datasets = {
            'rolling_new': 'student_week_aggregations_rolling_new.csv',
            'rolling': 'student_week_aggregations_rolling.csv',
            'steve_dang_100': 'student_week_aggregations_steve_dang_100.csv',
            'combined': 'student_week_aggregations_combined.csv'
        }
        
        # Goal type configurations
        self.goal_types = {
            'minutes': {
                'target_column': 'minutes_per_week',
                'schema_name': 'time_goal_extended',
                'description': 'Predicting minutes per week spent on learning'
            },
            'proficiency': {
                'target_column': 'avg_proficiency', 
                'schema_name': 'extended',
                'description': 'Predicting average proficiency scores'
            }
        }
        
        # Window size configurations
        self.window_sizes = {
            'small': [3, 5, 8],
            'medium': [8, 12, 15],
            'large': [15, 20, 25],
            'comprehensive': list(range(3, 31, 3)),  # 3, 6, 9, ..., 30
            'focused': [8, 15],  # Common sizes for quick testing
        }
        
        # Feature selection configurations - INCLUDING ENGINEERED FEATURES
        self.feature_sets = {
            'all': None,  # Use all features from schema
            'top_5_temporal': [
                'week_id', 'weeks_since_start', 'is_first_week', 
                'temporal_gap_weeks', 'time_since_last_activity'
            ],
            'top_5_performance': [
                'avg_proficiency', 'total_correct', 'total_incorrect',
                'success_rate', 'avg_attempts_per_problem'
            ],
            'top_5_engagement': [
                'minutes_per_week', 'problems_attempted', 'sessions_count',
                'avg_session_length', 'days_active'
            ],
            'top_5_selected': [
                'minutes_mean', 'current_minutes_per_week', 'minutes_std',
                'target_lag5', 'current_week_id'
            ],
            'minimal_goal': [
                'week_id', 'avg_proficiency', 'minutes_per_week'
            ],
            'skill_focused': [
                'avg_proficiency', 'skill_proficiency_change',
                'skill_difficulty_avg', 'skill_engagement_score'
            ]
        }
        
        # Cross-validation configurations
        self.cv_configs = {
            'standard': {'n_splits': 5, 'test_size': 1},
            'robust': {'n_splits': 7, 'test_size': 1},
            'quick': {'n_splits': 3, 'test_size': 1},
            'multi_step': {'n_splits': 5, 'test_size': 2}
        }
        
        # Model selection configurations
        self.model_sets = {
            'baselines_only': ['average_all', 'naive_forecast', 'median_all'],
            'linear_models': ['linear_regression', 'ridge', 'lasso'],
            'tree_models': ['random_forest', 'xgboost'],
            'neural_models': ['mlp', 'lstm'],
            'goal_based': ['adams_baseline_50', 'adams_baseline_60', 'adams_baseline_70'],
            'mixed_effects': ['mixed_effects'],
            'top_performers': ['lasso', 'random_forest', 'ridge', 'mixed_effects'],
            'all': None  # Use all available models
        }
        
        # Output configurations
        self.output_base_dir = 'evaluation_outputs_with_features'
        self.save_predictions = True
        self.save_models = False  # Set to True to save trained models
        self.save_feature_importance = True
        
        # NEW: Hyperparameter sensitivity configurations
        self.hyperparameter_grids = {
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 5000]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'mlp': {
                'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        # NEW: Analysis modes
        self.analysis_modes = {
            'single_config': 'Current approach - one config per model',
            'hyperparameter_sensitivity': 'Multiple configs per model for sensitivity analysis',
            'both': 'Run both single and multiple configs'
        }
        
        # NEW: Hyperparameter sensitivity configurations
        self.hyperparameter_grids = {
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 5000]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'mlp': {
                'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        # NEW: Analysis modes
        self.analysis_modes = {
            'single_config': 'Current approach - one config per model',
            'hyperparameter_sensitivity': 'Multiple configs per model for sensitivity analysis',
            'both': 'Run both single and multiple configs'
        }

    def get_experiment_name(self, dataset_name: str, goal_type: str, 
                          window_size: int, feature_set: str, 
                          cv_config: str, model_set: str) -> str:
        """Generate experiment name from configuration."""
        return f"{dataset_name}_{goal_type}_w{window_size}_{feature_set}_{cv_config}_{model_set}"
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get full path to dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.available_datasets.keys())}")
        return str(Path(self.dataset_directory) / self.available_datasets[dataset_name])

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate experiment configuration."""
        required_keys = ['dataset_name', 'goal_type', 'window_size', 'feature_set', 'cv_config', 'model_set']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate individual components
        if config['dataset_name'] not in self.available_datasets:
            raise ValueError(f"Invalid dataset_name: {config['dataset_name']}")
        
        if config['goal_type'] not in self.goal_types:
            raise ValueError(f"Invalid goal_type: {config['goal_type']}")
        
        if config['feature_set'] not in self.feature_sets:
            raise ValueError(f"Invalid feature_set: {config['feature_set']}")
        
        if config['cv_config'] not in self.cv_configs:
            raise ValueError(f"Invalid cv_config: {config['cv_config']}")
        
        if config['model_set'] not in self.model_sets:
            raise ValueError(f"Invalid model_set: {config['model_set']}")
        
        if not isinstance(config['window_size'], int) or config['window_size'] < 1:
            raise ValueError(f"Invalid window_size: {config['window_size']}")
        
        return True


# Default experiment configuration
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


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


def create_models_with_hyperparameters(analysis_mode: str = 'single_config', 
                                     config_obj = None) -> Dict[str, Dict[str, Any]]:
    """Create models based on analysis mode with hyperparameter variations."""
    
    if config_obj is None:
        config_obj = DEFAULT_EXPERIMENT_CONFIG
    
    if analysis_mode == 'single_config':
        return create_all_models()  # Current function
    
    elif analysis_mode == 'hyperparameter_sensitivity':
        models = {}
        
        # Get baseline models (keep these as single configs)
        baseline_models = create_all_models()
        baseline_categories = ['baseline', 'goal_based']
        
        for name, config in baseline_models.items():
            if config['category'] in baseline_categories:
                models[name] = config
        
        # Generate hyperparameter variations for key models
        for base_model, param_grid in config_obj.hyperparameter_grids.items():
            if base_model not in param_grid:
                continue
                
            # Generate all combinations of hyperparameters
            param_names = list(param_grid.keys())
            param_combinations = list(itertools.product(*param_grid.values()))
            
            for i, param_values in enumerate(param_combinations):
                params = dict(zip(param_names, param_values))
                model_name = f"{base_model}_hp_{i:02d}"
                
                # Create model instance based on base model type
                try:
                    if base_model == 'lasso':
                        model_instance = Lasso(**params, random_state=42)
                        category = 'linear'
                    elif base_model == 'ridge':
                        model_instance = Ridge(**params, random_state=42)
                        category = 'linear'
                    elif base_model == 'random_forest':
                        model_instance = RandomForestRegressor(**params, random_state=42)
                        category = 'tree'
                    elif base_model == 'xgboost' and HAS_XGBOOST:
                        model_instance = XGBRegressor(**params, random_state=42)
                        category = 'tree'
                    elif base_model == 'mlp':
                        mlp_params = {
                            'activation': 'relu',
                            'solver': 'adam',
                            'max_iter': 500,
                            'early_stopping': True,
                            'validation_fraction': 0.1,
                            'n_iter_no_change': 10,
                            'random_state': 42,
                            **params
                        }
                        model_instance = MLPRegressor(**mlp_params)
                        category = 'neural'
                    else:
                        continue  # Skip if model not available
                    
                    models[model_name] = {
                        'model': model_instance,
                        'category': category,
                        'base_model': base_model,
                        'hyperparams': params,
                        'description': f'{base_model} ({", ".join(f"{k}={v}" for k, v in params.items())})'
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not create {model_name} with params {params}: {e}")
                    continue
        
        return models
    
    elif analysis_mode == 'both':
        single_models = create_all_models()
        hp_models = create_models_with_hyperparameters('hyperparameter_sensitivity', config_obj)
        
        # Rename single models to distinguish them
        renamed_single = {}
        for name, config in single_models.items():
            if name in ['lasso', 'ridge', 'random_forest', 'xgboost', 'mlp']:
                renamed_single[f"{name}_default"] = {
                    **config,
                    'base_model': name,
                    'hyperparams': 'default',
                    'description': f"{config['description']} (default)"
                }
            else:
                renamed_single[name] = config
        
        return {**renamed_single, **hp_models}
    
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")


def get_model_category(base_model: str) -> str:
    """Get category for a base model."""
    category_map = {
        'lasso': 'linear',
        'ridge': 'linear', 
        'linear_regression': 'linear',
        'random_forest': 'tree',
        'xgboost': 'tree',
        'mlp': 'neural',
        'lstm': 'neural'
    }
    return category_map.get(base_model, 'unknown')


def create_custom_schema_with_features(base_schema: DataSchema, selected_features: List[str]) -> DataSchema:
    """Create a custom schema with only selected features."""
    # Check if these are engineered features (contain underscores suggesting they're generated)
    engineered_indicators = ['current_', 'lag', '_mean', '_std', '_range', '_iqr', 'target_lag', 'gap_', 'recent_', 'avg_']
    
    is_engineered_features = any(any(indicator in feature for indicator in engineered_indicators) 
                                for feature in selected_features)
    
    if is_engineered_features:
        # For engineered features, create a custom schema that will be filtered by the adapter
        # We'll use all base schema features and let the adapter do the filtering
        print(f"🔧 Detected engineered features: {selected_features}")
        print(f"   Will be filtered by adapter after feature engineering")
        
        # Store the target engineered features in the schema for later filtering
        custom_schema = DataSchema(
            student_column=base_schema.student_column,
            time_column=base_schema.time_column,
            target_column=base_schema.target_column,
            feature_columns=base_schema.feature_columns,  # Keep all base features
            student_id_strategy=base_schema.student_id_strategy
        )
        
        # Add custom attribute to store target engineered features
        custom_schema._target_engineered_features = selected_features
        return custom_schema
    
    else:
        # Original logic for base schema features
        available_features = set(base_schema.feature_columns)
        valid_features = [f for f in selected_features if f in available_features]
        
        if not valid_features:
            raise ValueError(f"None of the selected features {selected_features} are available in schema. "
                           f"Available features: {available_features}")
        
        custom_schema = DataSchema(
            student_column=base_schema.student_column,
            time_column=base_schema.time_column,
            target_column=base_schema.target_column,
            feature_columns=valid_features,
            student_id_strategy=base_schema.student_id_strategy
        )
        
        return custom_schema


def run_evaluation_with_predictions(
    experiment_config: Dict[str, Any],
    config_obj: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    analysis_mode: str = 'single_config'
):
    """Run comprehensive evaluation with prediction saving using experiment configuration."""
    
    # Validate configuration
    config_obj.validate_configuration(experiment_config)
    
    # Extract configuration
    dataset_name = experiment_config['dataset_name']
    goal_type = experiment_config['goal_type']
    window_size = experiment_config['window_size']
    feature_set = experiment_config['feature_set']
    cv_config = experiment_config['cv_config']
    model_set = experiment_config['model_set']
    
    # Get experiment name
    experiment_name = config_obj.get_experiment_name(
        dataset_name, goal_type, window_size, feature_set, cv_config, model_set
    )
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    # Get dataset path
    data_path = config_obj.get_dataset_path(dataset_name)
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists or update the path.")
        return None
    
    # Get goal configuration
    goal_config = config_obj.goal_types[goal_type]
    target_column = goal_config['target_column']
    schema_name = goal_config['schema_name']
    
    # Get base schema
    base_schema = get_schema(schema_name)
    
    # Apply feature selection if specified
    if feature_set != 'all' and config_obj.feature_sets[feature_set] is not None:
        selected_features = config_obj.feature_sets[feature_set]
        schema = create_custom_schema_with_features(base_schema, selected_features)
        print(f"🎯 Using feature set '{feature_set}': {len(selected_features)} features")
        print(f"   Features: {selected_features}")
    else:
        schema = base_schema
        print(f"🎯 Using all features from schema: {len(schema.feature_columns)} features")
    
    # Get CV configuration
    cv_params = config_obj.cv_configs[cv_config]
    
    # Setup evaluation configuration
    evaluation_config = {
        'experiment_name': experiment_name,
        'dataset_name': dataset_name,
        'dataset_path': data_path,
        'goal_type': goal_type,
        'target_column': target_column,
        'schema_name': schema_name,
        'window_size': window_size,
        'feature_set': feature_set,
        'n_features': len(schema.feature_columns),
        'cv_config': cv_config,
        'model_set': model_set,
        'timestamp': datetime.now().isoformat(),
        **cv_params
    }
    
    # Create dataset
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=window_size,
        validate_data=False
    )
    
    print(f"\nDataset Configuration:")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Total sequences: {len(dataset)}")
    print(f"  - Goal: {goal_type} ({target_column})")
    print(f"  - Window size: {window_size}")
    print(f"  - Features: {len(schema.feature_columns)}")
    print(f"  - CV: {cv_config} ({cv_params})")
    print(f"  - Models: {model_set}")
    
    # Create prediction saver
    output_dir = f"{config_obj.output_base_dir}/{experiment_name}"
    saver = PredictionSaver(output_dir)
    saver.save_evaluation_config(evaluation_config)
    
    # Get models based on model set and analysis mode
    if analysis_mode == 'single_config':
        all_models = create_all_models()
        if model_set == 'all' or config_obj.model_sets[model_set] is None:
            models = all_models
        else:
            selected_model_names = config_obj.model_sets[model_set]
            models = {name: all_models[name] for name in selected_model_names if name in all_models}
    else:
        # Use hyperparameter analysis
        all_models = create_models_with_hyperparameters(analysis_mode, config_obj)
        if model_set == 'all' or config_obj.model_sets[model_set] is None:
            models = all_models
        else:
            # Filter hyperparameter models to match model set
            selected_model_names = config_obj.model_sets[model_set]
            models = {}
            
            # Include baseline/goal-based models as they are
            for name, config in all_models.items():
                if name in selected_model_names or config.get('category') in ['baseline', 'goal_based']:
                    models[name] = config
                    continue
                    
                # Include hyperparameter variants of selected models
                base_model = config.get('base_model', name)
                if base_model in selected_model_names:
                    models[name] = config
    
    if not models:
        print(f"❌ No models available for model set: {model_set}")
        return None
    
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
                        target_col=target_column  # Use the configured target column
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
            
            # Apply engineered feature filtering if needed
            if hasattr(schema, '_target_engineered_features'):
                # Create a custom adapter that filters engineered features
                adapter = EngineeredFeatureFilterAdapter(adapter, schema._target_engineered_features)
            
            # Create extended cross-validator
            cv = ExtendedCrossValidator(
                adapter, dataset, 
                prediction_saver=saver if config_obj.save_predictions else None,
                model_name=model_name
            )
            
            # Run cross-validation with configured parameters
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
            if config_obj.save_predictions:
                saver.save_model_summary(model_name, cv_results, model_config, training_time)
            
            # Store results
            results[model_name] = {
                **cv_results,
                'category': model_config['category'],
                'description': model_config['description'],
                'training_time': training_time
            }
            
            print(f"✅ Completed: MAE={cv_results['mae_mean']:.2f}±{cv_results['mae_std']:.2f}, Time={training_time:.1f}s")
            
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
    
    # Run hyperparameter sensitivity analysis if applicable
    if analysis_mode in ['hyperparameter_sensitivity', 'both']:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        sensitivity_df = analyze_hyperparameter_sensitivity(results)
        
        if not sensitivity_df.empty:
            # Save sensitivity analysis
            sensitivity_path = saver.output_dir / 'hyperparameter_sensitivity.csv'
            sensitivity_df.to_csv(sensitivity_path, index=False)
            
            # Print summary
            print_hyperparameter_sensitivity_summary(sensitivity_df)
            
            print(f"\nHyperparameter sensitivity analysis saved to: {sensitivity_path}")
        else:
            print("No hyperparameter sensitivity data available (need multiple configs per model).")
    
    # Run hyperparameter sensitivity analysis if applicable
    if analysis_mode in ['hyperparameter_sensitivity', 'both']:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        sensitivity_df = analyze_hyperparameter_sensitivity(results)
        
        if not sensitivity_df.empty:
            # Save sensitivity analysis
            sensitivity_path = saver.output_dir / 'hyperparameter_sensitivity.csv'
            sensitivity_df.to_csv(sensitivity_path, index=False)
            
            # Print summary
            print_hyperparameter_sensitivity_summary(sensitivity_df)
            
            print(f"\nHyperparameter sensitivity analysis saved to: {sensitivity_path}")
        else:
            print("No hyperparameter sensitivity data available (need multiple configs per model).")
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {experiment_name}")
    print(f"{'='*80}")
    print(f"\nResults saved to: {saver.output_dir}")
    print_summary(results)
    
    return results, evaluation_config


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
              f"{result['mae_mean']:<10.2f} {result['rmse_mean']:<10.2f}")


class EngineeredFeatureFilterAdapter:
    """Adapter wrapper that filters engineered features after they're created."""
    
    def __init__(self, base_adapter, target_features: List[str]):
        self.base_adapter = base_adapter
        self.target_features = target_features
        
    def fit(self, *args, **kwargs):
        return self.base_adapter.fit(*args, **kwargs)
    
    def predict(self, data):
        return self.base_adapter.predict(data)
    
    def cross_validate(self, *args, **kwargs):
        return self.base_adapter.cross_validate(*args, **kwargs)
    
    def _dataloader_to_arrays(self, dataloader):
        # Get full feature arrays from base adapter
        X_full, y = self.base_adapter._dataloader_to_arrays(dataloader)
        
        # Get feature names from base adapter 
        if hasattr(self.base_adapter, 'get_feature_names'):
            all_feature_names = self.base_adapter.get_feature_names()
            
            if all_feature_names and len(all_feature_names) == X_full.shape[1]:
                # Find indices of target features
                target_indices = []
                for target_feature in self.target_features:
                    if target_feature in all_feature_names:
                        target_indices.append(all_feature_names.index(target_feature))
                
                if target_indices:
                    print(f"🔧 Filtering to {len(target_indices)} engineered features: {self.target_features}")
                    X_filtered = X_full[:, target_indices]
                    return X_filtered, y
                else:
                    print(f"⚠️  No target features found in engineered features, using all")
                    return X_full, y
            else:
                print(f"⚠️  Feature name mismatch, using all features")
                return X_full, y
        else:
            print(f"⚠️  No feature names available, using all features")
            return X_full, y
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base adapter."""
        return getattr(self.base_adapter, name)


def analyze_hyperparameter_sensitivity(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Analyze hyperparameter sensitivity across model configurations."""
    
    print("\n" + "🔍 HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    sensitivity_results = []
    
    # Group results by base model
    base_models = {}
    for model_name, result in results_data.items():
        if 'error' in result:
            continue  # Skip failed models
            
        # Determine base model from name or stored attribute
        if 'base_model' in result:
            base_model = result['base_model']
        elif '_hp_' in model_name:
            base_model = model_name.split('_hp_')[0]
        elif '_default' in model_name:
            base_model = model_name.replace('_default', '')
        else:
            # Use full model name as base for single configs
            base_model = model_name
        
        if base_model not in base_models:
            base_models[base_model] = []
        base_models[base_model].append((model_name, result))
    
    # Compute sensitivity metrics for each base model
    for base_model, model_results in base_models.items():
        if len(model_results) < 2:
            # Need at least 2 configurations for meaningful sensitivity analysis
            continue
            
        mae_values = [result['mae_mean'] for _, result in model_results]
        rmse_values = [result['rmse_mean'] for _, result in model_results]
        
        # Calculate sensitivity metrics
        mae_mean = np.mean(mae_values)
        mae_std = np.std(mae_values)
        mae_range = max(mae_values) - min(mae_values)
        coefficient_of_variation = mae_std / mae_mean if mae_mean > 0 else 0
        
        # Find best and worst configurations
        best_idx = np.argmin(mae_values)
        worst_idx = np.argmax(mae_values)
        best_model, best_result = model_results[best_idx]
        worst_model, worst_result = model_results[worst_idx]
        
        sensitivity_metrics = {
            'base_model': base_model,
            'n_configurations': len(mae_values),
            'mae_mean_across_configs': mae_mean,
            'mae_std_across_configs': mae_std,
            'mae_range': mae_range,
            'coefficient_of_variation': coefficient_of_variation,
            'best_mae': min(mae_values),
            'worst_mae': max(mae_values),
            'sensitivity_ratio': max(mae_values) / min(mae_values) if min(mae_values) > 0 else np.inf,
            'best_config': best_result.get('hyperparams', 'unknown'),
            'best_model_name': best_model,
            'worst_config': worst_result.get('hyperparams', 'unknown'),
            'worst_model_name': worst_model,
            'rmse_std_across_configs': np.std(rmse_values)
        }
        
        sensitivity_results.append(sensitivity_metrics)
    
    # Create DataFrame and sort by coefficient of variation (lower = more robust)
    sensitivity_df = pd.DataFrame(sensitivity_results)
    if not sensitivity_df.empty:
        sensitivity_df = sensitivity_df.sort_values('coefficient_of_variation')
        
        # Round numerical columns
        numerical_cols = ['mae_mean_across_configs', 'mae_std_across_configs', 'mae_range', 
                         'coefficient_of_variation', 'best_mae', 'worst_mae', 'sensitivity_ratio',
                         'rmse_std_across_configs']
        for col in numerical_cols:
            if col in sensitivity_df.columns:
                sensitivity_df[col] = sensitivity_df[col].round(3)
    
    return sensitivity_df


def print_hyperparameter_sensitivity_summary(sensitivity_df: pd.DataFrame):
    """Print a summary of hyperparameter sensitivity analysis."""
    
    if sensitivity_df.empty:
        print("No hyperparameter sensitivity data available.")
        return
    
    print("\nHyperparameter Sensitivity Summary:")
    print("=" * 60)
    print(f"{'Model':<15} {'Configs':<8} {'CV':<8} {'Range':<8} {'Best MAE':<10}")
    print("-" * 60)
    
    for _, row in sensitivity_df.iterrows():
        print(f"{row['base_model']:<15} {row['n_configurations']:<8} "
              f"{row['coefficient_of_variation']:<8.3f} {row['mae_range']:<8.3f} "
              f"{row['best_mae']:<10.3f}")
    
    # Highlight most and least sensitive models
    if len(sensitivity_df) > 0:
        most_robust = sensitivity_df.iloc[0]  # Lowest CV
        least_robust = sensitivity_df.iloc[-1]  # Highest CV
        
        print(f"\n🏆 Most Robust Model: {most_robust['base_model']}")
        print(f"   Coefficient of Variation: {most_robust['coefficient_of_variation']:.3f}")
        print(f"   Best configuration: {most_robust['best_config']}")
        
        print(f"\n⚠️  Most Sensitive Model: {least_robust['base_model']}")
        print(f"   Coefficient of Variation: {least_robust['coefficient_of_variation']:.3f}")
        print(f"   Performance range: {least_robust['mae_range']:.3f}")
        
        # Overall insights
        avg_cv = sensitivity_df['coefficient_of_variation'].mean()
        print(f"\n📊 Overall Insights:")
        print(f"   Average coefficient of variation: {avg_cv:.3f}")
        
        robust_threshold = 0.1  # CV < 0.1 considered robust
        robust_models = sensitivity_df[sensitivity_df['coefficient_of_variation'] < robust_threshold]
        print(f"   Models with CV < {robust_threshold}: {len(robust_models)}/{len(sensitivity_df)}")
        
        if len(robust_models) > 0:
            print(f"   Robust models: {', '.join(robust_models['base_model'].tolist())}")


def run_ablation_study():
    """Run ablation study with user's specific engineered features."""
    
    print("\n" + "🔬 ABLATION STUDY: TOP 5 ENGINEERED FEATURES")
    print("=" * 80)
    
    # Ablation study configuration
    ablation_config = {
        'dataset_name': 'rolling_new',
        'goal_type': 'minutes',
        'window_size': 6,  # Use optimal window size from our analysis
        'feature_set': 'top_5_selected',  # Your engineered features
        'cv_config': 'standard',
        'model_set': 'all'  # Test all models
    }
    
    print("Configuration:")
    print(f"  Dataset: {ablation_config['dataset_name']}")
    print(f"  Goal: {ablation_config['goal_type']}")
    print(f"  Window size: {ablation_config['window_size']}")
    print(f"  Feature set: {ablation_config['feature_set']}")
    print(f"  Features: {DEFAULT_EXPERIMENT_CONFIG.feature_sets['top_5_selected']}")
    print(f"  CV: {ablation_config['cv_config']}")
    print(f"  Models: {ablation_config['model_set']}")
    
    # Run the experiment
    results, config = run_evaluation_with_predictions(
        experiment_config=ablation_config,
        config_obj=DEFAULT_EXPERIMENT_CONFIG
    )
    
    # Additional analysis
    print("\n" + "📊 ABLATION RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        # Sort by MAE performance
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if 'mae_mean' in v],
            key=lambda x: x[1]['mae_mean']
        )
        
        print(f"\nTop 5 performing models with engineered features:")
        for i, (model_name, result) in enumerate(sorted_results[:5], 1):
            print(f"{i}. {model_name}: MAE={result['mae_mean']:.2f}±{result['mae_std']:.2f}")
        
        # Compare with baseline
        baseline_models = ['average_all', 'naive_forecast', 'adams_baseline_50']
        baseline_results = [(name, results[name]) for name in baseline_models if name in results and 'mae_mean' in results[name]]
        
        if baseline_results:
            print(f"\nBaseline comparisons:")
            for name, result in baseline_results:
                print(f"  {name}: MAE={result['mae_mean']:.2f}±{result['mae_std']:.2f}")
        
        # Best model improvement
        if sorted_results:
            best_model, best_result = sorted_results[0]
            print(f"\n🏆 Best model: {best_model}")
            print(f"   MAE: {best_result['mae_mean']:.2f}±{best_result['mae_std']:.2f}")
            
            if baseline_results:
                baseline_mae = min(r[1]['mae_mean'] for r in baseline_results)
                improvement = ((baseline_mae - best_result['mae_mean']) / baseline_mae) * 100
                print(f"   Improvement over baseline: {improvement:.1f}%")
    
    return results, config


if __name__ == "__main__":
    # Run ablation study with user's specific engineered features
    print("🚀 Running ablation study with engineered features...")
    run_ablation_study()
    
    # Optional: Run additional experiments
    # Uncomment below to run window size analysis
    """
    # Run evaluation for minutes_per_week with different window sizes
    for window_size in [8]:  # Start with window size 8
        print(f"\n\n{'#'*80}")
        print(f"# WINDOW SIZE: {window_size}")
        print(f"{'#'*80}")
        
        # Define a dummy experiment_config for the main loop
        # In a real scenario, you'd load this from a config file or pass it as an argument
        dummy_experiment_config = {
            'dataset_name': 'rolling_new',
            'goal_type': 'minutes',
            'window_size': window_size,
            'feature_set': 'all', # Or a specific set like 'top_5_temporal'
            'cv_config': 'standard',
            'model_set': 'all' # Or a specific set like 'baselines_only'
        }
        
        run_evaluation_with_predictions(
            experiment_config=dummy_experiment_config,
            config_obj=DEFAULT_EXPERIMENT_CONFIG
        )
    """ 