#!/usr/bin/env python3
"""
Student ID integration utilities for different model types.

This module provides recommendations and utilities for incorporating student ID 
features across different model architectures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from ..core.schema import DataSchema, StudentIDStrategy


def get_recommended_student_id_strategy(model_type: str, n_students: int, 
                                       data_per_student: float = None) -> StudentIDStrategy:
    """
    Get recommended student ID strategy based on model type and data characteristics.
    
    Args:
        model_type: Type of model ('linear', 'ridge', 'tree', 'forest', 'xgboost', 'neural', 'mixed_effects')
        n_students: Number of unique students in dataset
        data_per_student: Average number of observations per student
        
    Returns:
        Recommended StudentIDStrategy
    """
    
    # Decision logic based on model type and data characteristics
    if model_type in ['mixed_effects', 'hierarchical']:
        # Mixed effects models handle student ID as random effects
        return StudentIDStrategy('mixed_effects')
    
    elif model_type in ['neural', 'lstm', 'transformer', 'cnn', 'mlp']:
        # Neural networks can benefit from embeddings
        if n_students > 1000:
            # Too many students for one-hot, use embeddings
            return StudentIDStrategy('embeddings', embedding_dim=min(16, n_students // 10))
        elif n_students > 100:
            # Medium number of students, use target encoding
            return StudentIDStrategy('target_encoding')
        else:
            # Few students, can use one-hot
            return StudentIDStrategy('onehot')
    
    elif model_type in ['forest', 'random_forest', 'xgboost', 'gradient_boosting', 'tree']:
        # Tree-based models handle high-dimensional features well
        if n_students > 500:
            # Many students, use target encoding to avoid too many features
            return StudentIDStrategy('target_encoding')
        else:
            # Can handle one-hot encoding
            return StudentIDStrategy('onehot')
    
    elif model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
        # Linear models need regularization with many features
        if n_students > 100:
            # Too many features for linear models, use target encoding
            return StudentIDStrategy('target_encoding')
        else:
            # Can use one-hot with regularization
            return StudentIDStrategy('onehot')
    
    elif model_type in ['baseline', 'mean', 'median']:
        # Baseline models don't need student ID
        return StudentIDStrategy('none')
    
    else:
        # Default to target encoding for unknown models
        return StudentIDStrategy('target_encoding')


def create_student_aware_schema(base_schema: DataSchema, model_type: str, 
                               data_path: str = None, df: pd.DataFrame = None) -> DataSchema:
    """
    Create a student-aware schema with appropriate student ID strategy.
    
    Args:
        base_schema: Base schema without student ID integration
        model_type: Type of model that will use this schema
        data_path: Path to data file (if df not provided)
        df: DataFrame with data (if data_path not provided)
        
    Returns:
        Enhanced schema with student ID strategy
    """
    
    # Load data if not provided
    if df is None:
        if data_path is None:
            raise ValueError("Either df or data_path must be provided")
        df = pd.read_csv(data_path)
    
    # Get data characteristics
    n_students = df[base_schema.student_column].nunique()
    data_per_student = len(df) / n_students
    
    # Get recommended strategy
    strategy = get_recommended_student_id_strategy(model_type, n_students, data_per_student)
    
    # Create enhanced schema
    enhanced_schema = DataSchema(
        student_column=base_schema.student_column,
        time_column=base_schema.time_column,
        target_column=base_schema.target_column,
        feature_columns=base_schema.feature_columns.copy(),
        columns=base_schema.columns.copy(),
        student_id_strategy=strategy,
        time_format=base_schema.time_format,
        min_sequence_length=base_schema.min_sequence_length,
        validation_rules=base_schema.validation_rules.copy()
    )
    
    # Add strategy-specific feature columns
    if strategy.strategy_type != 'none':
        additional_features = strategy.get_additional_feature_columns(df, base_schema.student_column)
        enhanced_schema.feature_columns.extend(additional_features)
    
    return enhanced_schema


def analyze_student_id_benefit(df: pd.DataFrame, student_column: str, 
                              target_column: str, time_column: str) -> Dict[str, Any]:
    """
    Analyze whether incorporating student ID is likely to be beneficial.
    
    Args:
        df: DataFrame with student data
        student_column: Name of student ID column
        target_column: Name of target column
        time_column: Name of time column
        
    Returns:
        Dictionary with analysis results
    """
    
    # Basic statistics
    n_students = df[student_column].nunique()
    n_observations = len(df)
    avg_obs_per_student = n_observations / n_students
    
    # Calculate student-level variance
    student_means = df.groupby(student_column)[target_column].mean()
    global_mean = df[target_column].mean()
    
    # Between-student variance vs within-student variance
    between_student_var = np.var(student_means)
    within_student_var = df.groupby(student_column)[target_column].var().mean()
    
    # Intraclass correlation (ICC) - measures how much variance is between students
    total_var = df[target_column].var()
    icc = between_student_var / total_var if total_var > 0 else 0
    
    # Temporal consistency per student
    student_temporal_consistency = []
    for student in df[student_column].unique():
        student_data = df[df[student_column] == student].sort_values(time_column)
        if len(student_data) > 1:
            # Calculate autocorrelation
            values = student_data[target_column].values
            if len(values) > 1 and np.var(values) > 0:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                if not np.isnan(autocorr):
                    student_temporal_consistency.append(autocorr)
    
    avg_temporal_consistency = np.mean(student_temporal_consistency) if student_temporal_consistency else 0
    
    # Recommendations
    recommendations = []
    
    if icc > 0.3:
        recommendations.append("High between-student variance suggests student ID features would be beneficial")
    elif icc < 0.1:
        recommendations.append("Low between-student variance suggests student ID features may not help much")
    
    if avg_obs_per_student < 3:
        recommendations.append("Few observations per student - consider target encoding over one-hot")
    
    if n_students > 500:
        recommendations.append("Many students - avoid one-hot encoding, use target encoding or embeddings")
    
    if avg_temporal_consistency > 0.5:
        recommendations.append("High temporal consistency suggests student effects are important")
    
    return {
        'n_students': n_students,
        'n_observations': n_observations,
        'avg_obs_per_student': avg_obs_per_student,
        'between_student_variance': between_student_var,
        'within_student_variance': within_student_var,
        'intraclass_correlation': icc,
        'avg_temporal_consistency': avg_temporal_consistency,
        'recommendations': recommendations,
        'student_id_likely_beneficial': icc > 0.2 or avg_temporal_consistency > 0.3
    }


def get_model_type_from_sklearn_model(model) -> str:
    """
    Infer model type from sklearn model object.
    
    Args:
        model: sklearn model object
        
    Returns:
        Model type string
    """
    
    model_name = type(model).__name__.lower()
    
    if 'linear' in model_name:
        return 'linear'
    elif 'ridge' in model_name:
        return 'ridge'
    elif 'lasso' in model_name:
        return 'lasso'
    elif 'forest' in model_name:
        return 'forest'
    elif 'tree' in model_name:
        return 'tree'
    elif 'xgb' in model_name or 'boost' in model_name:
        return 'xgboost'
    elif 'svm' in model_name:
        return 'svm'
    else:
        return 'unknown'


# Model-specific utilities
class StudentAwareModelRecommender:
    """Recommends best student ID integration approach for different models."""
    
    def __init__(self, df: pd.DataFrame, student_column: str, target_column: str, time_column: str):
        self.df = df
        self.student_column = student_column
        self.target_column = target_column
        self.time_column = time_column
        self.analysis = analyze_student_id_benefit(df, student_column, target_column, time_column)
    
    def get_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Get model-specific recommendations for student ID integration."""
        
        recommendations = {}
        
        # Tree-based models
        recommendations['tree_based'] = {
            'models': ['RandomForest', 'XGBoost', 'GradientBoosting'],
            'recommended_strategy': 'target_encoding' if self.analysis['n_students'] > 100 else 'onehot',
            'rationale': 'Tree-based models can handle high-dimensional features but target encoding is more efficient with many students',
            'expected_benefit': 'High' if self.analysis['intraclass_correlation'] > 0.3 else 'Medium'
        }
        
        # Linear models
        recommendations['linear'] = {
            'models': ['LinearRegression', 'Ridge', 'Lasso'],
            'recommended_strategy': 'target_encoding' if self.analysis['n_students'] > 50 else 'onehot',
            'rationale': 'Linear models need regularization with many features, target encoding reduces dimensionality',
            'expected_benefit': 'High' if self.analysis['intraclass_correlation'] > 0.2 else 'Low'
        }
        
        # Neural networks
        recommendations['neural'] = {
            'models': ['MLP', 'LSTM', 'Transformer'],
            'recommended_strategy': 'embeddings' if self.analysis['n_students'] > 100 else 'target_encoding',
            'rationale': 'Neural networks can learn complex representations through embeddings',
            'expected_benefit': 'High' if self.analysis['intraclass_correlation'] > 0.3 else 'Medium'
        }
        
        # Mixed effects
        recommendations['mixed_effects'] = {
            'models': ['Mixed Effects Models'],
            'recommended_strategy': 'mixed_effects',
            'rationale': 'Mixed effects models naturally handle student-level heterogeneity',
            'expected_benefit': 'High' if self.analysis['intraclass_correlation'] > 0.2 else 'Medium'
        }
        
        return recommendations
    
    def print_summary(self):
        """Print a summary of the analysis and recommendations."""
        print("Student ID Integration Analysis")
        print("=" * 50)
        print(f"Dataset: {self.analysis['n_students']} students, {self.analysis['n_observations']} observations")
        print(f"Average observations per student: {self.analysis['avg_obs_per_student']:.1f}")
        print(f"Intraclass correlation: {self.analysis['intraclass_correlation']:.3f}")
        print(f"Temporal consistency: {self.analysis['avg_temporal_consistency']:.3f}")
        print(f"Student ID likely beneficial: {self.analysis['student_id_likely_beneficial']}")
        
        print("\nRecommendations:")
        for rec in self.analysis['recommendations']:
            print(f"  - {rec}")
        
        print("\nModel-Specific Recommendations:")
        recommendations = self.get_recommendations()
        for model_type, rec in recommendations.items():
            print(f"\n{model_type.title()} Models:")
            print(f"  Strategy: {rec['recommended_strategy']}")
            print(f"  Expected benefit: {rec['expected_benefit']}")
            print(f"  Rationale: {rec['rationale']}") 