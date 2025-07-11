#!/usr/bin/env python3
"""
Demo: Student ID Integration with Different Model Types

This script demonstrates how to use the student ID integration features
to improve model performance across different architectures.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Local imports
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SchemaBasedSKLearnAdapter
from src.framework.core.base import CrossValidator
from src.framework.utils.student_id_utils import (
    StudentAwareModelRecommender,
    create_student_aware_schema,
    analyze_student_id_benefit,
    get_model_type_from_sklearn_model
)


def demo_student_id_analysis():
    """Analyze a dataset to understand student ID integration benefits."""
    
    print("=" * 60)
    print("STUDENT ID INTEGRATION ANALYSIS")
    print("=" * 60)
    
    # Load your data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset with {len(df)} observations")
    print(f"Students: {df['anon_student_id'].nunique()}")
    print(f"Time periods: {df['week_id'].nunique()}")
    
    # Create recommender
    recommender = StudentAwareModelRecommender(
        df=df,
        student_column='anon_student_id',
        target_column='minutes_per_week',
        time_column='week_id'
    )
    
    # Print analysis
    recommender.print_summary()
    
    return recommender


def demo_model_comparison():
    """Compare models with and without student ID features."""
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: WITH vs WITHOUT STUDENT ID")
    print("=" * 60)
    
    # Load data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    
    # Base schema (without student ID)
    base_schema = get_schema('time_goal_extended')
    
    # Create student-aware schemas for different model types
    forest_schema = create_student_aware_schema(base_schema, 'forest', data_path)
    ridge_schema = create_student_aware_schema(base_schema, 'ridge', data_path)
    
    print(f"Base schema features: {len(base_schema.feature_columns)}")
    print(f"Forest schema features: {len(forest_schema.feature_columns)}")
    print(f"Ridge schema features: {len(ridge_schema.feature_columns)}")
    
    # Models to compare
    models = {
        'RandomForest_baseline': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'schema': base_schema
        },
        'RandomForest_with_student_id': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'schema': forest_schema
        },
        'Ridge_baseline': {
            'model': Ridge(alpha=1.0, random_state=42),
            'schema': base_schema
        },
        'Ridge_with_student_id': {
            'model': Ridge(alpha=1.0, random_state=42),
            'schema': ridge_schema
        }
    }
    
    results = {}
    
    for model_name, config in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Create dataset
        dataset = SchemaBasedTimeSeriesDataset(
            data_path=data_path,
            schema=config['schema'],
            sequence_length=5,
            validate_data=False
        )
        
        # Create adapter
        adapter = SchemaBasedSKLearnAdapter(
            sklearn_model=config['model'],
            schema=config['schema'],
            lag_window=5
        )
        
        # Cross-validate
        cv = CrossValidator(adapter, dataset)
        cv_results = cv.cross_validate(n_splits=5, test_size=1)
        
        results[model_name] = {
            'mae': cv_results['mae_mean'],
            'mae_std': cv_results['mae_std'],
            'r2': cv_results.get('r2_mean', 0),
            'schema_type': config['schema'].student_id_strategy.strategy_type
        }
        
        print(f"  MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
        print(f"  Strategy: {config['schema'].student_id_strategy.strategy_type}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for model_type in ['RandomForest', 'Ridge']:
        baseline_key = f'{model_type}_baseline'
        student_key = f'{model_type}_with_student_id'
        
        if baseline_key in results and student_key in results:
            baseline_mae = results[baseline_key]['mae']
            student_mae = results[student_key]['mae']
            improvement = (baseline_mae - student_mae) / baseline_mae * 100
            
            print(f"\n{model_type}:")
            print(f"  Baseline MAE: {baseline_mae:.3f}")
            print(f"  With Student ID: {student_mae:.3f}")
            print(f"  Improvement: {improvement:+.1f}%")
            print(f"  Strategy: {results[student_key]['schema_type']}")
    
    return results


def demo_neural_network_embeddings():
    """Demonstrate student embeddings for neural networks."""
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK STUDENT EMBEDDINGS")
    print("=" * 60)
    
    # This would require implementing a neural network with embeddings
    # For now, just show the schema setup
    
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    base_schema = get_schema('time_goal_extended')
    
    # Create embedding-based schema
    embedding_schema = create_student_aware_schema(base_schema, 'neural', data_path)
    
    print(f"Base features: {base_schema.feature_columns}")
    print(f"Embedding features: {embedding_schema.feature_columns}")
    print(f"Strategy: {embedding_schema.student_id_strategy.strategy_type}")
    
    # Load data to see the numeric student IDs
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=embedding_schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Get a sample
    sample_x, sample_y = dataset[0]
    print(f"Sample input shape: {sample_x.shape}")
    print(f"Student ID numeric feature: {sample_x[-1, -1]}")  # Last feature should be student_id_numeric
    
    return embedding_schema


def demo_mixed_effects():
    """Demonstrate mixed effects model setup."""
    
    print("\n" + "=" * 60)
    print("MIXED EFFECTS MODEL SETUP")
    print("=" * 60)
    
    # Your existing mixed effects model already handles student IDs
    # This is just to show the recommended approach
    
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    base_schema = get_schema('time_goal_extended')
    
    mixed_effects_schema = create_student_aware_schema(base_schema, 'mixed_effects', data_path)
    
    print(f"Schema strategy: {mixed_effects_schema.student_id_strategy.strategy_type}")
    print("Mixed effects models handle student IDs as random effects internally")
    print("Features remain the same, but student ID is passed separately")
    
    return mixed_effects_schema


def demo_cross_validation_considerations():
    """Show important cross-validation considerations with student ID."""
    
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION CONSIDERATIONS")
    print("=" * 60)
    
    print("Current approach: TimeSeriesSplit (by time)")
    print("  - Pros: Respects temporal ordering")
    print("  - Cons: Students appear in both train and test")
    print("  - Good for: Predicting future performance of known students")
    
    print("\nAlternative: GroupKFold (by student)")
    print("  - Pros: Tests generalization to new students")
    print("  - Cons: May break temporal patterns")
    print("  - Good for: Generalization to new students")
    
    print("\nRecommendation for student ID features:")
    print("  - Use TimeSeriesSplit for temporal prediction tasks")
    print("  - Use GroupKFold for student generalization tasks")
    print("  - Consider using both for comprehensive evaluation")
    
    # Demo data leakage prevention
    print("\nData leakage prevention:")
    print("  - Target encoding: Use only past data for each prediction")
    print("  - Rolling window: Update statistics as time progresses")
    print("  - Stratified folds: Ensure balanced student representation")


if __name__ == "__main__":
    print("Student ID Integration Demo")
    print("This demo shows how to incorporate student ID features in different model types")
    
    # Run demos
    try:
        # 1. Analyze dataset
        recommender = demo_student_id_analysis()
        
        # 2. Compare models with/without student ID
        results = demo_model_comparison()
        
        # 3. Show neural network embeddings
        embedding_schema = demo_neural_network_embeddings()
        
        # 4. Show mixed effects setup
        mixed_effects_schema = demo_mixed_effects()
        
        # 5. Cross-validation considerations
        demo_cross_validation_considerations()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure the data file exists and dependencies are installed") 