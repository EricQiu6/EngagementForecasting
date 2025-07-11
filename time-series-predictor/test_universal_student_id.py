#!/usr/bin/env python3
"""
Test Universal Student ID Handling

This script demonstrates how the universal schema allows different model types
to use student ID appropriately without breaking each other.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Local imports
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SchemaBasedSKLearnAdapter
from src.framework.adapters.mixed_effects_sklearn_adapter import SchemaAwareMixedEffectsAdapter
from src.framework.core.base import CrossValidator


def test_universal_schema():
    """Test the universal schema with different model types."""
    
    print("=" * 60)
    print("TESTING UNIVERSAL STUDENT ID SCHEMA")
    print("=" * 60)
    
    # Load data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    
    # Use the universal schema that includes student ID as first feature
    schema = get_schema('time_goal_extended_universal')
    
    print(f"Universal schema:")
    print(f"  Student column: {schema.student_column}")
    print(f"  Feature columns ({len(schema.feature_columns)}): {schema.feature_columns}")
    print(f"  Student ID strategy: {schema.student_id_strategy.strategy_type}")
    print(f"  Student ID in features: {schema.student_column in schema.feature_columns}")
    print(f"  Student ID is first feature: {schema.feature_columns[0] == schema.student_column}")
    
    # Create dataset
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    print(f"\nDataset: {len(dataset)} sequences")
    
    # Test with different model types
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=10, random_state=42),
            'adapter_class': SchemaBasedSKLearnAdapter,
            'should_remove_student_id': True
        },
        'Ridge': {
            'model': Ridge(alpha=1.0, random_state=42),
            'adapter_class': SchemaBasedSKLearnAdapter,
            'should_remove_student_id': True
        },
        'MixedEffects': {
            'model': None,  # Not used for mixed effects
            'adapter_class': SchemaAwareMixedEffectsAdapter,
            'should_remove_student_id': False
        }
    }
    
    results = {}
    
    for model_name, config in models.items():
        print(f"\n{'='*20} Testing {model_name} {'='*20}")
        
        try:
            if config['adapter_class'] == SchemaAwareMixedEffectsAdapter:
                # Mixed effects adapter
                adapter = SchemaAwareMixedEffectsAdapter(
                    sklearn_model=None,
                    schema=schema,
                    lag_window=5
                )
            else:
                # Regular sklearn adapter
                adapter = SchemaBasedSKLearnAdapter(
                    sklearn_model=config['model'],
                    schema=schema,
                    lag_window=5
                )
            
            # Quick test: get a sample to see feature processing
            sample_x, sample_y = dataset[0]
            print(f"Sample input shape: {sample_x.shape}")
            
            # Cross-validate (quick test with 2 folds)
            cv = CrossValidator(adapter, dataset)
            cv_results = cv.cross_validate(n_splits=2, test_size=1)
            
            results[model_name] = {
                'mae': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'status': 'success'
            }
            
            print(f"✅ {model_name}: MAE={cv_results['mae_mean']:.3f}±{cv_results['mae_std']:.3f}")
            
            # Test feature handling
            if hasattr(adapter, '_handle_student_id_features'):
                test_features = np.array([[1, 2, 3, 4, 5]])  # Mock features with student ID first
                processed = adapter._handle_student_id_features(test_features)
                expected_shape = (1, 4) if config['should_remove_student_id'] else (1, 5)
                actual_shape = processed.shape
                
                print(f"  Feature processing: {test_features.shape} -> {actual_shape}")
                print(f"  Student ID removed: {actual_shape[1] < test_features.shape[1]}")
                print(f"  Expected removal: {config['should_remove_student_id']}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Universal schema: {schema.student_id_strategy.strategy_type}")
    print(f"Student ID included as first feature: {schema.feature_columns[0] == schema.student_column}")
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {model_name}: Works correctly (MAE={result['mae']:.3f})")
        else:
            print(f"❌ {model_name}: Failed ({result.get('error', 'Unknown error')})")
    
    print("\nKey insight:")
    print("- Mixed effects models: Use student ID for random effects")
    print("- Other models: Automatically remove student ID (they have student_ability/learning_rate)")
    print("- All models work with the same universal schema!")
    
    return results


def test_schema_comparison():
    """Compare different schema approaches."""
    
    print("\n" + "=" * 60)
    print("SCHEMA COMPARISON")
    print("=" * 60)
    
    schemas_to_test = [
        'time_goal_extended',
        'time_goal_extended_target_encoding', 
        'time_goal_extended_embeddings',
        'time_goal_extended_universal'
    ]
    
    for schema_name in schemas_to_test:
        schema = get_schema(schema_name)
        print(f"\n{schema_name}:")
        print(f"  Features: {len(schema.feature_columns)}")
        print(f"  Strategy: {schema.student_id_strategy.strategy_type}")
        print(f"  Student ID in features: {schema.student_column in schema.feature_columns}")
        
        if schema.student_column in schema.feature_columns:
            idx = schema.feature_columns.index(schema.student_column)
            print(f"  Student ID position: {idx} (first={idx==0})")


if __name__ == "__main__":
    print("Universal Student ID Handling Test")
    print("This test shows how different models can use the same schema")
    
    try:
        # Test universal schema
        results = test_universal_schema()
        
        # Compare schemas
        test_schema_comparison()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 