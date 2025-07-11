#!/usr/bin/env python3
"""
Simple Mixed Effects Verification

This script clearly shows whether mixed effects models are working properly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Local imports
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SchemaBasedSKLearnAdapter
from src.framework.adapters.mixed_effects_sklearn_adapter import SchemaAwareMixedEffectsAdapter
from src.framework.core.base import CrossValidator


def verify_mixed_effects_works():
    """Simple test to verify mixed effects actually works."""
    
    print("MIXED EFFECTS VERIFICATION")
    print("=" * 50)
    
    # Load data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    schema = get_schema('time_goal_extended_universal')
    
    # Create dataset
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    print(f"Dataset: {len(dataset)} sequences")
    
    # Test mixed effects adapter directly
    adapter = SchemaAwareMixedEffectsAdapter(
        sklearn_model=None,
        schema=schema,
        lag_window=5
    )
    
    # Simple single test
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    print(f"\nFitting mixed effects model...")
    fit_results = adapter.fit(dataloader)
    
    # Check results
    print(f"Fit status: {fit_results['status']}")
    print(f"Students found: {fit_results.get('n_students', 0)}")
    
    # Check if student effects were learned
    if hasattr(adapter, 'student_effects') and adapter.student_effects:
        n_students = len(adapter.student_effects)
        effect_values = list(adapter.student_effects.values())
        effect_std = np.std(effect_values)
        
        print(f"\n✅ SUCCESS - Mixed effects working:")
        print(f"  - Students with effects: {n_students}")
        print(f"  - Effect variability: {effect_std:.3f}")
        print(f"  - Global mean: {adapter.global_mean:.3f}")
        
        # Show some examples
        print(f"\nExample student effects:")
        for i, (student_id, effect) in enumerate(list(adapter.student_effects.items())[:5]):
            predicted_mean = adapter.global_mean + effect
            print(f"  {student_id}: {effect:+.3f} → predicted mean: {predicted_mean:.3f}")
        
        return True
    else:
        print(f"\n❌ FAILED - No student effects learned")
        print(f"Mixed effects fell back to simplified mode")
        return False


def compare_predictions():
    """Compare mixed effects vs baseline to see if they're different."""
    
    print("\n" + "=" * 50)
    print("PREDICTION COMPARISON")
    print("=" * 50)
    
    # Load data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    schema = get_schema('time_goal_extended_universal')
    
    # Create dataset
    dataset = SchemaBasedTimeSeriesDataset(
        data_path=data_path,
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Create test split
    from torch.utils.data import DataLoader, random_split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Test baseline (just global mean)
    all_y = []
    for _, batch_y in train_loader:
        all_y.extend(batch_y.numpy().flatten())
    global_mean = np.mean(all_y)
    
    # Test mixed effects
    mixed_adapter = SchemaAwareMixedEffectsAdapter(
        sklearn_model=None,
        schema=schema,
        lag_window=5
    )
    mixed_adapter.fit(train_loader)
    
    # Make predictions on test set
    mixed_preds = mixed_adapter.predict(test_loader)
    baseline_preds = np.full(len(mixed_preds), global_mean)
    
    # Get actual test targets
    test_y = []
    for _, batch_y in test_loader:
        test_y.extend(batch_y.numpy().flatten())
    test_y = np.array(test_y)
    
    # Compare predictions
    mixed_mae = np.mean(np.abs(test_y - mixed_preds))
    baseline_mae = np.mean(np.abs(test_y - baseline_preds))
    
    # Check prediction variability
    mixed_std = np.std(mixed_preds)
    baseline_std = np.std(baseline_preds)
    
    print(f"Baseline (global mean) predictions:")
    print(f"  MAE: {baseline_mae:.3f}")
    print(f"  Prediction std: {baseline_std:.3f} (should be 0.0)")
    
    print(f"\nMixed effects predictions:")
    print(f"  MAE: {mixed_mae:.3f}")
    print(f"  Prediction std: {mixed_std:.3f} (should be > 0)")
    
    # Verify mixed effects is actually different
    if mixed_std > 1.0:  # Predictions should vary across students
        improvement = (baseline_mae - mixed_mae) / baseline_mae * 100
        print(f"\n✅ Mixed effects working correctly:")
        print(f"  - Predictions vary across students (std={mixed_std:.3f})")
        print(f"  - Performance vs baseline: {improvement:+.1f}%")
        return True
    else:
        print(f"\n❌ Mixed effects not working:")
        print(f"  - Predictions don't vary across students (std={mixed_std:.3f})")
        print(f"  - Likely falling back to global mean")
        return False


if __name__ == "__main__":
    print("Mixed Effects Verification Script")
    print("This will clearly show if mixed effects is working properly\n")
    
    try:
        # Test 1: Direct verification
        success1 = verify_mixed_effects_works()
        
        # Test 2: Prediction comparison
        success2 = compare_predictions()
        
        print("\n" + "=" * 50)
        print("FINAL VERDICT")
        print("=" * 50)
        
        if success1 and success2:
            print("✅ MIXED EFFECTS IS WORKING CORRECTLY")
            print("   - Student effects are learned")
            print("   - Predictions vary by student")
            print("   - Using true mixed effects, not global mean")
        else:
            print("❌ MIXED EFFECTS IS NOT WORKING")
            print("   - Falling back to simplified mode")
            print("   - Need to debug further")
        
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        import traceback
        traceback.print_exc() 