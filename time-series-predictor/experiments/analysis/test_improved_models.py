#!/usr/bin/env python3
"""
Test improved models that handle the zero-inflated nature of the data better.
"""

import sys
import os
from pathlib import Path

# Add framework to path
sys.path.append(str(Path(__file__).parent / 'framework_v2'))

from framework_v2 import (
    StudentTimeSeriesDataset, 
    PyTorchAdapter,
    CrossValidator,
    get_device
)
from framework_v2.models.neural_nets import create_model


def test_improved_models():
    """Test the improved models."""
    print("="*60)
    print("🚀 TESTING IMPROVED MODELS")
    print("="*60)
    
    device = get_device()
    
    # Create dataset
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    if not os.path.exists(data_path):
        data_path = '~/cmu/goalsetting-recommendation-algorithm/data-analysis/student_week_aggregations_rolling_new.csv'
    
    dataset = StudentTimeSeriesDataset(
        data_path=data_path,
        sequence_length=5,
        target_column='avg_proficiency',
        student_column='anon_student_id',
        time_column='week_id',
        load_in_memory=True
    )
    
    print(f"📊 Dataset: {len(dataset)} sequences")
    print(f"🎯 Target: avg_proficiency (33% zeros)")
    
    # Models to test
    models_to_test = [
        # Original models for comparison
        ("Student Ability Linear (baseline)", create_model('student_ability_linear', history_window=5)),
        
        # Improved models
        ("Zero-Inflated Poisson", create_model('zip', history_window=5)),
        ("Improved Student Model", create_model('improved_student', history_window=5)),
    ]
    
    results = {}
    
    for name, pytorch_model in models_to_test:
        print(f"\n🔍 Testing: {name}")
        print(f"   Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
        
        # Wrap in adapter
        model = PyTorchAdapter(
            pytorch_model, 
            device=device,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        # Quick test with fewer splits
        cv = CrossValidator(model, dataset)
        result = cv.cross_validate(
            n_splits=2,
            test_size=1,
            epochs=20,
            early_stopping_patience=5,
            verbose=False
        )
        
        results[name] = result
        
        print(f"   📈 MAE: {result['mae_mean']:.3f} ± {result['mae_std']:.3f}")
        print(f"   📈 RMSE: {result['rmse_mean']:.3f} ± {result['rmse_std']:.3f}")
        print(f"   📈 R²: {result['r2_mean']:.3f} ± {result['r2_std']:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    sorted_by_mae = sorted(results.items(), key=lambda x: x[1]['mae_mean'])
    for i, (name, result) in enumerate(sorted_by_mae, 1):
        print(f"{i}. {name}: MAE={result['mae_mean']:.3f}, R²={result['r2_mean']:.3f}")
    
    print("\n💡 Key Improvements:")
    print("- Zero-Inflated Poisson: Better handles the 33% zeros")
    print("- Improved features: Momentum and variance capture trends")
    print("- Better architecture: LayerNorm and proper regularization")


if __name__ == "__main__":
    test_improved_models() 