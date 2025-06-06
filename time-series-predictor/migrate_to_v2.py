#!/usr/bin/env python3
"""
Migration Helper: Legacy Framework -> Framework V2

This script helps migrate existing sklearn models and workflows
from the legacy framework to the new scalable Framework V2.
"""

import sys
import os
from pathlib import Path

# Add both frameworks to path
sys.path.append(str(Path(__file__).parent / 'framework_v2'))
sys.path.append(str(Path(__file__).parent / 'model'))

def migrate_sklearn_model_example():
    """Show how to migrate a sklearn model from legacy to new framework."""
    
    print("🔄 MIGRATION EXAMPLE: SKLearn Model")
    print("="*50)
    
    print("\n📝 OLD WAY (Legacy Framework):")
    print("""
    from framework import TimeSeriesFramework
    from sklearn.linear_model import LinearRegression
    
    # Legacy approach
    predictor = LinearRegression()
    framework = TimeSeriesFramework(predictor, lag_window=5)
    results = framework.cross_validate(
        data_path='data/data_tidied.csv',
        n_splits=5,
        test_size=1
    )
    """)
    
    print("\n✨ NEW WAY (Framework V2):")
    print("""
    from framework_v2 import StudentTimeSeriesDataset, SKLearnAdapter, CrossValidator
    from sklearn.linear_model import LinearRegression
    
    # New scalable approach
    dataset = StudentTimeSeriesDataset('data/data_tidied.csv', sequence_length=5)
    model = SKLearnAdapter(LinearRegression(), lag_window=5)
    cv = CrossValidator(model, dataset)
    results = cv.cross_validate(n_splits=5, test_size=1)
    """)
    
    print("\n🎯 BENEFITS OF NEW APPROACH:")
    print("   ✅ Supports both sklearn AND PyTorch models")
    print("   ✅ GPU acceleration for deep learning")
    print("   ✅ Streaming data processing (scales to large datasets)")
    print("   ✅ Professional ML workflows (early stopping, validation)")
    print("   ✅ Model save/load functionality")
    print("   ✅ Better error handling and logging")


def migrate_custom_model_example():
    """Show how to create a custom model in the new framework."""
    
    print("\n🔄 MIGRATION EXAMPLE: Custom Model")
    print("="*50)
    
    print("\n📝 OLD WAY (Legacy Framework):")
    print("""
    # Custom model had to implement sklearn interface manually
    class MyCustomModel:
        def fit(self, X, y): ...
        def predict(self, X): ...
        def get_params(self): ...
    
    framework = TimeSeriesFramework(MyCustomModel())
    """)
    
    print("\n✨ NEW WAY (Framework V2):")
    print("""
    # Option 1: Wrap existing sklearn-style model
    from framework_v2 import SKLearnAdapter
    model = SKLearnAdapter(MyCustomModel())
    
    # Option 2: Create PyTorch model for deep learning
    import torch.nn as nn
    from framework_v2 import PyTorchAdapter
    
    class MyNeuralNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = PyTorchAdapter(MyNeuralNet())
    """)


def run_side_by_side_comparison():
    """Run the same model in both frameworks for comparison."""
    
    print("\n🏁 SIDE-BY-SIDE COMPARISON")
    print("="*50)
    
    from sklearn.linear_model import LinearRegression
    
    # Test legacy framework
    print("🔍 Testing Legacy Framework...")
    try:
        from framework import TimeSeriesFramework
        legacy_model = TimeSeriesFramework(LinearRegression(), lag_window=5)
        legacy_results = legacy_model.cross_validate(
            data_path='data/data_tidied.csv',
            n_splits=3,
            test_size=1
        )
        print(f"   ✅ Legacy MAE: {legacy_results['mae_mean']:.3f} ± {legacy_results['mae_std']:.3f}")
    except Exception as e:
        print(f"   ❌ Legacy framework error: {e}")
        legacy_results = None
    
    # Test new framework
    print("🔍 Testing New Framework...")
    try:
        from framework_v2 import StudentTimeSeriesDataset, SKLearnAdapter, CrossValidator
        
        dataset = StudentTimeSeriesDataset('data/data_tidied.csv', sequence_length=5)
        model = SKLearnAdapter(LinearRegression(), lag_window=5)
        cv = CrossValidator(model, dataset)
        new_results = cv.cross_validate(n_splits=3, test_size=1)
        
        print(f"   ✅ New MAE: {new_results['mae_mean']:.3f} ± {new_results['mae_std']:.3f}")
        
        # Compare results
        if legacy_results:
            mae_diff = abs(legacy_results['mae_mean'] - new_results['mae_mean'])
            print(f"   📊 MAE difference: {mae_diff:.3f} (should be minimal)")
            
            if mae_diff < 0.01:
                print("   ✅ Results match - migration successful!")
            else:
                print("   ⚠️  Results differ - may need investigation")
        
    except Exception as e:
        print(f"   ❌ New framework error: {e}")


def migration_checklist():
    """Provide a checklist for migration."""
    
    print("\n📋 MIGRATION CHECKLIST")
    print("="*50)
    
    checklist = [
        "Install PyTorch (optional, for deep learning): pip install torch",
        "Update imports: from framework_v2 import ...",
        "Replace TimeSeriesFramework with SKLearnAdapter + CrossValidator",
        "Create StudentTimeSeriesDataset for your data",
        "Test that results match legacy framework",
        "Optionally: Explore PyTorch models for better performance",
        "Update any custom models to use new interfaces",
        "Update any scripts/notebooks to use new framework"
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"   {i}. {item}")
    
    print("\n🔗 HELPFUL RESOURCES:")
    print("   📖 Run: python demo_framework_v2.py")
    print("   📖 Check: framework_v2/README.md (when available)")
    print("   📖 Examples: framework_v2/examples/ (when available)")


def main():
    """Run migration guide."""
    
    print("🚀 FRAMEWORK V2 MIGRATION GUIDE")
    print("🎯 Transitioning from Legacy to Scalable Framework")
    print()
    
    # Show migration examples
    migrate_sklearn_model_example()
    migrate_custom_model_example()
    
    # Run side-by-side comparison
    run_side_by_side_comparison()
    
    # Provide checklist
    migration_checklist()
    
    print("\n🎉 Migration guide completed!")
    print("💡 Run 'python demo_framework_v2.py' to see the new framework in action")


if __name__ == "__main__":
    main() 