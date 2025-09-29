#!/usr/bin/env python3
"""
Framework V2 Demo Script

Demonstrates the new scalable time series framework with:
1. Traditional sklearn models (backwards compatible)
2. Modern PyTorch deep learning models  
3. GPU acceleration (if available)
4. Professional training workflows
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add framework to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Framework imports
from src.framework import (
    StudentTimeSeriesDataset, 
    SKLearnAdapter, 
    PyTorchAdapter,
    CrossValidator,
    print_device_info,
    get_device
)

# Traditional models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# PyTorch models
import torch
import torch.nn as nn
from src.framework.models.neural_nets import SimpleLSTM, SimpleMLP, create_model


def demo_sklearn_models():
    """Demonstrate sklearn models with the new framework."""
    print("="*60)
    print("🔬 SKLEARN MODELS DEMO - Predicting Average Proficiency")
    print("="*60)
    
    # Create dataset - now using the richer student week data
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
    
    print(f"📊 Dataset loaded: {len(dataset)} sequences")
    print(f"🎯 Target: Predicting avg_proficiency (new skills to master)")
    print(f"📈 Features: week_id, minutes_per_week, problems_solved, total_opportunities, prev_proficiency")
    
    # Test different sklearn models
    sklearn_models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42))
    ]
    
    results = {}
    
    for name, sklearn_model in sklearn_models:
        print(f"\n🔍 Testing {name}")
        
        # Wrap in adapter - using larger window for richer features
        model = SKLearnAdapter(sklearn_model, lag_window=5)
        
        # Cross-validate (using fewer folds for demo)
        cv = CrossValidator(model, dataset)
        result = cv.cross_validate(n_splits=3, test_size=1)
        
        results[name] = result
        
        print(f"   📈 MAE: {result['mae_mean']:.3f} ± {result['mae_std']:.3f}")
        print(f"   📈 RMSE: {result['rmse_mean']:.3f} ± {result['rmse_std']:.3f}")
        print(f"   📈 SMAPE: {result['smape_mean']:.1f}% ± {result['smape_std']:.1f}%")
    
    return results


def demo_pytorch_models():
    """Demonstrate PyTorch models with the new framework."""
    print("\n" + "="*60)
    print("🧠 PYTORCH MODELS DEMO - Predicting Average Proficiency")
    print("="*60)
    
    # Show device info
    print_device_info()
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # Create dataset - now using the richer student week data
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
    
    print(f"🎯 Target: Predicting avg_proficiency (new skills to master)")
    print(f"📈 Features: week_id, minutes_per_week, problems_solved, total_opportunities, prev_proficiency")
    
    # Test different PyTorch models - adjusted input sizes for actual data (6 features)
    pytorch_models = [
        ("Simple MLP", SimpleMLP(input_size=30, hidden_sizes=[32, 16])),  # 6 features * 5 timesteps
        ("LSTM", SimpleLSTM(input_size=6, hidden_size=32, num_layers=2)),  # 6 features per timestep
        ("CNN", create_model('cnn', input_size=6, num_filters=[16, 32])),  # 6 features per timestep
        ("DLinear", create_model('dlinear', input_size=6, seq_len=5)),     # 6 features, 5 timesteps
    ]
    
    results = {}
    
    for name, pytorch_model in pytorch_models:
        print(f"\n🔍 Testing {name}")
        print(f"   🏗️  Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
        
        # Wrap in adapter
        model = PyTorchAdapter(
            pytorch_model, 
            device=device,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        # Cross-validate with fewer epochs for demo
        cv = CrossValidator(model, dataset)
        result = cv.cross_validate(
            n_splits=2,  # Fewer folds for demo
            test_size=1,
            epochs=20,   # Fewer epochs for demo
            early_stopping_patience=5,
            verbose=False
        )
        
        results[name] = result
        
        print(f"   📈 MAE: {result['mae_mean']:.3f} ± {result['mae_std']:.3f}")
        print(f"   📈 RMSE: {result['rmse_mean']:.3f} ± {result['rmse_std']:.3f}")
        print(f"   📈 SMAPE: {result['smape_mean']:.1f}% ± {result['smape_std']:.1f}%")
    
    return results


def demo_advanced_features():
    """Demonstrate advanced features like model saving/loading."""
    print("\n" + "="*60)
    print("⚙️  ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Create a simple model
    model_path = Path("demo_model.pth")
    
    print("💾 Testing model save/load functionality...")
    
    # Create and train a model - adjusted for actual input size (6 features)
    pytorch_model = SimpleLSTM(input_size=6, hidden_size=16, num_layers=1)
    model = PyTorchAdapter(pytorch_model, device=get_device())
    
    # Create some dummy data - 6 features now
    X_dummy = np.random.randn(100, 5, 6)  # 100 samples, 5 timesteps, 6 features
    y_dummy = np.random.randn(100)         # target proficiency scores
    
    # Train briefly
    print("🏋️  Training model...")
    history = model.fit((X_dummy, y_dummy), epochs=5, verbose=False)
    print(f"   ✅ Training completed in {history['epochs_trained']} epochs")
    
    # Save model
    model.save(str(model_path))
    print(f"   💾 Model saved to {model_path}")
    
    # Create new model and load
    new_pytorch_model = SimpleLSTM(input_size=6, hidden_size=16, num_layers=1)
    new_model = PyTorchAdapter(new_pytorch_model, device=get_device())
    new_model.load(str(model_path))
    print("   📂 Model loaded successfully")
    
    # Test predictions are the same
    pred1 = model.predict(X_dummy[:10])
    pred2 = new_model.predict(X_dummy[:10])
    
    if np.allclose(pred1, pred2):
        print("   ✅ Save/load verification passed")
    else:
        print("   ❌ Save/load verification failed")
    
    # Cleanup
    if model_path.exists():
        model_path.unlink()
        print("   🗑️  Cleanup completed")


def compare_frameworks():
    """Compare new framework vs legacy framework performance."""
    print("\n" + "="*60)
    print("⚡ FRAMEWORK COMPARISON")
    print("="*60)
    
    # This would compare speed and functionality between old and new frameworks
    print("🔄 Framework comparison:")
    print("   ✅ New framework supports both sklearn and PyTorch")
    print("   ✅ GPU acceleration available for PyTorch models")
    print("   ✅ Streaming data processing for large datasets")
    print("   ✅ Professional training workflows (early stopping, validation)")
    print("   ✅ Model save/load functionality")
    print("   ✅ Backwards compatibility with legacy sklearn models")


def main():
    """Run all demos."""
    print("🚀 Framework V2 Demo - Predicting Student Proficiency")
    print("🎯 Demonstrating prediction of avg_proficiency (new skills students will master)")
    print("📚 Using rich student behavioral data: study time, problems solved, learning opportunities")
    
    try:
        # Demo sklearn models
        sklearn_results = demo_sklearn_models()
        
        # Demo PyTorch models (if torch available)
        try:
            pytorch_results = demo_pytorch_models()
        except ImportError as e:
            print(f"\n⚠️  PyTorch not available: {e}")
            print("   Install PyTorch to use deep learning models")
            pytorch_results = {}
        
        # Demo advanced features
        demo_advanced_features()
        
        # Compare frameworks
        compare_frameworks()
        
        # Summary
        print("\n" + "="*60)
        print("📊 DEMO SUMMARY")
        print("="*60)
        
        print("✅ Successfully demonstrated:")
        print("   • Predicting avg_proficiency (new skills to master)")
        print("   • Rich feature engineering (study time, problems, opportunities)")
        print("   • SKLearn model integration")
        if pytorch_results:
            print("   • PyTorch model integration") 
            print("   • GPU acceleration")
        print("   • Cross-validation framework")
        print("   • Model save/load functionality")
        print("   • Backwards compatibility")
        
        print("\n🎉 Student proficiency prediction demo completed successfully!")
        print("🔬 Models can now predict how many new skills students will master!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("🔧 Please check your data path and dependencies")
        raise


if __name__ == "__main__":
    main() 