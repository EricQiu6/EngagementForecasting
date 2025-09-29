#!/usr/bin/env python3
"""
Demo script for Student Ability Models

Tests the models that incorporate student ability, learning rate, and week difficulty.
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
    PyTorchAdapter,
    CrossValidator,
    print_device_info,
    get_device
)

# Models
from src.framework.models.neural_nets import create_model


def demo_student_ability_models():
    """Demonstrate the student ability models."""
    print("="*60)
    print("🎓 STUDENT ABILITY MODELS DEMO")
    print("="*60)
    
    # Show device info
    print_device_info()
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # Create dataset with the new CSV file
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
    print(f"🎯 Target: avg_proficiency (new skills to master)")
    print(f"📈 Features: week_id, minutes, problems, opportunities, proficiency,")
    print(f"            n_skills, week_difficulty, student_ability, student_learning_rate")
    
    # Test both models
    models_to_test = [
        ("Student Ability Linear", create_model('student_ability_linear', history_window=5)),
        ("Student Ability Neural", create_model('student_ability_neural', history_window=5, hidden_size=32)),
    ]
    
    # Also compare with baseline models
    baseline_models = [
        ("Simple MLP (baseline)", create_model('mlp', input_size=45, hidden_sizes=[32, 16])),  # 9 features * 5 timesteps
        ("LSTM (baseline)", create_model('lstm', input_size=9, hidden_size=32, num_layers=2)),  # 9 features
    ]
    
    all_results = {}
    
    # Test student ability models
    print("\n" + "="*40)
    print("📚 STUDENT ABILITY MODELS")
    print("="*40)
    
    for name, pytorch_model in models_to_test:
        print(f"\n🔍 Testing {name}")
        print(f"   🏗️  Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
        
        # Wrap in adapter
        model = PyTorchAdapter(
            pytorch_model, 
            device=device,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        # Cross-validate
        cv = CrossValidator(model, dataset)
        result = cv.cross_validate(
            n_splits=2,  # Fewer folds for demo
            test_size=1,
            epochs=30,   # More epochs for convergence
            early_stopping_patience=5,
            verbose=False
        )
        
        all_results[name] = result
        
        print(f"   📈 MAE: {result['mae_mean']:.3f} ± {result['mae_std']:.3f}")
        print(f"   📈 RMSE: {result['rmse_mean']:.3f} ± {result['rmse_std']:.3f}")
        print(f"   📈 SMAPE: {result['smape_mean']:.1f}% ± {result['smape_std']:.1f}%")
        print(f"   📈 R²: {result['r2_mean']:.3f} ± {result['r2_std']:.3f}")
        
        # For linear model, show coefficients
        if "Linear" in name:
            print("\n   📊 Model Coefficients:")
            # Get the trained model instance from the adapter
            trained_model = model.model
            coeffs = trained_model.get_coefficients()
            
            print(f"      α (intercept): {coeffs['alpha']:.4f}")
            print(f"      β_a (ability): {coeffs['beta_ability']:.4f}")
            print(f"      β_l (learning rate): {coeffs['beta_learning_rate']:.4f}")
            
            for i in range(trained_model.history_window):
                print(f"      β_y[t-{i+1}] (past proficiency): {coeffs[f'beta_performance_lag{i+1}']:.4f}")
            
            for i in range(trained_model.history_window):
                print(f"      β_d[t-{i+1}] (past difficulty): {coeffs[f'beta_difficulty_lag{i+1}']:.4f}")
    
    # Test baseline models for comparison
    print("\n" + "="*40)
    print("📊 BASELINE MODELS (for comparison)")
    print("="*40)
    
    for name, pytorch_model in baseline_models:
        print(f"\n🔍 Testing {name}")
        print(f"   🏗️  Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
        
        # Wrap in adapter
        model = PyTorchAdapter(
            pytorch_model, 
            device=device,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        # Cross-validate
        cv = CrossValidator(model, dataset)
        result = cv.cross_validate(
            n_splits=2,
            test_size=1,
            epochs=20,
            early_stopping_patience=5,
            verbose=False
        )
        
        all_results[name] = result
        
        print(f"   📈 MAE: {result['mae_mean']:.3f} ± {result['mae_std']:.3f}")
        print(f"   📈 RMSE: {result['rmse_mean']:.3f} ± {result['rmse_std']:.3f}")
        print(f"   📈 SMAPE: {result['smape_mean']:.1f}% ± {result['smape_std']:.1f}%")
        print(f"   📈 R²: {result['r2_mean']:.3f} ± {result['r2_std']:.3f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("📊 SUMMARY COMPARISON")
    print("="*60)
    
    print("\nModel Performance Ranking (by MAE):")
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['mae_mean'])
    
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"{i}. {name}: MAE = {result['mae_mean']:.3f}, R² = {result['r2_mean']:.3f}")
    
    # Interpretation
    print("\n📝 Key Insights:")
    print("- Student ability and learning rate provide personalized predictions")
    print("- Week difficulty helps account for temporal variations")
    print("- The linear model provides interpretable coefficients")
    print("- The neural model can capture non-linear interactions")


def main():
    """Run the demo."""
    print("🚀 Student Ability Model Demo")
    print("🎯 Testing models with student-specific parameters")
    
    try:
        demo_student_ability_models()
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("🔧 Please check your data path and dependencies")
        raise


if __name__ == "__main__":
    main() 