#!/usr/bin/env python3
"""
Window Size Analysis Results Summary
===================================
"""

import json
import pandas as pd
from pathlib import Path

def analyze_window_results():
    """Analyze existing window size results and provide insights."""
    
    print("🔍 WINDOW SIZE ANALYSIS RESULTS")
    print("=" * 60)
    
    base_dir = Path("evaluation_outputs_with_features")
    
    # Window results we have so far
    results = {}
    
    # Window 1 results
    w1_file = base_dir / "rolling_new_minutes_w1_all_standard_all" / "overall_results.json"
    if w1_file.exists():
        with open(w1_file) as f:
            results[1] = json.load(f)
    
    # Window 6 results  
    w6_file = base_dir / "rolling_new_minutes_w6_all_standard_all" / "overall_results.json"
    if w6_file.exists():
        with open(w6_file) as f:
            results[6] = json.load(f)
    
    # Analyze each window
    for window_size, data in results.items():
        print(f"\n📊 WINDOW SIZE {window_size}")
        print("-" * 30)
        
        model_results = data['model_results']
        summary = data['summary_statistics']
        
        print(f"Total models: {summary['n_models']}")
        print(f"Successful models: {summary['n_successful']}")
        print(f"Success rate: {summary['n_successful']/summary['n_models']*100:.1f}%")
        
        if summary['n_successful'] > 0:
            print(f"Best model: {summary['best_model']} (MAE: {summary['mae_range'][0]:.3f})")
            print(f"Worst model: {summary['worst_model']} (MAE: {summary['mae_range'][1]:.3f})")
            
            # Create detailed ranking
            successful_models = []
            for model_name, model_data in model_results.items():
                if isinstance(model_data, dict) and 'mae_mean' in model_data:
                    successful_models.append({
                        'model': model_name,
                        'mae': model_data['mae_mean'],
                        'rmse': model_data['rmse_mean'],
                        'category': model_data['category'],
                        'training_time': model_data['training_time']
                    })
            
            # Sort by MAE
            successful_models.sort(key=lambda x: x['mae'])
            
            print(f"\nTop 5 Models:")
            for i, model in enumerate(successful_models[:5]):
                print(f"  {i+1}. {model['model']:15s} MAE={model['mae']:.3f} "
                      f"({model['category']:12s}) Time={model['training_time']:.1f}s")
        else:
            print("❌ All models failed!")
            # Show error types
            errors = set()
            for model_data in model_results.values():
                if isinstance(model_data, dict) and 'error' in model_data:
                    errors.add(model_data['error'])
            for error in errors:
                print(f"   Error: {error}")
    
    # Overall insights
    print(f"\n🎯 INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    if 1 in results and 6 in results:
        w1_success = results[1]['summary_statistics']['n_successful']
        w6_success = results[6]['summary_statistics']['n_successful']
        
        print(f"Window Size 1:")
        print(f"  - Success rate: {w1_success}/16 models ({w1_success/16*100:.1f}%)")
        print(f"  - Issue: Most models fail with 'SVD did not converge' errors")
        print(f"  - Problem: Window size 1 provides insufficient temporal context")
        
        print(f"\nWindow Size 6:")
        print(f"  - Success rate: {w6_success}/16 models ({w6_success/16*100:.1f}%)")
        print(f"  - Best model: {results[6]['summary_statistics']['best_model']} (MAE: {results[6]['summary_statistics']['mae_range'][0]:.3f})")
        print(f"  - Significant improvement over window size 1")
        
        # Compare best performing models between windows
        if w1_success > 0 and w6_success > 0:
            w1_best_mae = results[1]['summary_statistics']['mae_range'][0]
            w6_best_mae = results[6]['summary_statistics']['mae_range'][0] 
            improvement = (w1_best_mae - w6_best_mae) / w1_best_mae * 100
            
            print(f"\nPerformance Comparison:")
            print(f"  - Window 1 best MAE: {w1_best_mae:.3f}")
            print(f"  - Window 6 best MAE: {w6_best_mae:.3f}")
            print(f"  - Improvement: {improvement:.1f}% better with window 6")
    
    print(f"\n📋 NEXT STEPS")
    print("-" * 30)
    print("1. ⏳ Wait for remaining window sizes (11, 16, 21, 26) to complete")
    print("2. 📊 Run comprehensive analysis on the best performing window size")
    print("3. 🔍 Analyze feature importance patterns across window sizes")
    print("4. 🎯 Based on early results, window sizes 6+ show promise")
    print("5. 💡 Expect optimal window size to be in range 6-16 for this dataset")
    
    return results

if __name__ == "__main__":
    analyze_window_results() 