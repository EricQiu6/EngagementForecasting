#!/usr/bin/env python3
"""
Custom Window Size Analysis Script
=================================

Runs window size analysis with specific parameters:
- Window range: [1, 6, 11, 16, 21, 26] (step size 5) 
- Goal: minutes
- Models: all
"""

import sys
from pathlib import Path
from comprehensive_evaluation_with_saved_predictions import run_window_size_analysis, DEFAULT_EXPERIMENT_CONFIG

def main():
    """Run custom window size analysis."""
    
    print("🚀 Starting custom window size analysis...")
    print("=" * 80)
    print("Configuration:")
    print("  Window range: [1, 6, 11, 16, 21, 26] (step size 5)")
    print("  Goal: minutes")
    print("  Models: all")
    print("  Dataset: rolling_new")
    print("=" * 80)
    
    # Define custom window sizes
    window_sizes = [1, 6, 11, 16, 21, 26]
    
    # Run the analysis
    try:
        results = run_window_size_analysis(
            dataset_name='rolling_new',
            goal_type='minutes',
            feature_set='all',
            window_sizes=window_sizes,
            model_set='all',  # Use all models instead of just top_performers
            cv_config='standard',
            config_obj=DEFAULT_EXPERIMENT_CONFIG
        )
        
        print("\n" + "=" * 80)
        print("🎉 WINDOW SIZE ANALYSIS COMPLETE!")
        print("=" * 80)
        
        # Print summary results
        if results and 'results_by_window' in results:
            print("\nSummary of best models by window size:")
            print("-" * 60)
            
            for window_size in window_sizes:
                if window_size in results['results_by_window']:
                    window_results = results['results_by_window'][window_size]
                    if 'error' not in window_results:
                        # Find best model for this window
                        successful_models = {k: v for k, v in window_results.items() 
                                           if isinstance(v, dict) and 'mae_mean' in v}
                        if successful_models:
                            best_model = min(successful_models.items(), 
                                           key=lambda x: x[1]['mae_mean'])
                            print(f"Window {window_size:2d}: {best_model[0]:20s} "
                                  f"(MAE: {best_model[1]['mae_mean']:.3f})")
                        else:
                            print(f"Window {window_size:2d}: No successful models")
                    else:
                        print(f"Window {window_size:2d}: Error - {window_results['error']}")
                else:
                    print(f"Window {window_size:2d}: No results")
        
        print(f"\n📁 Results saved to: {DEFAULT_EXPERIMENT_CONFIG.output_base_dir}/window_size_analysis/")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 