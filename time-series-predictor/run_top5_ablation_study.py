#!/usr/bin/env python3
"""
Top 5 Features Ablation Study
=============================

Run evaluation with just the top 5 most important features:
- minutes_mean
- current_minutes_per_week  
- minutes_std
- target_lag5
- current_week_id
"""

import sys
from pathlib import Path
from comprehensive_evaluation_with_saved_predictions import run_evaluation_with_predictions, DEFAULT_EXPERIMENT_CONFIG

def main():
    """Run top 5 features ablation study."""
    
    print("🎯 TOP 5 FEATURES ABLATION STUDY")
    print("=" * 80)
    print("Testing performance with only the top 5 most important features:")
    print("  1. minutes_mean")
    print("  2. current_minutes_per_week")
    print("  3. minutes_std")
    print("  4. target_lag5")
    print("  5. current_week_id")
    print("=" * 80)
    
    # Create custom configuration with top 5 features
    config = DEFAULT_EXPERIMENT_CONFIG
    
    # Add the custom feature set to the configuration
    config.feature_sets['top_5_selected'] = [
        'minutes_mean',
        'current_minutes_per_week', 
        'minutes_std',
        'target_lag5',
        'current_week_id'
    ]
    
    # Run evaluation with top 5 features
    experiment_config = {
        'dataset_name': 'rolling_new',
        'goal_type': 'minutes',
        'window_size': 6,  # Use window size 6 (good performance)
        'feature_set': 'top_5_selected',
        'cv_config': 'standard',
        'model_set': 'all'  # Test all models
    }
    
    print("🚀 Running evaluation with top 5 features...")
    
    try:
        results, eval_config = run_evaluation_with_predictions(experiment_config, config)
        
        print("\n" + "=" * 80)
        print("🎉 TOP 5 FEATURES EVALUATION COMPLETE!")
        print("=" * 80)
        
        if results:
            # Find best models
            successful_results = {k: v for k, v in results.items() if 'error' not in v}
            if successful_results:
                sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['mae_mean'])
                
                print("\n🏆 PERFORMANCE WITH TOP 5 FEATURES:")
                print("-" * 60)
                print(f"{'Rank':<5} {'Model':<20} {'MAE':<10} {'Category':<15}")
                print("-" * 60)
                
                for i, (model_name, result) in enumerate(sorted_results[:10], 1):
                    print(f"{i:<5} {model_name:<20} {result['mae_mean']:<10.3f} {result['category']:<15}")
                
                # Compare to baseline (all features)
                best_model = sorted_results[0]
                print(f"\n📊 COMPARISON:")
                print(f"Best model with top 5 features: {best_model[0]} (MAE: {best_model[1]['mae_mean']:.3f})")
                print(f"Best model with ALL features (w6): lasso (MAE: 8.188)")
                
                performance_change = best_model[1]['mae_mean'] - 8.188
                if performance_change > 0:
                    print(f"Performance degradation: +{performance_change:.3f} MAE ({performance_change/8.188*100:+.1f}%)")
                else:
                    print(f"Performance improvement: {performance_change:.3f} MAE ({performance_change/8.188*100:+.1f}%)")
                
                print(f"\n📁 Detailed results saved to: {eval_config['experiment_name']}")
                
                return results
            else:
                print("❌ No models succeeded with top 5 features")
        else:
            print("❌ No results returned")
    
    except Exception as e:
        print(f"❌ Ablation study failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_comparison_analysis():
    """Compare performance across different feature sets."""
    
    print("\n" + "🔍 FEATURE SET COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Feature sets to compare
    feature_sets = {
        'top_5_selected': [
            'minutes_mean', 'current_minutes_per_week', 'minutes_std', 
            'target_lag5', 'current_week_id'
        ],
        'top_5_temporal': [
            'week_id', 'weeks_since_start', 'is_first_week', 
            'temporal_gap_weeks', 'time_since_last_activity'
        ],
        'top_5_performance': [
            'avg_proficiency', 'total_correct', 'total_incorrect',
            'success_rate', 'avg_attempts_per_problem'
        ]
    }
    
    config = DEFAULT_EXPERIMENT_CONFIG
    
    # Add all feature sets to config
    for name, features in feature_sets.items():
        config.feature_sets[name] = features
    
    comparison_results = {}
    
    for feature_set_name in feature_sets.keys():
        print(f"\n{'='*60}")
        print(f"Testing feature set: {feature_set_name}")
        print(f"Features: {feature_sets[feature_set_name]}")
        print(f"{'='*60}")
        
        experiment_config = {
            'dataset_name': 'rolling_new',
            'goal_type': 'minutes',
            'window_size': 6,
            'feature_set': feature_set_name,
            'cv_config': 'standard',
            'model_set': 'linear_models'  # Focus on linear models for speed
        }
        
        try:
            results, eval_config = run_evaluation_with_predictions(experiment_config, config)
            if results:
                successful_results = {k: v for k, v in results.items() if 'error' not in v}
                if successful_results:
                    best_model = min(successful_results.items(), key=lambda x: x[1]['mae_mean'])
                    comparison_results[feature_set_name] = {
                        'best_model': best_model[0],
                        'best_mae': best_model[1]['mae_mean'],
                        'n_successful': len(successful_results)
                    }
                    print(f"✅ Best: {best_model[0]} (MAE: {best_model[1]['mae_mean']:.3f})")
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            comparison_results[feature_set_name] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\n📊 FEATURE SET COMPARISON SUMMARY")
    print("-" * 80)
    print(f"{'Feature Set':<20} {'Best Model':<15} {'MAE':<10} {'Status':<10}")
    print("-" * 80)
    
    for feature_set, result in comparison_results.items():
        if 'error' not in result:
            print(f"{feature_set:<20} {result['best_model']:<15} {result['best_mae']:<10.3f} {'✅':<10}")
        else:
            print(f"{feature_set:<20} {'Failed':<15} {'N/A':<10} {'❌':<10}")
    
    # Baseline comparison
    print(f"{'all_features (w6)':<20} {'lasso':<15} {'8.188':<10} {'✅ Baseline':<10}")
    
    return comparison_results

if __name__ == "__main__":
    # Run main ablation study
    main_results = main()
    
    # Run comparison analysis
    comparison_results = run_comparison_analysis()
    
    print(f"\n✅ Ablation study complete!")
    print(f"🎯 Key insight: How much performance do we lose with just 5 features vs all features?") 