#!/usr/bin/env python3
"""
Custom Experiment Runner
========================

Run experiment with specific user configuration:
- Dataset: data_splits_student_week_aggregations_rolling_new/[train|val].csv  
- Features: all features available
- Goal: minutes
- Window range: [1, 26], step size: 5

Note: Using val.csv as the test dataset since that's the actual filename in the directory.
"""

import sys
from pathlib import Path
from comprehensive_evaluation_with_saved_predictions import (
    ExperimentConfig, 
    DEFAULT_EXPERIMENT_CONFIG,
    run_window_size_analysis,
    run_evaluation_with_predictions
)


def setup_custom_config(dataset_type='train'):
    """Setup custom configuration for the user's experiment."""
    
    # Create a custom config based on the default
    custom_config = ExperimentConfig()
    
    # Add both train and validation/test datasets
    custom_config.available_datasets['train_split'] = 'data_splits_student_week_aggregations_rolling_new/train.csv'
    custom_config.available_datasets['test_split'] = 'data_splits_student_week_aggregations_rolling_new/val.csv'  # Using val.csv as test set
    
    # Update dataset directory to current directory since the path is relative
    custom_config.dataset_directory = './'
    
    # Add custom window size range: [1, 26] with step 5
    # This creates [1, 6, 11, 16, 21, 26]
    custom_window_sizes = list(range(1, 27, 5))
    custom_config.window_sizes['custom_range'] = custom_window_sizes
    
    # Add custom model set: all except neural networks
    custom_config.model_sets['all_except_neural'] = [
        # Baselines
        'average_all', 'naive_forecast', 'median_all', 'median_no_zeros', 'mean_no_zeros',
        # Goal-based
        'adams_baseline_50', 'adams_baseline_60', 'adams_baseline_70',
        # Linear
        'linear_regression', 'ridge', 'lasso',
        # Tree
        'random_forest', 'xgboost',
        # Mixed Effects
        'mixed_effects'
        # Note: Excludes 'mlp' and 'lstm' (neural models)
    ]
    
    # Set the dataset to use
    dataset_key = f'{dataset_type}_split'
    dataset_path = custom_config.available_datasets[dataset_key]
    
    print("✅ Custom configuration created:")
    print(f"   Available datasets:")
    print(f"     - train_split: {custom_config.available_datasets['train_split']}")
    print(f"     - test_split: {custom_config.available_datasets['test_split']} (val.csv)")
    print(f"   Using dataset: {dataset_key} -> {dataset_path}")
    print(f"   Window sizes: {custom_window_sizes}")
    print(f"   Models: all_except_neural (14 models, excludes MLP & LSTM)")
    
    return custom_config, dataset_key


def run_single_window_experiment(window_size: int, custom_config: ExperimentConfig, dataset_key: str = 'train_split'):
    """Run a single experiment with specified window size."""
    
    experiment_config = {
        'dataset_name': dataset_key,
        'goal_type': 'minutes', 
        'window_size': window_size,
        'feature_set': 'all',  # Use all available features
        'cv_config': 'standard',
        'model_set': 'all_except_neural'  # Use all models except neural networks (14 total)
    }
    
    print(f"\n{'='*60}")
    print(f"Running experiment with window size: {window_size}")
    print(f"{'='*60}")
    
    try:
        results, eval_config = run_evaluation_with_predictions(experiment_config, custom_config)
        
        # Print quick summary
        if results:
            successful_results = {k: v for k, v in results.items() if 'error' not in v}
            if successful_results:
                best_model = min(successful_results.items(), key=lambda x: x[1]['mae_mean'])
                print(f"🏆 Best model for window {window_size}: {best_model[0]} (MAE: {best_model[1]['mae_mean']:.3f})")
                return results, eval_config
            else:
                print(f"❌ No successful results for window size {window_size}")
        else:
            print(f"❌ No results for window size {window_size}")
            
    except Exception as e:
        print(f"❌ Window size {window_size} failed: {str(e)}")
        return None, None
    
    return None, None


def run_full_window_analysis(custom_config: ExperimentConfig, dataset_key: str = 'train_split'):
    """Run the full window size analysis with custom configuration."""
    
    dataset_path = custom_config.available_datasets[dataset_key]
    
    print(f"\n{'#'*80}")
    print(f"# FULL WINDOW SIZE ANALYSIS")
    print(f"# Dataset: {dataset_path}")
    print(f"# Goal: minutes")
    print(f"# Features: all")
    print(f"# Window sizes: {custom_config.window_sizes['custom_range']}")
    print(f"{'#'*80}")
    
    try:
        results = run_window_size_analysis(
            dataset_name=dataset_key,
            goal_type='minutes',
            feature_set='all',
            window_sizes=custom_config.window_sizes['custom_range'],
            model_set='all_except_neural',
            cv_config='standard',
            config_obj=custom_config
        )
        
        print(f"\n✅ Full window analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Full window analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    
    print("🚀 Custom Experiment Runner")
    print("=" * 80)
    print("Configuration:")
    print("  Dataset: data_splits_student_week_aggregations_rolling_new/[train|val].csv")
    print("  Features: all features available")
    print("  Goal: minutes") 
    print("  Window range: [1, 26], step size: 5")
    print("  Window sizes: [1, 6, 11, 16, 21, 26]")
    print("  Models: All except neural networks (14 models)")
    print("  Note: Using val.csv as test dataset")
    
    # Determine dataset type from command line or user input
    dataset_type = 'train'  # default
    
    # Check if user specified dataset type in command line
    if len(sys.argv) > 1:
        # Check for dataset specification: train, test, --dataset=train, --dataset=test
        if 'train' in sys.argv:
            dataset_type = 'train'
            sys.argv.remove('train')
        elif 'test' in sys.argv:
            dataset_type = 'test'
            sys.argv.remove('test')
        elif any(arg.startswith('--dataset=') for arg in sys.argv):
            for arg in sys.argv:
                if arg.startswith('--dataset='):
                    dataset_type = arg.split('=')[1]
                    sys.argv.remove(arg)
                    break
    
    # Setup custom configuration
    custom_config, dataset_key = setup_custom_config(dataset_type)
    
    # Check if user wants to run individual experiments or full analysis
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'single' and len(sys.argv) > 2:
            # Run single window size experiment
            window_size = int(sys.argv[2])
            if window_size in custom_config.window_sizes['custom_range']:
                run_single_window_experiment(window_size, custom_config, dataset_key)
            else:
                print(f"❌ Window size {window_size} not in range. Available: {custom_config.window_sizes['custom_range']}")
        
        elif mode == 'full':
            # Run full window analysis
            run_full_window_analysis(custom_config, dataset_key)
        
        else:
            print("Usage:")
            print("  python run_custom_experiment.py [train|test] single <window_size>  # Run single window experiment")
            print("  python run_custom_experiment.py [train|test] full                  # Run full window analysis")
            print("  python run_custom_experiment.py --dataset=train full               # Alternative syntax")
            print("  python run_custom_experiment.py --dataset=test full                # Alternative syntax")
            print(f"  Available window sizes: {custom_config.window_sizes['custom_range']}")
            print(f"  Available datasets: train, test")
            print(f"  Default models: all_except_neural (14 models, excludes MLP & LSTM)")
    
    else:
        print(f"\nChoose dataset:")
        print(f"1. Train dataset (train.csv)")
        print(f"2. Test dataset (val.csv)")
        
        dataset_choice = input("\nEnter dataset choice (1/2): ").strip()
        if dataset_choice == '2':
            dataset_type = 'test'
            custom_config, dataset_key = setup_custom_config('test')
        
        print(f"\nChoose experiment mode:")
        print(f"1. Run full window analysis (all window sizes)")
        print(f"2. Run single window experiment")
        print(f"3. Show configuration only")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            run_full_window_analysis(custom_config, dataset_key)
        elif choice == '2':
            print(f"Available window sizes: {custom_config.window_sizes['custom_range']}")
            window_size = int(input("Enter window size: "))
            if window_size in custom_config.window_sizes['custom_range']:
                run_single_window_experiment(window_size, custom_config, dataset_key)
            else:
                print(f"❌ Invalid window size. Must be one of: {custom_config.window_sizes['custom_range']}")
        elif choice == '3':
            print("\n✅ Configuration displayed above.")
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main() 