#!/usr/bin/env python3
"""
Demo: Using the Experiment Configuration System
==============================================

This script demonstrates how to use the new experiment configuration system
in comprehensive_evaluation_with_saved_predictions.py.

Examples include:
1. Running single experiments with different configurations
2. Running window size analysis
3. Running feature ablation studies
4. Running comprehensive experiment suites
"""

from comprehensive_evaluation_with_saved_predictions import (
    DEFAULT_EXPERIMENT_CONFIG,
    run_evaluation_with_predictions,
    run_window_size_analysis,
    run_feature_ablation_study,
    run_comprehensive_experiment_suite
)


def demo_single_experiments():
    """Demo running single experiments with different configurations."""
    
    print("=" * 80)
    print("DEMO: Single Experiments")
    print("=" * 80)
    
    # Example 1: Basic experiment with minutes prediction
    print("\n1. Basic minutes prediction experiment:")
    experiment_config_1 = {
        'dataset_name': 'rolling_new',
        'goal_type': 'minutes',
        'window_size': 8,
        'feature_set': 'all',
        'cv_config': 'standard',
        'model_set': 'top_performers'
    }
    
    print("Configuration:", experiment_config_1)
    # results, config = run_evaluation_with_predictions(experiment_config_1)
    
    # Example 2: Proficiency prediction with limited features
    print("\n2. Proficiency prediction with limited features:")
    experiment_config_2 = {
        'dataset_name': 'rolling_new',
        'goal_type': 'proficiency',
        'window_size': 12,
        'feature_set': 'top_5_performance',
        'cv_config': 'robust',
        'model_set': 'linear_models'
    }
    
    print("Configuration:", experiment_config_2)
    # results, config = run_evaluation_with_predictions(experiment_config_2)
    
    # Example 3: Quick experiment with baselines only
    print("\n3. Quick baseline experiment:")
    experiment_config_3 = {
        'dataset_name': 'rolling_new',
        'goal_type': 'minutes',
        'window_size': 5,
        'feature_set': 'minimal_goal',
        'cv_config': 'quick',
        'model_set': 'baselines_only'
    }
    
    print("Configuration:", experiment_config_3)
    # results, config = run_evaluation_with_predictions(experiment_config_3)


def demo_window_analysis():
    """Demo running window size analysis."""
    
    print("=" * 80)
    print("DEMO: Window Size Analysis")
    print("=" * 80)
    
    # Example 1: Standard window analysis
    print("\n1. Standard window analysis for minutes prediction:")
    print("Parameters:")
    print("  - Dataset: rolling_new")
    print("  - Goal: minutes")  
    print("  - Feature set: all")
    print("  - Window sizes: [5, 8, 12, 15, 20]")
    print("  - Models: top_performers")
    
    # results = run_window_size_analysis(
    #     dataset_name='rolling_new',
    #     goal_type='minutes',
    #     feature_set='all',
    #     window_sizes=[5, 8, 12, 15, 20],
    #     model_set='top_performers'
    # )
    
    # Example 2: Focused window analysis with limited features
    print("\n2. Focused window analysis with limited features:")
    print("Parameters:")
    print("  - Dataset: rolling_new")
    print("  - Goal: proficiency")
    print("  - Feature set: top_5_performance")
    print("  - Window sizes: [3, 8, 15]")
    print("  - Models: linear_models")
    
    # results = run_window_size_analysis(
    #     dataset_name='rolling_new',
    #     goal_type='proficiency',
    #     feature_set='top_5_performance',
    #     window_sizes=[3, 8, 15],
    #     model_set='linear_models'
    # )


def demo_feature_ablation():
    """Demo running feature ablation studies."""
    
    print("=" * 80)
    print("DEMO: Feature Ablation Study")
    print("=" * 80)
    
    # Example 1: Compare different feature sets
    print("\n1. Comparing different feature sets:")
    print("Parameters:")
    print("  - Dataset: rolling_new")
    print("  - Goal: minutes")
    print("  - Window size: 8")
    print("  - Feature sets: ['all', 'top_5_temporal', 'top_5_performance', 'top_5_engagement']")
    print("  - Models: top_performers")
    
    # results = run_feature_ablation_study(
    #     dataset_name='rolling_new',
    #     goal_type='minutes',
    #     window_size=8,
    #     feature_sets=['all', 'top_5_temporal', 'top_5_performance', 'top_5_engagement'],
    #     model_set='top_performers'
    # )
    
    # Example 2: Minimal feature comparison
    print("\n2. Minimal vs full feature comparison:")
    print("Parameters:")
    print("  - Dataset: rolling_new")
    print("  - Goal: proficiency")
    print("  - Window size: 12")
    print("  - Feature sets: ['minimal_goal', 'all']")
    print("  - Models: all")
    
    # results = run_feature_ablation_study(
    #     dataset_name='rolling_new',
    #     goal_type='proficiency',
    #     window_size=12,
    #     feature_sets=['minimal_goal', 'all'],
    #     model_set='all'
    # )


def demo_comprehensive_suite():
    """Demo running comprehensive experiment suite."""
    
    print("=" * 80)
    print("DEMO: Comprehensive Experiment Suite")
    print("=" * 80)
    
    print("\nThe comprehensive suite includes:")
    print("1. Main evaluation for minutes prediction")
    print("2. Main evaluation for proficiency prediction")
    print("3. Window size analysis for minutes")
    print("4. Feature ablation study for minutes")
    print("\nThis will run multiple experiments automatically.")
    
    # results = run_comprehensive_experiment_suite()


def demo_configuration_options():
    """Demo available configuration options."""
    
    print("=" * 80)
    print("DEMO: Configuration Options")
    print("=" * 80)
    
    config = DEFAULT_EXPERIMENT_CONFIG
    
    print("\nAvailable Datasets:")
    for key, value in config.available_datasets.items():
        print(f"  {key}: {value}")
    
    print("\nAvailable Goal Types:")
    for key, value in config.goal_types.items():
        print(f"  {key}: {value['description']}")
        print(f"    Target: {value['target_column']}")
        print(f"    Schema: {value['schema_name']}")
    
    print("\nAvailable Window Size Sets:")
    for key, value in config.window_sizes.items():
        print(f"  {key}: {value}")
    
    print("\nAvailable Feature Sets:")
    for key, value in config.feature_sets.items():
        if value is None:
            print(f"  {key}: All features from schema")
        else:
            print(f"  {key}: {value}")
    
    print("\nAvailable CV Configurations:")
    for key, value in config.cv_configs.items():
        print(f"  {key}: {value}")
    
    print("\nAvailable Model Sets:")
    for key, value in config.model_sets.items():
        if value is None:
            print(f"  {key}: All available models")
        else:
            print(f"  {key}: {value}")


def main():
    """Main demo function."""
    
    print("🚀 Experiment Configuration System Demo")
    print("=" * 80)
    print("This demo shows how to use the new configuration system.")
    print("All actual experiments are commented out - uncomment to run.")
    
    # Run configuration demos
    demo_configuration_options()
    demo_single_experiments()
    demo_window_analysis()
    demo_feature_ablation()
    demo_comprehensive_suite()
    
    print("\n" + "=" * 80)
    print("COMMAND LINE EXAMPLES")
    print("=" * 80)
    
    print("\nSingle experiment:")
    print("python comprehensive_evaluation_with_saved_predictions.py --mode single --dataset rolling_new --goal minutes --window-size 8 --feature-set all --model-set top_performers")
    
    print("\nWindow size analysis:")
    print("python comprehensive_evaluation_with_saved_predictions.py --mode window_analysis --dataset rolling_new --goal minutes --feature-set all --model-set top_performers")
    
    print("\nFeature ablation study:")
    print("python comprehensive_evaluation_with_saved_predictions.py --mode feature_ablation --dataset rolling_new --goal minutes --window-size 8 --model-set top_performers")
    
    print("\nComprehensive suite:")
    print("python comprehensive_evaluation_with_saved_predictions.py --mode comprehensive_suite")
    
    print("\n✅ Demo completed! Check the configurations above and run experiments as needed.")


if __name__ == "__main__":
    main() 