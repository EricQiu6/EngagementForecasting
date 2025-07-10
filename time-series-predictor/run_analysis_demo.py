#!/usr/bin/env python3
"""
Demonstration of Comprehensive Evaluation Analysis
=================================================

This script shows how to run the complete pipeline:
1. Run comprehensive evaluation with saved predictions
2. Analyze the saved results
3. Generate plots and statistical analysis

Usage:
    python run_analysis_demo.py [--target-type minutes_per_week] [--window-size 8]
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import time

def run_evaluation_with_predictions(target_type='minutes_per_week', window_size=8):
    """Run comprehensive evaluation with prediction saving."""
    print(f"🔄 Running comprehensive evaluation...")
    print(f"   Target: {target_type}")
    print(f"   Window size: {window_size}")
    
    # Import and run the evaluation
    try:
        from comprehensive_evaluation_with_saved_predictions import run_evaluation_with_predictions
        
        results = run_evaluation_with_predictions(
            schema_name='time_goal_extended',
            window_size=window_size,
            target_type=target_type,
            save_predictions=True
        )
        
        print(f"✅ Evaluation completed successfully!")
        return f"evaluation_outputs/{target_type}_window{window_size}"
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def run_analysis(results_dir):
    """Run comprehensive analysis on saved predictions."""
    print(f"\n🔍 Running comprehensive analysis...")
    print(f"   Results directory: {results_dir}")
    
    try:
        from comprehensive_evaluation_analysis import PredictionAnalyzer
        
        # Initialize analyzer
        analyzer = PredictionAnalyzer(results_dir)
        
        # Run analysis
        analysis_results = analyzer.run_comprehensive_analysis()
        
        print(f"✅ Analysis completed successfully!")
        return analysis_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description='Demonstration of comprehensive evaluation and analysis pipeline'
    )
    parser.add_argument('--target-type', type=str, default='minutes_per_week',
                        choices=['minutes_per_week', 'avg_proficiency'],
                        help='Target variable to predict')
    parser.add_argument('--window-size', type=int, default=8,
                        help='Window size for time series sequences')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation step (use existing results)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory with existing results (if skipping evaluation)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION AND ANALYSIS DEMO")
    print("="*80)
    
    results_dir = None
    
    # Step 1: Run evaluation (if not skipping)
    if not args.skip_evaluation:
        print("\n🚀 STEP 1: Running Comprehensive Evaluation with Saved Predictions")
        print("-" * 60)
        
        start_time = time.time()
        results_dir = run_evaluation_with_predictions(
            target_type=args.target_type,
            window_size=args.window_size
        )
        evaluation_time = time.time() - start_time
        
        if results_dir is None:
            print("❌ Evaluation failed. Cannot proceed with analysis.")
            sys.exit(1)
        
        print(f"⏱️  Evaluation completed in {evaluation_time:.1f} seconds")
        
    else:
        if args.results_dir:
            results_dir = args.results_dir
        else:
            # Try to find the most recent results
            results_base = Path(f"evaluation_outputs/{args.target_type}_window{args.window_size}")
            if results_base.exists():
                results_dir = str(results_base)
            else:
                print(f"❌ No existing results found for {args.target_type} with window {args.window_size}")
                print(f"   Looked in: {results_base}")
                sys.exit(1)
    
    # Step 2: Run analysis
    print(f"\n🔍 STEP 2: Running Comprehensive Analysis")
    print("-" * 60)
    
    start_time = time.time()
    analysis_results = run_analysis(results_dir)
    analysis_time = time.time() - start_time
    
    if analysis_results is None:
        print("❌ Analysis failed.")
        sys.exit(1)
    
    print(f"⏱️  Analysis completed in {analysis_time:.1f} seconds")
    
    # Step 3: Display summary
    print(f"\n📊 STEP 3: Summary of Results")
    print("-" * 60)
    
    performance_summary = analysis_results['performance_summary']
    significance_results = analysis_results['significance_results']
    
    print(f"\n🏆 MODEL PERFORMANCE RANKING:")
    print(f"{'Rank':<6} {'Model':<25} {'Category':<15} {'MAE':<10} {'R²':<8}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(performance_summary.head(10).iterrows(), 1):
        print(f"{i:<6} {row['model']:<25} {row['category']:<15} "
              f"{row['mae_mean']:<10.3f} {row['r2_mean']:<8.3f}")
    
    # Statistical significance summary
    significant_pairs = significance_results[significance_results['is_significant']]
    print(f"\n🧪 STATISTICAL SIGNIFICANCE:")
    print(f"   Total comparisons: {len(significance_results)}")
    print(f"   Significant differences: {len(significant_pairs)}")
    
    if len(significant_pairs) > 0:
        print(f"   Most significant:")
        best_pair = significant_pairs.loc[significant_pairs['p_value'].idxmin()]
        print(f"   - {best_pair['model1']} vs {best_pair['model2']}")
        print(f"     p-value: {best_pair['p_value']:.6f}")
        print(f"     Effect size: {best_pair['effect_size']}")
    
    # File locations
    output_dir = analysis_results['output_dir']
    print(f"\n📁 GENERATED FILES:")
    print(f"   Analysis results: {output_dir}")
    print(f"   Summary report: {output_dir}/analysis_summary_report_*.md")
    print(f"   Performance plots: {output_dir}/*.png")
    print(f"   Data files: {output_dir}/*.csv")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🎯 Best model: {performance_summary.iloc[0]['model']} "
          f"(MAE: {performance_summary.iloc[0]['mae_mean']:.3f})")

if __name__ == "__main__":
    main()