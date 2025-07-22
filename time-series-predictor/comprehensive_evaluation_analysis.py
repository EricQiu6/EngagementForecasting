"""
Comprehensive Analysis of Saved Predictions
==========================================

This script loads saved predictions from comprehensive_evaluation_with_saved_predictions.py
and performs detailed analysis including:
1. Statistical significance testing (bootstrapping)
2. Predicted vs actual plots
3. Feature importance analysis
4. Window size and architecture aggregation
5. Error analysis and model comparison

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import bootstrap
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PredictionAnalyzer:
    """Analyzes saved predictions from comprehensive evaluation."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.evaluation_config = {}
        self.prediction_data = {}
        
        # Load all data
        self._load_evaluation_data()
        
    def _load_evaluation_data(self):
        """Load all saved evaluation data."""
        print(f"Loading evaluation data from: {self.results_dir}")
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        # Load evaluation config
        config_file = self.results_dir / 'evaluation_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.evaluation_config = json.load(f)
            print(f"✅ Loaded evaluation config: {len(self.evaluation_config)} parameters")
        
        # Load overall results
        overall_results_file = self.results_dir / 'overall_results.json'
        if overall_results_file.exists():
            with open(overall_results_file, 'r') as f:
                overall_data = json.load(f)
                self.results_data = overall_data.get('model_results', {})
            print(f"✅ Loaded overall results: {len(self.results_data)} models")
        
        # Load individual model predictions
        model_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            model_predictions = {}
            
            # Load fold predictions
            fold_files = list(model_dir.glob('fold_*_predictions.json'))
            for fold_file in fold_files:
                fold_idx = int(fold_file.stem.split('_')[1])
                with open(fold_file, 'r') as f:
                    fold_data = json.load(f)
                    model_predictions[fold_idx] = fold_data
            
            # Load model summary
            summary_file = model_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    model_predictions['summary'] = summary_data
            
            if model_predictions:
                self.prediction_data[model_name] = model_predictions
        
        print(f"✅ Loaded predictions for {len(self.prediction_data)} models")
        
        # Create consolidated prediction DataFrame
        self._create_prediction_dataframe()
    
    def _create_prediction_dataframe(self):
        """Create a consolidated DataFrame with all predictions."""
        prediction_records = []
        
        for model_name, model_data in self.prediction_data.items():
            for fold_idx, fold_data in model_data.items():
                if fold_idx == 'summary':
                    continue
                
                y_true = np.array(fold_data['y_true'])
                y_pred = np.array(fold_data['y_pred'])
                indices = fold_data['indices']
                
                for i, (true_val, pred_val, idx) in enumerate(zip(y_true, y_pred, indices)):
                    prediction_records.append({
                        'model': model_name,
                        'fold': fold_idx,
                        'sample_idx': idx,
                        'y_true': true_val,
                        'y_pred': pred_val,
                        'error': abs(true_val - pred_val),
                        'squared_error': (true_val - pred_val) ** 2,
                        'percentage_error': abs(true_val - pred_val) / max(abs(true_val), 1e-8) * 100
                    })
        
        self.prediction_df = pd.DataFrame(prediction_records)
        print(f"✅ Created prediction DataFrame: {len(self.prediction_df)} predictions")
    
    def run_comprehensive_analysis(self, output_dir: Optional[Union[str, Path]] = None):
        """Run all analysis components."""
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        # 1. Model performance summary
        print("\n1. Creating model performance summary...")
        performance_summary = self.create_performance_summary()
        performance_summary.to_csv(output_dir / f'model_performance_summary_{timestamp}.csv')
        
        # 2. Statistical significance testing
        print("\n2. Running statistical significance tests...")
        significance_results = self.perform_significance_testing()
        significance_results.to_csv(output_dir / f'significance_testing_{timestamp}.csv')
        
        # 3. Predicted vs actual plots
        print("\n3. Creating predicted vs actual plots...")
        self.create_predicted_vs_actual_plots(output_dir, timestamp)
        
        # 4. Error analysis
        print("\n4. Performing error analysis...")
        self.create_error_analysis_plots(output_dir, timestamp)
        
        # 5. Model comparison plots
        print("\n5. Creating model comparison plots...")
        self.create_model_comparison_plots(output_dir, timestamp)
        
        # 6. Feature importance analysis (if available)
        print("\n6. Analyzing feature importance...")
        self.analyze_feature_importance(output_dir, timestamp)
        
        # 7. Hyperparameter sensitivity analysis (if available)
        print("\n7. Analyzing hyperparameter sensitivity...")
        sensitivity_df = self.analyze_hyperparameter_sensitivity_from_results(output_dir, timestamp)
        
        # 8. Key bootstrap findings (category comparisons)
        print("\n8. Computing key bootstrap findings...")
        bootstrap_results = self.compute_bootstrap_confidence_intervals()
        bootstrap_results.to_csv(output_dir / f'key_bootstrap_findings_{timestamp}.csv')
        
        # 9. Window and Architecture Analysis
        print("\n9. Analyzing window and architecture combinations...")
        window_arch_results = self.analyze_window_architecture_combinations(output_dir, timestamp)
        
        # 10. Create summary report
        print("\n10. Creating summary report...")
        self.create_summary_report(output_dir, timestamp, performance_summary, significance_results)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        
        return {
            'performance_summary': performance_summary,
            'significance_results': significance_results,
            'bootstrap_results': bootstrap_results,
            'window_architecture_results': window_arch_results,
            'output_dir': output_dir
        }
    
    def create_performance_summary(self) -> pd.DataFrame:
        """Create comprehensive performance summary."""
        # Calculate metrics from prediction data
        summary_data = []
        
        for model_name in self.prediction_df['model'].unique():
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            # Calculate overall metrics
            y_true = model_data['y_true'].values
            y_pred = model_data['y_pred'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
            
            # Calculate fold-wise metrics for std
            fold_maes = []
            fold_rmses = []
            fold_r2s = []
            
            for fold in model_data['fold'].unique():
                fold_data = model_data[model_data['fold'] == fold]
                fold_y_true = fold_data['y_true'].values
                fold_y_pred = fold_data['y_pred'].values
                
                if len(fold_y_true) > 0:
                    fold_mae = mean_absolute_error(fold_y_true, fold_y_pred)
                    fold_rmse = np.sqrt(mean_squared_error(fold_y_true, fold_y_pred))
                    fold_r2 = r2_score(fold_y_true, fold_y_pred) if len(np.unique(fold_y_true)) > 1 else 0
                    
                    fold_maes.append(fold_mae)
                    fold_rmses.append(fold_rmse)
                    fold_r2s.append(fold_r2)
            
            # Additional metrics
            median_ae = np.median(model_data['error'])
            percentile_90_ae = np.percentile(model_data['error'], 90)
            mean_percentage_error = np.mean(model_data['percentage_error'])
            
            # Get category from results_data if available
            category = 'unknown'
            if model_name in self.results_data:
                category = self.results_data[model_name].get('category', 'unknown')
            
            summary_data.append({
                'model': model_name,
                'category': category,
                'mae_mean': mae,
                'mae_std': np.std(fold_maes) if fold_maes else 0,
                'rmse_mean': rmse,
                'rmse_std': np.std(fold_rmses) if fold_rmses else 0,
                'r2_mean': r2,
                'r2_std': np.std(fold_r2s) if fold_r2s else 0,
                'median_ae': median_ae,
                'percentile_90_ae': percentile_90_ae,
                'mean_percentage_error': mean_percentage_error,
                'n_predictions': len(model_data),
                'n_folds': len(model_data['fold'].unique())
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('mae_mean')
        
        # Add rank columns
        summary_df['mae_rank'] = summary_df['mae_mean'].rank()
        summary_df['rmse_rank'] = summary_df['rmse_mean'].rank()
        summary_df['r2_rank'] = summary_df['r2_mean'].rank(ascending=False)
        
        # Round numerical columns to 2 decimal places for cleaner CSV output
        numerical_columns = ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std', 
                           'median_ae', 'percentile_90_ae', 'mean_percentage_error']
        for col in numerical_columns:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(2)
        
        return summary_df
    
    def perform_significance_testing(self, alpha: float = 0.05) -> pd.DataFrame:
        """Perform pairwise significance tests between models."""
        models = list(self.prediction_df['model'].unique())
        significance_results = []
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:  # Only test unique pairs
                    continue
                
                model1_data = self.prediction_df[self.prediction_df['model'] == model1]
                model2_data = self.prediction_df[self.prediction_df['model'] == model2]
                
                # Get errors for comparison
                model1_errors = model1_data['error'].values
                model2_errors = model2_data['error'].values
                
                # Paired t-test (if same samples)
                if len(model1_errors) == len(model2_errors):
                    try:
                        # Check if samples are from same indices
                        model1_indices = set(model1_data['sample_idx'].values)
                        model2_indices = set(model2_data['sample_idx'].values)
                        
                        if model1_indices == model2_indices:
                            # Paired test
                            stat, p_value = stats.ttest_rel(model1_errors, model2_errors)
                            test_type = 'paired_ttest'
                        else:
                            # Independent test
                            stat, p_value = stats.ttest_ind(model1_errors, model2_errors)
                            test_type = 'independent_ttest'
                    except:
                        # Fallback to Mann-Whitney U test
                        stat, p_value = stats.mannwhitneyu(model1_errors, model2_errors, alternative='two-sided')
                        test_type = 'mannwhitney'
                else:
                    # Independent samples t-test
                    stat, p_value = stats.ttest_ind(model1_errors, model2_errors)
                    test_type = 'independent_ttest'
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(model1_errors) - 1) * np.std(model1_errors, ddof=1)**2 + 
                                     (len(model2_errors) - 1) * np.std(model2_errors, ddof=1)**2) / 
                                    (len(model1_errors) + len(model2_errors) - 2))
                
                cohens_d = (np.mean(model1_errors) - np.mean(model2_errors)) / pooled_std if pooled_std > 0 else 0
                
                significance_results.append({
                    'model1': model1,
                    'model2': model2,
                    'model1_mae': np.mean(model1_errors),
                    'model2_mae': np.mean(model2_errors),
                    'mae_difference': np.mean(model1_errors) - np.mean(model2_errors),
                    'test_statistic': stat,
                    'p_value': p_value,
                    'is_significant': p_value < alpha,
                    'test_type': test_type,
                    'cohens_d': cohens_d,
                    'effect_size': self._interpret_effect_size(abs(cohens_d))
                })
        
        significance_df = pd.DataFrame(significance_results)
        
        # Round numerical columns to 2 decimal places for cleaner CSV output
        numerical_columns = ['model1_mae', 'model2_mae', 'mae_difference', 'test_statistic', 'p_value', 'cohens_d']
        for col in numerical_columns:
            if col in significance_df.columns:
                significance_df[col] = significance_df[col].round(2)
        
        return significance_df
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def create_predicted_vs_actual_plots(self, output_dir: Path, timestamp: str):
        """Create predicted vs actual plots for all models."""
        # Get top models by performance
        performance_summary = self.create_performance_summary()
        top_models = performance_summary.head(6)['model'].tolist()
        
        # Create subplot grid
        n_models = len(top_models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, model_name in enumerate(top_models):
            ax = axes[i]
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            y_true = model_data['y_true'].values
            y_pred = model_data['y_pred'].values
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Calculate R²
            r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
            mae = mean_absolute_error(y_true, y_pred)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model_name}\nMAE: {mae:.2f}, R²: {r2:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'predicted_vs_actual_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_analysis_plots(self, output_dir: Path, timestamp: str):
        """Create error analysis plots."""
        # 1. Error distribution plots
        top_models = self.create_performance_summary().head(6)['model'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, model_name in enumerate(top_models):
            ax = axes[i]
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            errors = model_data['error'].values
            
            # Histogram of errors
            ax.hist(errors, bins=30, alpha=0.7, density=True, edgecolor='black')
            ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
            ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}')
            
            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Density')
            ax.set_title(f'{model_name}\nError Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'error_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residual plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, model_name in enumerate(top_models):
            ax = axes[i]
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            y_true = model_data['y_true'].values
            y_pred = model_data['y_pred'].values
            residuals = y_true - y_pred
            
            # Residual plot
            ax.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax.axhline(y=0, color='red', linestyle='--')
            
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{model_name}\nResidual Plot')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'residual_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison_plots(self, output_dir: Path, timestamp: str):
        """Create comprehensive model comparison plots."""
        performance_summary = self.create_performance_summary()
        
        # 1. Model performance ranking
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MAE comparison
        ax = axes[0]
        models = performance_summary['model'].values
        maes = performance_summary['mae_mean'].values
        mae_stds = performance_summary['mae_std'].values
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), maes, yerr=mae_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Model Performance Comparison (MAE)')
        ax.grid(True, alpha=0.3)
        
        # RMSE comparison
        ax = axes[1]
        rmses = performance_summary['rmse_mean'].values
        rmse_stds = performance_summary['rmse_std'].values
        
        bars = ax.bar(range(len(models)), rmses, yerr=rmse_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Root Mean Squared Error')
        ax.set_title('Model Performance Comparison (RMSE)')
        ax.grid(True, alpha=0.3)
        
        # R² comparison
        ax = axes[2]
        r2s = performance_summary['r2_mean'].values
        r2_stds = performance_summary['r2_std'].values
        
        bars = ax.bar(range(len(models)), r2s, yerr=r2_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance Comparison (R²)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance by category
        if 'category' in performance_summary.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            category_stats = performance_summary.groupby('category').agg({
                'mae_mean': ['mean', 'std', 'min', 'max'],
                'model': 'count'
            }).round(3)
            
            categories = category_stats.index.values
            means = category_stats[('mae_mean', 'mean')].values
            stds = category_stats[('mae_mean', 'std')].values
            counts = category_stats[('model', 'count')].values
            
            bars = ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.8)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (stds[list(bars).index(bar)] if stds[list(bars).index(bar)] > 0 else 0),
                        f'n={count}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Mean Absolute Error')
            ax.set_title('Performance by Model Category')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'performance_by_category_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_hyperparameter_sensitivity_plots(self, sensitivity_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create visualizations for hyperparameter sensitivity analysis."""
        
        if sensitivity_df.empty:
            print("No hyperparameter sensitivity data available for plotting.")
            return
        
        print("Creating hyperparameter sensitivity plots...")
        
        # Create comprehensive sensitivity visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Coefficient of variation comparison (lower = more robust)
        ax = axes[0, 0]
        models = sensitivity_df['base_model'].values
        cvs = sensitivity_df['coefficient_of_variation'].values
        
        colors = ['green' if cv < 0.1 else 'orange' if cv < 0.2 else 'red' for cv in cvs]
        bars = ax.bar(models, cvs, color=colors, alpha=0.7)
        
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Hyperparameter Sensitivity (Lower = More Robust)')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Robust (CV < 0.1)')
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate (CV < 0.2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cv in zip(bars, cvs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Performance range across hyperparameters
        ax = axes[0, 1]
        ranges = sensitivity_df['mae_range'].values
        
        bars = ax.bar(models, ranges, color='skyblue', alpha=0.7)
        ax.set_ylabel('MAE Range')
        ax.set_title('Performance Range Across Hyperparameters')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, range_val in zip(bars, ranges):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{range_val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Best vs worst performance
        ax = axes[1, 0]
        x_pos = np.arange(len(models))
        width = 0.35
        
        best_maes = sensitivity_df['best_mae'].values
        worst_maes = sensitivity_df['worst_mae'].values
        
        ax.bar(x_pos - width/2, best_maes, width, label='Best Config', alpha=0.7, color='green')
        ax.bar(x_pos + width/2, worst_maes, width, label='Worst Config', alpha=0.7, color='red')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        ax.set_title('Best vs Worst Hyperparameter Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Number of configurations tested
        ax = axes[1, 1]
        n_configs = sensitivity_df['n_configurations'].values
        
        bars = ax.bar(models, n_configs, color='purple', alpha=0.7)
        ax.set_ylabel('Number of Configurations')
        ax.set_title('Hyperparameter Configurations Tested')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, n_config in zip(bars, n_configs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{n_config}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels for better readability
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'hyperparameter_sensitivity_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create sensitivity ranking visualization
        self._create_sensitivity_ranking_plot(sensitivity_df, output_dir, timestamp)
        
        print(f"✅ Hyperparameter sensitivity plots saved")
    
    def _create_sensitivity_ranking_plot(self, sensitivity_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create a detailed ranking plot for hyperparameter sensitivity."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by coefficient of variation for ranking
        sorted_df = sensitivity_df.sort_values('coefficient_of_variation')
        
        models = sorted_df['base_model'].values
        cvs = sorted_df['coefficient_of_variation'].values
        ranges = sorted_df['mae_range'].values
        n_configs = sorted_df['n_configurations'].values
        
        # Create horizontal bar chart with coefficient of variation
        y_pos = np.arange(len(models))
        
        # Color bars based on robustness
        colors = ['darkgreen' if cv < 0.05 else 'green' if cv < 0.1 else 
                 'orange' if cv < 0.2 else 'red' for cv in cvs]
        
        bars = ax.barh(y_pos, cvs, color=colors, alpha=0.7)
        
        # Add range information as text
        for i, (cv, range_val, n_config) in enumerate(zip(cvs, ranges, n_configs)):
            ax.text(cv + 0.01, i, f'Range: {range_val:.2f}, N: {n_config}', 
                   va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Coefficient of Variation (Lower = More Robust)')
        ax.set_title('Model Robustness Ranking\n(Based on Hyperparameter Sensitivity)')
        
        # Add robustness zones
        ax.axvline(x=0.05, color='darkgreen', linestyle='--', alpha=0.5, label='Very Robust')
        ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Robust')
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'sensitivity_ranking_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_hyperparameter_sensitivity_from_results(self, output_dir: Path, timestamp: str) -> pd.DataFrame:
        """Analyze hyperparameter sensitivity from saved results data."""
        
        # Check if hyperparameter sensitivity data exists
        sensitivity_file = self.results_dir / 'hyperparameter_sensitivity.csv'
        
        if sensitivity_file.exists():
            print("Loading hyperparameter sensitivity data from file...")
            sensitivity_df = pd.read_csv(sensitivity_file)
            
            # Create visualizations
            self.create_hyperparameter_sensitivity_plots(sensitivity_df, output_dir, timestamp)
            
            return sensitivity_df
        
        else:
            # Try to extract hyperparameter sensitivity from results data
            print("No hyperparameter sensitivity file found. Analyzing from results data...")
            
            # Import the analysis function from the evaluation script
            from comprehensive_evaluation_with_saved_predictions import analyze_hyperparameter_sensitivity
            
            sensitivity_df = analyze_hyperparameter_sensitivity(self.results_data)
            
            if not sensitivity_df.empty:
                # Save the analysis
                sensitivity_df.to_csv(output_dir / f'hyperparameter_sensitivity_{timestamp}.csv', index=False)
                
                # Create visualizations
                self.create_hyperparameter_sensitivity_plots(sensitivity_df, output_dir, timestamp)
                
                print(f"✅ Hyperparameter sensitivity analysis completed and saved")
            else:
                print("No hyperparameter sensitivity data available (single configs only)")
            
            return sensitivity_df
     
    def analyze_feature_importance(self, output_dir: Path, timestamp: str):
        """Analyze feature importance if available from the models."""
        print("Extracting and analyzing feature importance...")
        
        # Look for feature importance data in saved results
        feature_importance_data = self._extract_feature_importance_data()
        
        if not feature_importance_data:
            print("No feature importance data found - trying to extract from model summaries...")
            # Try to extract from model summaries if available
            feature_importance_data = self._extract_from_model_summaries()
        
        if not feature_importance_data:
            print("No feature importance data available - skipping feature importance analysis")
            return
        
        # Convert to DataFrame for analysis
        df_importance = pd.DataFrame(feature_importance_data)
        
        # Create feature importance analysis
        self._create_feature_importance_analysis(df_importance, output_dir, timestamp)
        
        # Focus on specific features of interest (including new class-level and prior achievement features)
        specific_features = [
            'student_learning_rate', 'student_ability', 'avg_difficulty',  # Original key features
            'performance_vs_class_mean_prof', 'performance_vs_class_mean_mins',  # Class comparison features
            'class_percentile_rank_prof', 'class_percentile_rank_mins',  # Peer ranking features
            'starting_ability_quartile', 'performance_consistency_score', 'learning_acceleration_capacity'  # Prior achievement features
        ]
        self._analyze_specific_features(df_importance, specific_features, output_dir, timestamp)
        
        print(f"✅ Feature importance analysis completed")
    
    def _extract_feature_importance_data(self):
        """Extract feature importance data from saved results."""
        feature_importance_data = []
        
        # Look for feature importance in model summaries
        for model_name, model_data in self.prediction_data.items():
            if 'summary' in model_data:
                summary = model_data['summary']
                
                # Check if feature importance is saved in summary
                if 'feature_importance' in summary:
                    importance_dict = summary['feature_importance']
                    
                    for feature_name, importance in importance_dict.items():
                        feature_importance_data.append({
                            'model': model_name,
                            'feature': feature_name,
                            'importance': importance,
                            'source': 'summary'
                        })
        
        return feature_importance_data
    
    def _extract_from_model_summaries(self):
        """Try to extract feature importance from model summaries."""
        feature_importance_data = []
        
        # Check if any models have feature importance in their results
        for model_name, model_results in self.results_data.items():
            if isinstance(model_results, dict):
                # Look for feature importance keys
                for key, value in model_results.items():
                    if 'feature' in key.lower() and 'importance' in key.lower():
                        if isinstance(value, dict):
                            for feature_name, importance in value.items():
                                feature_importance_data.append({
                                    'model': model_name,
                                    'feature': feature_name,
                                    'importance': importance,
                                    'source': 'results'
                                })
        
        return feature_importance_data
    
    def _create_feature_importance_analysis(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create comprehensive feature importance analysis using advanced methods from feature_importance_analysis.py."""
        
        print("Creating comprehensive feature importance analysis...")
        
        # 1. Top features by model
        top_features_by_model = self._get_top_features_by_model(df_importance)
        self._save_top_features_summary(top_features_by_model, output_dir, timestamp)
        
        # 2. Create advanced visualizations (migrated from feature_importance_analysis.py)
        self._create_advanced_feature_visualizations(df_importance, output_dir, timestamp)
        
        # 3. Feature categorization analysis
        self._analyze_feature_categories(df_importance, output_dir, timestamp)
        
        # 4. Gap features analysis
        self._analyze_gap_features(df_importance, output_dir, timestamp)
        
        # 5. Feature consistency analysis
        self._analyze_feature_consistency(df_importance, output_dir, timestamp)
        
    def _get_top_features_by_model(self, df_importance: pd.DataFrame) -> dict:
        """Get top 5 features for each model."""
        top_features_by_model = {}
        for model in df_importance['model'].unique():
            model_data = df_importance[df_importance['model'] == model]
            top_features = model_data.nlargest(5, 'importance')
            top_features_by_model[model] = top_features
        return top_features_by_model
        
    def _save_top_features_summary(self, top_features_by_model: dict, output_dir: Path, timestamp: str):
        """Save top features summary to CSV."""
        summary_records = []
        
        for model, top_features in top_features_by_model.items():
            for idx, (_, row) in enumerate(top_features.iterrows()):
                summary_records.append({
                    'model': model,
                    'rank': idx + 1,
                    'feature': row['feature'],
                    'importance': row['importance']
                })
        
        df_summary = pd.DataFrame(summary_records)
        
        # Round numerical columns to 2 decimal places for cleaner CSV output
        if 'importance' in df_summary.columns:
            df_summary['importance'] = df_summary['importance'].round(2)
        
        df_summary.to_csv(output_dir / f'top_features_by_model_{timestamp}.csv', index=False)
        print(f"📊 Top features summary saved")
    
    def _create_advanced_feature_visualizations(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create advanced feature visualizations migrated from feature_importance_analysis.py."""
        
        # Get unique features and models
        feature_names = df_importance['feature'].unique()
        model_names = df_importance['model'].unique()
        
        # Create pivot table for easier analysis
        pivot_data = df_importance.pivot_table(
            index='feature', 
            columns='model', 
            values='importance', 
            fill_value=0
        )
        
        # 1. Advanced heatmap with top features across all models
        self._create_advanced_heatmap(pivot_data, output_dir, timestamp)
        
        # 2. Feature importance comparison bar charts
        self._create_feature_comparison_charts(df_importance, output_dir, timestamp)
        
        # 3. Model comparison by feature type
        self._create_model_feature_type_analysis(df_importance, output_dir, timestamp)
        
    def _create_advanced_heatmap(self, pivot_data: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create advanced heatmap showing top features across all models."""
        
        # Get top 25 features by sum of importance across all models
        feature_sums = pivot_data.sum(axis=1).sort_values(ascending=False)
        top_features = feature_sums.head(25).index.tolist()
        
        # Filter to top features
        heatmap_data = pivot_data.loc[top_features]
        
        # Normalize each model's importance to 0-1 scale for fair comparison
        normalized_data = heatmap_data.copy()
        for col in normalized_data.columns:
            max_val = normalized_data[col].max()
            if max_val > 0:
                normalized_data[col] = normalized_data[col] / max_val
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Highlight specific features of interest (including new features)
        specific_features = [
            'student_learning_rate', 'student_ability', 'avg_difficulty',
            'performance_vs_class_mean', 'class_percentile_rank', 'starting_ability_quartile'
        ]
        
        # Create color map for y-axis labels
        feature_colors = []
        for feature in normalized_data.index:
            if any(spec_feat in feature for spec_feat in specific_features):
                feature_colors.append('red')
            elif 'gap' in feature.lower():
                feature_colors.append('blue')
            else:
                feature_colors.append('black')
        
        # Create heatmap
        sns.heatmap(
            normalized_data, 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Normalized Importance'},
            ax=ax,
            annot=True,
            fmt='.2f',
            cbar=True
        )
        
        ax.set_title('Feature Importance Heatmap (Top 25 Features)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Color the y-axis labels
        y_tick_labels = ax.get_yticklabels()
        for label, color in zip(y_tick_labels, feature_colors):
            label.set_color(color)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Key Student Features'),
            Patch(facecolor='blue', label='Gap Features'),
            Patch(facecolor='black', label='Other Features')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'advanced_feature_importance_heatmap_{timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_feature_comparison_charts(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create bar charts comparing feature importance across models."""
        
        # Focus on key features (including new class-level and prior achievement features)
        specific_features = [
            'student_learning_rate', 'student_ability', 'avg_difficulty',  # Original key features
            'performance_vs_class_mean', 'class_percentile_rank',  # Class comparison features
            'starting_ability_quartile', 'performance_consistency_score'  # Prior achievement features
        ]
        
        # Find features that contain our key terms
        key_feature_data = []
        for target_feature in specific_features:
            matching_features = df_importance[
                df_importance['feature'].str.contains(target_feature, case=False, na=False)
            ]
            if len(matching_features) > 0:
                # Get the highest importance match for each model
                for model in df_importance['model'].unique():
                    model_matches = matching_features[matching_features['model'] == model]
                    if len(model_matches) > 0:
                        best_match = model_matches.loc[model_matches['importance'].idxmax()]
                        key_feature_data.append({
                            'target_feature': target_feature,
                            'model': model,
                            'feature': best_match['feature'],
                            'importance': best_match['importance']
                        })
        
        if key_feature_data:
            df_key = pd.DataFrame(key_feature_data)
            
            # Create subplot for each target feature
            fig, axes = plt.subplots(1, len(specific_features), figsize=(18, 6))
            if len(specific_features) == 1:
                axes = [axes]
            
            for i, target_feature in enumerate(specific_features):
                ax = axes[i]
                feature_data = df_key[df_key['target_feature'] == target_feature]
                
                if len(feature_data) > 0:
                    # Create bar plot
                    models = feature_data['model'].tolist()
                    importances = feature_data['importance'].tolist()
                    
                    bars = ax.bar(range(len(models)), importances, 
                                 color='steelblue', alpha=0.7)
                    
                    ax.set_xticks(range(len(models)))
                    ax.set_xticklabels(models, rotation=45, ha='right')
                    ax.set_ylabel('Feature Importance')
                    ax.set_title(f'{target_feature.replace("_", " ").title()}\nImportance Across Models')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, importance in zip(bars, importances):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{importance:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'key_features_comparison_{timestamp}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    def _analyze_feature_categories(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Analyze feature importance by categories (migrated from feature_importance_analysis.py)."""
        
        print("Analyzing feature importance by categories...")
        
        # Define feature categories
        categories = {
            'Current': [],
            'Lag': [],
            'Statistical': [],
            'Change': [],
            'Gap': [],
            'Goal': [],
            'Student': [],
            'Class_Peer': [],     # NEW: Class-level and peer comparison features
            'Prior_Achievement': [], # NEW: Prior achievement and baseline features
            'Interaction': []
        }
        
        # Categorize features
        for feature in df_importance['feature'].unique():
            feature_lower = feature.lower()
            
            if 'current_' in feature_lower:
                categories['Current'].append(feature)
            elif 'lag' in feature_lower:
                categories['Lag'].append(feature)
            elif any(stat in feature_lower for stat in ['mean', 'std', 'sum', 'range', 'iqr', 'trend', 'acceleration']):
                categories['Statistical'].append(feature)
            elif 'change' in feature_lower:
                categories['Change'].append(feature)
            elif 'gap' in feature_lower:
                categories['Gap'].append(feature)
            elif 'goal' in feature_lower:
                categories['Goal'].append(feature)
            elif any(student_term in feature_lower for student_term in ['student', 'ability', 'learning_rate']):
                categories['Student'].append(feature)
            elif any(class_term in feature_lower for class_term in ['class', 'percentile', 'performance_vs']):
                categories['Class_Peer'].append(feature)  # NEW: Class/peer features
            elif any(prior_term in feature_lower for prior_term in ['starting', 'consistency', 'acceleration_capacity', 'quartile']):
                categories['Prior_Achievement'].append(feature)  # NEW: Prior achievement features
            elif 'x' in feature or '_*_' in feature:
                categories['Interaction'].append(feature)
        
        # Calculate average importance by category for each model
        category_importance = {}
        for model in df_importance['model'].unique():
            model_data = df_importance[df_importance['model'] == model]
            model_category_imp = {}
            
            for cat_name, features in categories.items():
                if features:
                    cat_importances = []
                    for feature in features:
                        feature_data = model_data[model_data['feature'] == feature]
                        if len(feature_data) > 0:
                            cat_importances.append(feature_data['importance'].iloc[0])
                    
                    if cat_importances:
                        model_category_imp[cat_name] = np.mean(cat_importances)
                    else:
                        model_category_imp[cat_name] = 0
                else:
                    model_category_imp[cat_name] = 0
            
            category_importance[model] = model_category_imp
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cat_df = pd.DataFrame(category_importance).T
        cat_df.plot(kind='bar', ax=ax)
        
        ax.set_title('Average Feature Importance by Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Importance')
        ax.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_by_category_{timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save category analysis to CSV
        # Round numerical columns to 2 decimal places for cleaner CSV output
        cat_df = cat_df.round(2)
        cat_df.to_csv(output_dir / f'feature_categories_analysis_{timestamp}.csv')
        
        print(f"📊 Feature category analysis saved")
        
    def _analyze_gap_features(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Analyze gap features specifically (migrated from feature_importance_analysis.py)."""
        
        print("Analyzing gap features...")
        
        # Find gap features
        gap_features = df_importance[
            df_importance['feature'].str.contains('gap', case=False, na=False)
        ]['feature'].unique()
        
        if len(gap_features) == 0:
            print("⚠️  No gap features found")
            return
            
        # Create gap features analysis
        gap_analysis = []
        
        for gap_feature in gap_features:
            gap_data = df_importance[df_importance['feature'] == gap_feature]
            
            # Calculate statistics across models
            avg_importance = gap_data['importance'].mean()
            std_importance = gap_data['importance'].std()
            max_importance = gap_data['importance'].max()
            
            # Calculate average rank across models
            avg_rank = 0
            rank_count = 0
            
            for model in gap_data['model'].unique():
                model_all_features = df_importance[df_importance['model'] == model]
                sorted_features = model_all_features.sort_values('importance', ascending=False)
                rank = sorted_features.index[sorted_features['feature'] == gap_feature].tolist()
                if rank:
                    avg_rank += rank[0] + 1  # 1-indexed rank
                    rank_count += 1
            
            if rank_count > 0:
                avg_rank = avg_rank / rank_count
            
            gap_analysis.append({
                'gap_feature': gap_feature,
                'avg_importance': avg_importance,
                'std_importance': std_importance,
                'max_importance': max_importance,
                'avg_rank': avg_rank,
                'models_count': len(gap_data)
            })
        
        # Convert to DataFrame and save
        df_gap = pd.DataFrame(gap_analysis)
        df_gap = df_gap.sort_values('avg_importance', ascending=False)
        
        # Round numerical columns to 2 decimal places for cleaner CSV output
        numerical_columns = ['avg_importance', 'std_importance', 'avg_rank']
        for col in numerical_columns:
            if col in df_gap.columns:
                df_gap[col] = df_gap[col].round(2)
        
        df_gap.to_csv(output_dir / f'gap_features_analysis_{timestamp}.csv', index=False)
        
        # Create visualization
        if len(df_gap) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Average importance
            ax1 = axes[0]
            bars = ax1.bar(range(len(df_gap)), df_gap['avg_importance'], 
                          yerr=df_gap['std_importance'], capsize=5, 
                          color='skyblue', alpha=0.7)
            ax1.set_xticks(range(len(df_gap)))
            ax1.set_xticklabels(df_gap['gap_feature'], rotation=45, ha='right')
            ax1.set_ylabel('Average Importance')
            ax1.set_title('Gap Features: Average Importance Across Models')
            ax1.grid(True, alpha=0.3)
            
            # Average rank
            ax2 = axes[1]
            bars = ax2.bar(range(len(df_gap)), df_gap['avg_rank'], 
                          color='lightcoral', alpha=0.7)
            ax2.set_xticks(range(len(df_gap)))
            ax2.set_xticklabels(df_gap['gap_feature'], rotation=45, ha='right')
            ax2.set_ylabel('Average Rank (lower is better)')
            ax2.set_title('Gap Features: Average Rank Across Models')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()  # Lower ranks at top
            
            plt.tight_layout()
            plt.savefig(output_dir / f'gap_features_analysis_{timestamp}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Gap features analysis completed: {len(gap_features)} gap features found")
        
    def _create_model_feature_type_analysis(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create analysis showing which models prioritize which types of features."""
        
        # Get top 5 features for each model
        model_top_features = {}
        for model in df_importance['model'].unique():
            model_data = df_importance[df_importance['model'] == model]
            top_5 = model_data.nlargest(5, 'importance')
            model_top_features[model] = top_5['feature'].tolist()
        
        # Create summary
        summary_data = []
        specific_features = [
            'student_learning_rate', 'student_ability', 'avg_difficulty',  # Original key features
            'performance_vs_class_mean', 'class_percentile_rank',  # Class comparison features  
            'starting_ability_quartile'  # Prior achievement features
        ]
        
        for model, top_features in model_top_features.items():
            # Count feature types in top 5
            current_count = sum(1 for f in top_features if 'current_' in f.lower())
            lag_count = sum(1 for f in top_features if 'lag' in f.lower())
            stat_count = sum(1 for f in top_features if any(stat in f.lower() 
                            for stat in ['mean', 'std', 'sum', 'range', 'trend']))
            gap_count = sum(1 for f in top_features if 'gap' in f.lower())
            student_count = sum(1 for f in top_features if any(term in f.lower() 
                               for term in ['student', 'ability', 'learning_rate']))
            
            # Check if specific features are in top 5
            has_student_lr = any('student_learning_rate' in f for f in top_features)
            has_student_ability = any('student_ability' in f for f in top_features)
            has_avg_difficulty = any('avg_difficulty' in f for f in top_features)
            
            # Check for new class-level and prior achievement features
            has_class_comparison = any('performance_vs_class_mean' in f for f in top_features)
            has_peer_ranking = any('class_percentile_rank' in f for f in top_features)
            has_starting_ability = any('starting_ability_quartile' in f for f in top_features)
            
            # Count new feature types
            class_peer_count = sum(1 for f in top_features if any(term in f.lower() 
                                 for term in ['class', 'percentile', 'performance_vs']))
            prior_achievement_count = sum(1 for f in top_features if any(term in f.lower() 
                                        for term in ['starting', 'consistency', 'acceleration_capacity', 'quartile']))
            
            summary_data.append({
                'model': model,
                'current_features': current_count,
                'lag_features': lag_count,
                'statistical_features': stat_count,
                'gap_features': gap_count,
                'student_features': student_count,
                'class_peer_features': class_peer_count,  # NEW
                'prior_achievement_features': prior_achievement_count,  # NEW
                'has_student_learning_rate': has_student_lr,
                'has_student_ability': has_student_ability,
                'has_avg_difficulty': has_avg_difficulty,
                'has_class_comparison': has_class_comparison,  # NEW
                'has_peer_ranking': has_peer_ranking,  # NEW
                'has_starting_ability': has_starting_ability,  # NEW
                'top_5_features': ', '.join(top_features)
            })
        
        # Save summary
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / f'model_feature_preferences_{timestamp}.csv', index=False)
        
        print(f"📊 Model feature type preferences saved")
    
    def _analyze_specific_features(self, df_importance: pd.DataFrame, specific_features: List[str], output_dir: Path, timestamp: str):
        """Analyze the three specific features of interest."""
        
        print(f"Analyzing specific features: {specific_features}")
        
        specific_analysis = []
        
        for target_feature in specific_features:
            # Find features that contain the target feature name
            matching_features = df_importance[
                df_importance['feature'].str.contains(target_feature, case=False, na=False)
            ]
            
            if len(matching_features) > 0:
                # Calculate statistics for this feature
                feature_stats = matching_features.groupby('model').agg({
                    'importance': ['mean', 'std', 'min', 'max']
                }).round(4)
                
                feature_stats.columns = ['mean_importance', 'std_importance', 'min_importance', 'max_importance']
                
                # Add to analysis
                for model in feature_stats.index:
                    specific_analysis.append({
                        'target_feature': target_feature,
                        'model': model,
                        'mean_importance': feature_stats.loc[model, 'mean_importance'],
                        'std_importance': feature_stats.loc[model, 'std_importance'],
                        'min_importance': feature_stats.loc[model, 'min_importance'],
                        'max_importance': feature_stats.loc[model, 'max_importance']
                    })
        
        if specific_analysis:
            df_specific = pd.DataFrame(specific_analysis)
            df_specific.to_csv(output_dir / f'specific_features_analysis_{timestamp}.csv', index=False)
            
            # Create specific features plot
            self._create_specific_features_plot(df_specific, output_dir, timestamp)
            
            print(f"✅ Specific features analysis completed")
        else:
            print("⚠️  No data found for specific features")
    
    def _create_specific_features_plot(self, df_specific: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create plot focusing on the specific features."""
        
        target_features = df_specific['target_feature'].unique()
        n_features = len(target_features)
        
        if n_features == 0:
            print("⚠️  No specific features found for plotting")
            return
        
        # Calculate grid dimensions
        n_cols = min(3, n_features)  # Max 3 columns
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Handle single subplot case
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, target_feature in enumerate(target_features):
            ax = axes[i]
            feature_data = df_specific[df_specific['target_feature'] == target_feature]
            
            # Create bar plot
            models = feature_data['model'].tolist()
            importances = feature_data['mean_importance'].tolist()
            errors = feature_data['std_importance'].tolist()
            
            bars = ax.bar(range(len(models)), importances, yerr=errors, 
                         capsize=5, alpha=0.7, color='steelblue')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Feature Importance')
            ax.set_title(f'{target_feature}\nImportance Across Models')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, importance in zip(bars, importances):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{importance:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        total_subplots = n_rows * n_cols
        for i in range(n_features, total_subplots):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'specific_features_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compute_bootstrap_confidence_intervals(self, n_bootstrap: int = 1000, confidence_level: float = 0.95) -> pd.DataFrame:
        """Compute focused bootstrap analysis comparing model categories to baselines."""
        print("Computing focused bootstrap analysis: model categories vs baselines...")
        
        # Define model categories
        baselines = [
            'median_no_zeros', 'median_all', 'mean_no_zeros', 'average_all', 
            'naive_forecast', 'adams_baseline_50', 'adams_baseline_60', 'adams_baseline_70'
        ]
        
        # Get available models
        available_models = set(self.prediction_df['model'].unique())
        
        # Filter baselines to only available ones
        available_baselines = [b for b in baselines if b in available_models]
        
        # Categorize remaining models
        trees = []
        linear = []
        neural = []
        other = []
        
        for model in available_models:
            if model in available_baselines:
                continue
            elif model in self.results_data:
                category = self.results_data[model].get('category', 'unknown').lower()
                if 'tree' in category or 'forest' in category or 'xgb' in category or 'gradient' in category:
                    trees.append(model)
                elif 'linear' in category:
                    linear.append(model)
                elif 'neural' in category or 'mlp' in category or 'lstm' in category:
                    neural.append(model)
                else:
                    other.append(model)
            else:
                # Try to infer from model name
                model_lower = model.lower()
                if any(tree_term in model_lower for tree_term in ['forest', 'tree', 'xgb', 'gradient']):
                    trees.append(model)
                elif any(linear_term in model_lower for linear_term in ['linear', 'ridge', 'lasso', 'elastic']):
                    linear.append(model)
                elif any(neural_term in model_lower for neural_term in ['mlp', 'lstm', 'neural', 'dlinear']):
                    neural.append(model)
                else:
                    other.append(model)
        
        print(f"📊 Model categorization:")
        print(f"   Baselines: {available_baselines}")
        print(f"   Trees: {trees}")
        print(f"   Linear: {linear}")
        print(f"   Neural: {neural}")
        print(f"   Other: {other}")
        
        # Calculate baseline average performance with CI
        baseline_results = self._calculate_category_bootstrap(available_baselines, n_bootstrap, confidence_level)
        
        # Results list for summary table
        summary_results = []
        
        # Add baseline row
        summary_results.append({
            'category': 'Baseline',
            'comparison_type': 'Average',
            'best_model': f"Average of {len(available_baselines)} models",
            'mae_mean': baseline_results['mae_mean'],
            'mae_ci_lower': baseline_results['mae_ci_lower'],
            'mae_ci_upper': baseline_results['mae_ci_upper'],
            'mae_ci': f"{baseline_results['mae_mean']:.2f} ({baseline_results['mae_ci_lower']:.2f}-{baseline_results['mae_ci_upper']:.2f})",
            'vs_baseline_pct': 0.0,
            'p_value': np.nan,
            'n_models': len(available_baselines)
        })
        
        # Analyze each category
        for category_name, models in [('Trees', trees), ('Linear', linear), ('Neural', neural)]:
            if not models:
                continue
                
            # Best model in category
            best_model = self._get_best_model_in_category(models)
            best_results = self._calculate_model_bootstrap(best_model, n_bootstrap, confidence_level)
            best_vs_baseline = self._compare_to_baseline(best_model, available_baselines)
            
            summary_results.append({
                'category': category_name,
                'comparison_type': 'Best',
                'best_model': best_model,
                'mae_mean': best_results['mae_mean'],
                'mae_ci_lower': best_results['mae_ci_lower'],
                'mae_ci_upper': best_results['mae_ci_upper'],
                'mae_ci': f"{best_results['mae_mean']:.2f} ({best_results['mae_ci_lower']:.2f}-{best_results['mae_ci_upper']:.2f})",
                'vs_baseline_pct': best_vs_baseline['improvement_pct'],
                'p_value': best_vs_baseline['p_value'],
                'n_models': 1
            })
            
            # Average of category
            avg_results = self._calculate_category_bootstrap(models, n_bootstrap, confidence_level)
            avg_vs_baseline = self._compare_category_to_baseline(models, available_baselines)
            
            summary_results.append({
                'category': category_name,
                'comparison_type': 'Average',
                'best_model': f"Average of {len(models)} models",
                'mae_mean': avg_results['mae_mean'],
                'mae_ci_lower': avg_results['mae_ci_lower'],
                'mae_ci_upper': avg_results['mae_ci_upper'],
                'mae_ci': f"{avg_results['mae_mean']:.2f} ({avg_results['mae_ci_lower']:.2f}-{avg_results['mae_ci_upper']:.2f})",
                'vs_baseline_pct': avg_vs_baseline['improvement_pct'],
                'p_value': avg_vs_baseline['p_value'],
                'n_models': len(models)
            })
        
        # Convert to DataFrame and round
        summary_df = pd.DataFrame(summary_results)
        
        # Round numerical columns
        numerical_columns = ['mae_mean', 'mae_ci_lower', 'mae_ci_upper', 'vs_baseline_pct', 'p_value']
        for col in numerical_columns:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(2)
        
        return summary_df
    
    def _calculate_model_bootstrap(self, model_name: str, n_bootstrap: int, confidence_level: float) -> dict:
        """Calculate bootstrap CI for a single model."""
        model_data = self.prediction_df[self.prediction_df['model'] == model_name]
        errors = model_data['error'].values
        
        bootstrap_maes = []
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(len(errors), size=len(errors), replace=True)
            bootstrap_errors = errors[bootstrap_indices]
            bootstrap_maes.append(np.mean(bootstrap_errors))
        
        alpha = 1 - confidence_level
        mae_ci_lower = np.percentile(bootstrap_maes, (alpha/2) * 100)
        mae_ci_upper = np.percentile(bootstrap_maes, (1 - alpha/2) * 100)
        
        return {
            'mae_mean': np.mean(errors),
            'mae_ci_lower': mae_ci_lower,
            'mae_ci_upper': mae_ci_upper
        }
    
    def _calculate_category_bootstrap(self, models: list, n_bootstrap: int, confidence_level: float) -> dict:
        """Calculate bootstrap CI for average performance across a category."""
        # Get all errors from all models in category
        all_category_errors = []
        for model in models:
            model_data = self.prediction_df[self.prediction_df['model'] == model]
            all_category_errors.extend(model_data['error'].values)
        
        all_category_errors = np.array(all_category_errors)
        
        bootstrap_maes = []
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(len(all_category_errors), size=len(all_category_errors), replace=True)
            bootstrap_errors = all_category_errors[bootstrap_indices]
            bootstrap_maes.append(np.mean(bootstrap_errors))
        
        alpha = 1 - confidence_level
        mae_ci_lower = np.percentile(bootstrap_maes, (alpha/2) * 100)
        mae_ci_upper = np.percentile(bootstrap_maes, (1 - alpha/2) * 100)
        
        return {
            'mae_mean': np.mean(all_category_errors),
            'mae_ci_lower': mae_ci_lower,
            'mae_ci_upper': mae_ci_upper
        }
    
    def _get_best_model_in_category(self, models: list) -> str:
        """Get the best performing model in a category."""
        best_mae = float('inf')
        best_model = None
        
        for model in models:
            model_data = self.prediction_df[self.prediction_df['model'] == model]
            mae = np.mean(model_data['error'].values)
            if mae < best_mae:
                best_mae = mae
                best_model = model
        
        return best_model
    
    def _compare_to_baseline(self, model_name: str, baseline_models: list) -> dict:
        """Compare a single model to baseline average."""
        # Get model errors
        model_data = self.prediction_df[self.prediction_df['model'] == model_name]
        model_errors = model_data['error'].values
        
        # Get baseline errors (average across all baseline models)
        baseline_errors = []
        for baseline in baseline_models:
            baseline_data = self.prediction_df[self.prediction_df['model'] == baseline]
            baseline_errors.extend(baseline_data['error'].values)
        baseline_errors = np.array(baseline_errors)
        
        # Calculate improvement
        model_mae = np.mean(model_errors)
        baseline_mae = np.mean(baseline_errors)
        improvement_pct = ((baseline_mae - model_mae) / baseline_mae) * 100
        
        # Statistical test (Mann-Whitney U test)
        try:
            _, p_value = stats.mannwhitneyu(model_errors, baseline_errors, alternative='less')
        except:
            p_value = np.nan
        
        return {
            'improvement_pct': improvement_pct,
            'p_value': p_value
        }
    
    def _compare_category_to_baseline(self, category_models: list, baseline_models: list) -> dict:
        """Compare category average to baseline average."""
        # Get category errors
        category_errors = []
        for model in category_models:
            model_data = self.prediction_df[self.prediction_df['model'] == model]
            category_errors.extend(model_data['error'].values)
        category_errors = np.array(category_errors)
        
        # Get baseline errors
        baseline_errors = []
        for baseline in baseline_models:
            baseline_data = self.prediction_df[self.prediction_df['model'] == baseline]
            baseline_errors.extend(baseline_data['error'].values)
        baseline_errors = np.array(baseline_errors)
        
        # Calculate improvement
        category_mae = np.mean(category_errors)
        baseline_mae = np.mean(baseline_errors)
        improvement_pct = ((baseline_mae - category_mae) / baseline_mae) * 100
        
        # Statistical test
        try:
            _, p_value = stats.mannwhitneyu(category_errors, baseline_errors, alternative='less')
        except:
            p_value = np.nan
        
        return {
            'improvement_pct': improvement_pct,
            'p_value': p_value
        }
    
    def create_summary_report(self, output_dir: Path, timestamp: str, 
                            performance_summary: pd.DataFrame, 
                            significance_results: pd.DataFrame):
        """Create a comprehensive summary report."""
        report_path = output_dir / f'analysis_summary_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Comprehensive Model Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Evaluation configuration
            f.write("## Evaluation Configuration\n\n")
            if self.evaluation_config:
                for key, value in self.evaluation_config.items():
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Performance summary
            f.write("## Model Performance Summary\n\n")
            f.write("### Top 5 Models by MAE\n\n")
            top_5 = performance_summary.head(5)
            f.write(top_5[['model', 'category', 'mae_mean', 'mae_std', 'rmse_mean', 'r2_mean']].to_string(index=False))
            f.write("\n\n")
            
            # Significance testing summary
            f.write("## Statistical Significance Testing\n\n")
            significant_pairs = significance_results[significance_results['is_significant'] == True]
            f.write(f"Total pairwise comparisons: {len(significance_results)}\n")
            f.write(f"Significant differences (p < 0.05): {len(significant_pairs)}\n\n")
            
            if len(significant_pairs) > 0:
                f.write("### Top 5 Most Significant Differences\n\n")
                top_significant = significant_pairs.nsmallest(5, 'p_value')
                for _, row in top_significant.iterrows():
                    f.write(f"- **{row['model1']}** vs **{row['model2']}**: ")
                    f.write(f"p = {row['p_value']:.2f}, effect size = {row['effect_size']}\n")
                f.write("\n")
            
            # Category analysis
            if 'category' in performance_summary.columns:
                f.write("## Performance by Category\n\n")
                category_stats = performance_summary.groupby('category').agg({
                    'mae_mean': ['mean', 'std', 'min', 'max', 'count']
                }).round(2)
                
                for category in category_stats.index:
                    stats = category_stats.loc[category]
                    f.write(f"### {category.title()}\n")
                    f.write(f"- Number of models: {int(stats[('mae_mean', 'count')])}\n")
                    f.write(f"- Average MAE: {stats[('mae_mean', 'mean')]:.2f} ± {stats[('mae_mean', 'std')]:.2f}\n")
                    f.write(f"- Best MAE: {stats[('mae_mean', 'min')]:.2f}\n")
                    f.write(f"- Worst MAE: {stats[('mae_mean', 'max')]:.2f}\n\n")
            
            # Feature importance summary
            f.write("## Advanced Feature Importance Analysis\n\n")
            f.write("Comprehensive feature importance analysis using methods migrated from feature_importance_analysis.py:\n\n")
            f.write("### Key Analysis Components\n")
            f.write("- **Advanced Heatmap**: Top 25 features across all models with normalized importance\n")
            f.write("- **Feature Categorization**: Features grouped by type (Current, Lag, Statistical, Change, Gap, Goal, Student, Interaction)\n")
            f.write("- **Gap Features Analysis**: Dedicated analysis of temporal gap features with ranking\n")
            f.write("- **Model Feature Preferences**: Which models prioritize which types of features\n")
            f.write("- **Key Student Features**: Focused analysis on critical student modeling features\n\n")
            f.write("### Special Focus Features\n")
            f.write("- **student_learning_rate**: Student's learning progression rate\n")
            f.write("- **student_ability**: Current student proficiency level\n")
            f.write("- **avg_difficulty**: Average difficulty of practice content\n")
            f.write("- **Gap features**: Temporal discontinuity indicators\n\n")
            
            # Window and architecture analysis summary
            f.write("## Window and Architecture Analysis\n\n")
            f.write("Analysis of window size and architecture combinations provides insights into:\n")
            f.write("- **Performance by window size**: How sequence length affects model accuracy\n")
            f.write("- **Performance by architecture**: Which model types perform best\n")
            f.write("- **Top combinations**: Best window + architecture pairings\n")
            f.write("- **Statistical significance**: Whether top combinations are significantly better\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            best_model = performance_summary.iloc[0]
            f.write(f"1. **Best Overall Model**: {best_model['model']} ({best_model['category']})\n")
            f.write(f"   - MAE: {best_model['mae_mean']:.2f} ± {best_model['mae_std']:.2f}\n")
            f.write(f"   - RMSE: {best_model['rmse_mean']:.2f}\n")
            f.write(f"   - R²: {best_model['r2_mean']:.2f}\n\n")
            
            # Find most stable model (lowest std)
            stable_model = performance_summary.loc[performance_summary['mae_std'].idxmin()]
            f.write(f"2. **Most Stable Model**: {stable_model['model']} ({stable_model['category']})\n")
            f.write(f"   - MAE: {stable_model['mae_mean']:.2f} ± {stable_model['mae_std']:.2f}\n")
            f.write(f"   - Consistency across folds is highest\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("### Performance Analysis\n")
            f.write("- `model_performance_summary.csv`: Detailed performance metrics\n")
            f.write("- `significance_testing.csv`: Statistical significance test results\n")
            f.write("- `key_bootstrap_findings.csv`: Category-based bootstrap comparison vs baselines\n")
            f.write("- `predicted_vs_actual.png`: Scatter plots of predictions vs actuals\n")
            f.write("- `error_distributions.png`: Error distribution histograms\n")
            f.write("- `residual_plots.png`: Residual analysis plots\n")
            f.write("- `model_comparison.png`: Performance comparison charts\n")
            f.write("- `performance_by_category.png`: Category-wise performance analysis\n\n")
            f.write("### Advanced Feature Importance Analysis\n")
            f.write("- `advanced_feature_importance_heatmap.png`: Top 25 features with normalized importance and color coding\n")
            f.write("- `key_features_comparison.png`: Comparison of key student features across models\n")
            f.write("- `feature_importance_by_category.png`: Average importance by feature category\n")
            f.write("- `gap_features_analysis.png`: Gap features importance and ranking analysis\n")
            f.write("- `top_features_by_model.csv`: Top 5 features for each model\n")
            f.write("- `feature_categories_analysis.csv`: Feature importance by category data\n")
            f.write("- `gap_features_analysis.csv`: Detailed gap features statistics\n")
            f.write("- `model_feature_preferences.csv`: Which feature types each model prioritizes\n")
            f.write("- `feature_consistency_analysis.csv`: Feature consistency statistics across models\n")
            f.write("- `specific_features_analysis.csv`: Detailed analysis of key student features\n\n")
            f.write("### Window and Architecture Analysis\n")
            f.write("- `window_aggregation.csv`: Performance aggregated by window size\n")
            f.write("- `architecture_aggregation.csv`: Performance aggregated by architecture\n")
            f.write("- `combination_aggregation.csv`: Performance aggregated by window+architecture\n")
            f.write("- `top_3_combinations.csv`: Top 3 window+architecture combinations\n")
            f.write("- `top_combinations_significance.csv`: Statistical significance test results\n")
            f.write("- `window_architecture_analysis.png`: Performance by window and architecture\n")
            f.write("- `combination_heatmap.png`: Window×Architecture performance heatmap\n")
        
        print(f"📋 Summary report saved to: {report_path}")
    
    def analyze_window_architecture_combinations(self, output_dir: Path, timestamp: str) -> pd.DataFrame:
        """Analyze window size and architecture combinations."""
        print("Analyzing window size and architecture combinations...")
        
        # Extract window and architecture info from model names and results
        combination_data = []
        
        for model_name in self.prediction_df['model'].unique():
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            # Calculate performance metrics
            mae = np.mean(model_data['error'].values)
            rmse = np.sqrt(np.mean(model_data['squared_error'].values))
            
            # Try to extract window size from model name or results
            window_size = 'unknown'
            if hasattr(self, 'evaluation_config') and 'window_size' in self.evaluation_config:
                window_size = self.evaluation_config['window_size']
            
            # Determine architecture type
            architecture = 'other'
            model_lower = model_name.lower()
            
            if any(term in model_lower for term in ['linear', 'ridge', 'lasso', 'elastic']):
                architecture = 'linear'
            elif any(term in model_lower for term in ['forest', 'tree', 'xgb', 'gradient']):
                architecture = 'tree'
            elif any(term in model_lower for term in ['mlp', 'lstm', 'neural', 'dlinear']):
                architecture = 'neural'
            elif any(term in model_lower for term in ['mixed', 'hierarchical']):
                architecture = 'mixed_effects'
            elif any(term in model_lower for term in ['baseline', 'naive', 'mean', 'median']):
                architecture = 'baseline'
            
            combination_data.append({
                'model': model_name,
                'window_size': window_size,
                'architecture': architecture,
                'mae': mae,
                'rmse': rmse,
                'n_predictions': len(model_data)
            })
        
        combination_df = pd.DataFrame(combination_data)
        
        # Save the analysis
        combination_df.to_csv(output_dir / f'window_architecture_analysis_{timestamp}.csv', index=False)
        
        # Create aggregated summaries
        if len(combination_df) > 1:
            # Aggregate by architecture
            arch_summary = combination_df.groupby('architecture').agg({
                'mae': ['mean', 'std', 'min', 'max', 'count']
            }).round(3)
            arch_summary.to_csv(output_dir / f'architecture_aggregation_{timestamp}.csv')
            
            # If we have multiple window sizes, aggregate by window
            if len(combination_df['window_size'].unique()) > 1:
                window_summary = combination_df.groupby('window_size').agg({
                    'mae': ['mean', 'std', 'min', 'max', 'count']
                }).round(3)
                window_summary.to_csv(output_dir / f'window_aggregation_{timestamp}.csv')
        
        print(f"✅ Window and architecture analysis saved")
        return combination_df
    
    def _analyze_feature_consistency(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Analyze feature consistency across models (migrated from feature_importance_analysis.py)."""
        
        print("Analyzing feature consistency across models...")
        
        # Calculate consistency metrics for each feature
        feature_consistency = []
        
        features = df_importance['feature'].unique()
        models = df_importance['model'].unique()
        
        for feature in features:
            feature_data = df_importance[df_importance['feature'] == feature]
            
            if len(feature_data) > 1:  # Feature appears in multiple models
                importances = feature_data['importance'].values
                
                # Calculate consistency metrics
                mean_importance = np.mean(importances)
                std_importance = np.std(importances)
                cv = std_importance / mean_importance if mean_importance > 0 else float('inf')
                min_importance = np.min(importances)
                max_importance = np.max(importances)
                range_importance = max_importance - min_importance
                
                # How many models have this feature in top 10?
                models_with_feature = len(feature_data)
                
                # Calculate average rank across models (if we can determine it)
                avg_rank = None
                ranks = []
                for model in feature_data['model'].unique():
                    model_all_features = df_importance[df_importance['model'] == model]
                    sorted_features = model_all_features.sort_values('importance', ascending=False)
                    feature_rank = sorted_features.index[sorted_features['feature'] == feature].tolist()
                    if feature_rank:
                        ranks.append(feature_rank[0] + 1)  # 1-indexed rank
                
                if ranks:
                    avg_rank = np.mean(ranks)
                
                feature_consistency.append({
                    'feature': feature,
                    'n_models': models_with_feature,
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'coefficient_of_variation': cv,
                    'min_importance': min_importance,
                    'max_importance': max_importance,
                    'range_importance': range_importance,
                    'avg_rank': avg_rank,
                    'consistency_score': 1 / (1 + cv) if cv != float('inf') else 0
                })
        
        # Convert to DataFrame and save
        df_consistency = pd.DataFrame(feature_consistency)
        
        if len(df_consistency) > 0:
            df_consistency = df_consistency.sort_values('consistency_score', ascending=False)
            
            # Round numerical columns to 2 decimal places for cleaner CSV output
            numerical_columns = ['mean_importance', 'std_importance', 'coefficient_of_variation', 
                               'min_importance', 'max_importance', 'range_importance', 'avg_rank', 'consistency_score']
            for col in numerical_columns:
                if col in df_consistency.columns:
                    df_consistency[col] = df_consistency[col].round(2)
            
            df_consistency.to_csv(output_dir / f'feature_consistency_analysis_{timestamp}.csv', index=False)
            
            # Create visualization
            self._create_feature_consistency_plot(df_consistency, output_dir, timestamp)
            
            print(f"📊 Feature consistency analysis completed: {len(df_consistency)} features analyzed")
        else:
            print("⚠️  No features found for consistency analysis")
    
    def _create_feature_consistency_plot(self, df_consistency: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create visualization for feature consistency analysis."""
        
        if len(df_consistency) == 0:
            return
        
        # Get top 15 most consistent features
        top_consistent = df_consistency.head(15)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # 1. Consistency score ranking
        ax1 = axes[0]
        y_pos = np.arange(len(top_consistent))
        consistency_scores = top_consistent['consistency_score'].values
        
        # Color bars based on consistency level
        colors = ['darkgreen' if score > 0.8 else 'green' if score > 0.6 else 
                 'orange' if score > 0.4 else 'red' for score in consistency_scores]
        
        bars = ax1.barh(y_pos, consistency_scores, color=colors, alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_consistent['feature'].values)
        ax1.set_xlabel('Consistency Score (Higher = More Consistent)')
        ax1.set_title('Top 15 Most Consistent Features Across Models')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, consistency_scores)):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', ha='left', va='center', fontsize=8)
        
        # 2. Mean importance vs consistency
        ax2 = axes[1]
        
        x = df_consistency['mean_importance'].values
        y = df_consistency['consistency_score'].values
        
        # Color points based on number of models
        colors = df_consistency['n_models'].values
        
        scatter = ax2.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=50)
        
        ax2.set_xlabel('Mean Importance')
        ax2.set_ylabel('Consistency Score')
        ax2.set_title('Feature Importance vs Consistency')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Number of Models')
        
        # Annotate some interesting points
        for idx, row in df_consistency.head(5).iterrows():
            ax2.annotate(row['feature'][:20] + '...' if len(row['feature']) > 20 else row['feature'], 
                        (row['mean_importance'], row['consistency_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_consistency_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature consistency visualization saved")