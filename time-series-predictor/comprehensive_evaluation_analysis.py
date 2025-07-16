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
        
        # 7. Bootstrap confidence intervals
        print("\n7. Computing bootstrap confidence intervals...")
        bootstrap_results = self.compute_bootstrap_confidence_intervals()
        bootstrap_results.to_csv(output_dir / f'bootstrap_confidence_intervals_{timestamp}.csv')
        
        # 8. Window and Architecture Analysis
        print("\n8. Analyzing window and architecture combinations...")
        window_arch_results = self.analyze_window_architecture_combinations(output_dir, timestamp)
        
        # 9. Create summary report
        print("\n9. Creating summary report...")
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
        
        return pd.DataFrame(significance_results)
    
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
            ax.set_title(f'{model_name}\nMAE: {mae:.3f}, R²: {r2:.3f}')
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
            ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
            ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.3f}')
            
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
        
        # Focus on specific features of interest
        specific_features = ['student_learning_rate', 'student_ability', 'avg_difficulty']
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
        
        # Highlight specific features of interest
        specific_features = ['student_learning_rate', 'student_ability', 'avg_difficulty']
        
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
        
        # Focus on key features
        specific_features = ['student_learning_rate', 'student_ability', 'avg_difficulty']
        
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
                                f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
            
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
        specific_features = ['student_learning_rate', 'student_ability', 'avg_difficulty']
        
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
            
            summary_data.append({
                'model': model,
                'current_features': current_count,
                'lag_features': lag_count,
                'statistical_features': stat_count,
                'gap_features': gap_count,
                'student_features': student_count,
                'has_student_learning_rate': has_student_lr,
                'has_student_ability': has_student_ability,
                'has_avg_difficulty': has_avg_difficulty,
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
        """Create plot focusing on the three specific features."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        target_features = df_specific['target_feature'].unique()
        
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
                        f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'specific_features_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compute_bootstrap_confidence_intervals(self, n_bootstrap: int = 1000, confidence_level: float = 0.95) -> pd.DataFrame:
        """Compute bootstrap confidence intervals for model performance."""
        bootstrap_results = []
        alpha = 1 - confidence_level
        
        for model_name in self.prediction_df['model'].unique():
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            errors = model_data['error'].values
            
            if len(errors) < 10:  # Skip models with too few predictions
                continue
            
            # Bootstrap resampling
            bootstrap_maes = []
            bootstrap_rmses = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_indices = np.random.choice(len(errors), size=len(errors), replace=True)
                bootstrap_errors = errors[bootstrap_indices]
                bootstrap_true = model_data['y_true'].values[bootstrap_indices]
                bootstrap_pred = model_data['y_pred'].values[bootstrap_indices]
                
                bootstrap_mae = np.mean(bootstrap_errors)
                bootstrap_rmse = np.sqrt(np.mean((bootstrap_true - bootstrap_pred)**2))
                
                bootstrap_maes.append(bootstrap_mae)
                bootstrap_rmses.append(bootstrap_rmse)
            
            # Calculate confidence intervals
            mae_ci_lower = np.percentile(bootstrap_maes, (alpha/2) * 100)
            mae_ci_upper = np.percentile(bootstrap_maes, (1 - alpha/2) * 100)
            rmse_ci_lower = np.percentile(bootstrap_rmses, (alpha/2) * 100)
            rmse_ci_upper = np.percentile(bootstrap_rmses, (1 - alpha/2) * 100)
            
            bootstrap_results.append({
                'model': model_name,
                'mae_mean': np.mean(errors),
                'mae_ci_lower': mae_ci_lower,
                'mae_ci_upper': mae_ci_upper,
                'mae_ci_width': mae_ci_upper - mae_ci_lower,
                'rmse_mean': np.sqrt(np.mean((model_data['y_true'] - model_data['y_pred'])**2)),
                'rmse_ci_lower': rmse_ci_lower,
                'rmse_ci_upper': rmse_ci_upper,
                'rmse_ci_width': rmse_ci_upper - rmse_ci_lower,
                'n_bootstrap': n_bootstrap,
                'confidence_level': confidence_level
            })
        
        return pd.DataFrame(bootstrap_results)
    
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
                    f.write(f"p = {row['p_value']:.4f}, effect size = {row['effect_size']}\n")
                f.write("\n")
            
            # Category analysis
            if 'category' in performance_summary.columns:
                f.write("## Performance by Category\n\n")
                category_stats = performance_summary.groupby('category').agg({
                    'mae_mean': ['mean', 'std', 'min', 'max', 'count']
                }).round(3)
                
                for category in category_stats.index:
                    stats = category_stats.loc[category]
                    f.write(f"### {category.title()}\n")
                    f.write(f"- Number of models: {int(stats[('mae_mean', 'count')])}\n")
                    f.write(f"- Average MAE: {stats[('mae_mean', 'mean')]:.3f} ± {stats[('mae_mean', 'std')]:.3f}\n")
                    f.write(f"- Best MAE: {stats[('mae_mean', 'min')]:.3f}\n")
                    f.write(f"- Worst MAE: {stats[('mae_mean', 'max')]:.3f}\n\n")
            
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
            f.write(f"   - MAE: {best_model['mae_mean']:.3f} ± {best_model['mae_std']:.3f}\n")
            f.write(f"   - RMSE: {best_model['rmse_mean']:.3f}\n")
            f.write(f"   - R²: {best_model['r2_mean']:.3f}\n\n")
            
            # Find most stable model (lowest std)
            stable_model = performance_summary.loc[performance_summary['mae_std'].idxmin()]
            f.write(f"2. **Most Stable Model**: {stable_model['model']} ({stable_model['category']})\n")
            f.write(f"   - MAE: {stable_model['mae_mean']:.3f} ± {stable_model['mae_std']:.3f}\n")
            f.write(f"   - Consistency across folds is highest\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("### Performance Analysis\n")
            f.write("- `model_performance_summary.csv`: Detailed performance metrics\n")
            f.write("- `significance_testing.csv`: Statistical significance test results\n")
            f.write("- `bootstrap_confidence_intervals.csv`: Bootstrap confidence intervals\n")
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
    
    def analyze_window_architecture_combinations(self, output_dir: Path, timestamp: str):
        """Analyze window and architecture combinations with statistical significance testing."""
        
        print("Performing window and architecture combination analysis...")
        
        # Extract window and architecture information from model names and categories
        window_arch_data = self._extract_window_architecture_data()
        
        if not window_arch_data:
            print("⚠️  No window/architecture data found - skipping window-architecture analysis")
            return None
        
        # Convert to DataFrame for analysis
        df_window_arch = pd.DataFrame(window_arch_data)
        
        # 1. Aggregate by window and architecture
        aggregated_results = self._aggregate_by_window_architecture(df_window_arch, output_dir, timestamp)
        
        # 2. Identify top 3 combinations
        top_combinations = self._identify_top_combinations(aggregated_results, output_dir, timestamp)
        
        # 3. Statistical significance testing of top 3 vs others
        significance_results = self._test_top_combinations_significance(
            df_window_arch, top_combinations, output_dir, timestamp
        )
        
        # 4. Create visualizations
        self._create_window_architecture_plots(aggregated_results, top_combinations, output_dir, timestamp)
        
        print("✅ Window and architecture analysis completed")
        
        return {
            'aggregated_results': aggregated_results,
            'top_combinations': top_combinations,
            'significance_results': significance_results
        }
    
    def _extract_window_architecture_data(self):
        """Extract window and architecture information from prediction data."""
        window_arch_data = []
        
        for model_name in self.prediction_df['model'].unique():
            model_data = self.prediction_df[self.prediction_df['model'] == model_name]
            
            # Get category from results_data
            category = 'unknown'
            if model_name in self.results_data:
                category = self.results_data[model_name].get('category', 'unknown')
            
            # Try to extract window size from model name or configuration
            window_size = self._extract_window_size(model_name)
            
            # Calculate metrics for this model
            y_true = model_data['y_true'].values
            y_pred = model_data['y_pred'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate rank within this dataset
            all_maes = []
            for other_model in self.prediction_df['model'].unique():
                other_data = self.prediction_df[self.prediction_df['model'] == other_model]
                other_mae = mean_absolute_error(other_data['y_true'], other_data['y_pred'])
                all_maes.append(other_mae)
            
            rank = sorted(all_maes).index(mae) + 1
            
            window_arch_data.append({
                'model': model_name,
                'window_size': window_size,
                'architecture': category,
                'mae': mae,
                'rmse': rmse,
                'rank': rank,
                'combination': f"window_{window_size}_{category}"
            })
        
        return window_arch_data
    
    def _extract_window_size(self, model_name: str) -> int:
        """Extract window size from model name or use default."""
        # Try to extract from model name patterns like "model_window15" or "lstm_w20"
        import re
        
        # Look for patterns like "window15", "w15", "_15_", etc.
        window_patterns = [
            r'window[_-]?(\d+)',
            r'w[_-]?(\d+)',
            r'_(\d+)_',
            r'window(\d+)',
        ]
        
        for pattern in window_patterns:
            match = re.search(pattern, model_name.lower())
            if match:
                return int(match.group(1))
        
        # Check evaluation config for window size (FIXED)
        if 'window_size' in self.evaluation_config:
            return self.evaluation_config['window_size']
        elif 'sequence_length' in self.evaluation_config:
            return self.evaluation_config['sequence_length']
        
        # This should never happen if evaluation_config is properly loaded
        print(f"⚠️  Warning: Could not extract window size for {model_name}, using fallback")
        return 10  # More reasonable fallback than 15
    
    def _aggregate_by_window_architecture(self, df_window_arch: pd.DataFrame, output_dir: Path, timestamp: str):
        """Aggregate results by window size and architecture."""
        
        print("Aggregating by window and architecture...")
        
        # Aggregate by window size
        window_aggregation = df_window_arch.groupby('window_size').agg({
            'mae': ['mean', 'std', 'min', 'max', 'count'],
            'rank': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        window_aggregation.columns = [f"{col[1]}_{col[0]}" for col in window_aggregation.columns]
        window_aggregation = window_aggregation.reset_index()
        
        # Aggregate by architecture
        arch_aggregation = df_window_arch.groupby('architecture').agg({
            'mae': ['mean', 'std', 'min', 'max', 'count'],
            'rank': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        arch_aggregation.columns = [f"{col[1]}_{col[0]}" for col in arch_aggregation.columns]
        arch_aggregation = arch_aggregation.reset_index()
        
        # Aggregate by combination
        combination_aggregation = df_window_arch.groupby(['window_size', 'architecture']).agg({
            'mae': ['mean', 'std', 'min', 'max', 'count'],
            'rank': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        combination_aggregation.columns = [f"{col[1]}_{col[0]}" for col in combination_aggregation.columns]
        combination_aggregation = combination_aggregation.reset_index()
        combination_aggregation['combination'] = combination_aggregation.apply(
            lambda row: f"window_{row['window_size']}_{row['architecture']}", axis=1
        )
        
        # Save aggregated results
        window_aggregation.to_csv(output_dir / f'window_aggregation_{timestamp}.csv', index=False)
        arch_aggregation.to_csv(output_dir / f'architecture_aggregation_{timestamp}.csv', index=False)
        combination_aggregation.to_csv(output_dir / f'combination_aggregation_{timestamp}.csv', index=False)
        
        print(f"📊 Aggregated results saved")
        
        return {
            'by_window': window_aggregation,
            'by_architecture': arch_aggregation,
            'by_combination': combination_aggregation
        }
    
    def _identify_top_combinations(self, aggregated_results: dict, output_dir: Path, timestamp: str):
        """Identify top 3 window + architecture combinations."""
        
        print("Identifying top 3 window + architecture combinations...")
        
        combination_data = aggregated_results['by_combination']
        
        # Sort by average MAE to get top 3
        top_3 = combination_data.nsmallest(3, 'mean_mae').copy()
        top_3['rank_overall'] = range(1, 4)
        
        # Add detailed description
        top_3['description'] = top_3.apply(
            lambda row: f"Window {row['window_size']} + {row['architecture'].title()} "
                       f"(MAE: {row['mean_mae']:.4f}±{row['std_mae']:.4f}, "
                       f"Avg Rank: {row['mean_rank']:.1f})", axis=1
        )
        
        # Save top combinations
        top_3.to_csv(output_dir / f'top_3_combinations_{timestamp}.csv', index=False)
        
        print(f"🏆 Top 3 combinations identified:")
        for _, row in top_3.iterrows():
            print(f"   {row['rank_overall']}. {row['description']}")
        
        return top_3
    
    def _test_top_combinations_significance(self, df_window_arch: pd.DataFrame, 
                                          top_combinations: pd.DataFrame, 
                                          output_dir: Path, timestamp: str):
        """Test statistical significance of top 3 combinations vs others."""
        
        print("Testing statistical significance of top 3 combinations...")
        
        # Get top 3 combination identifiers
        top_3_combinations = set(top_combinations['combination'].values)
        
        # Separate data into top 3 and others
        df_window_arch['is_top_3'] = df_window_arch['combination'].isin(top_3_combinations)
        
        top_3_data = df_window_arch[df_window_arch['is_top_3']]
        other_data = df_window_arch[~df_window_arch['is_top_3']]
        
        if len(other_data) == 0:
            print("⚠️  No other combinations to compare against")
            return None
        
        # Statistical tests
        significance_results = []
        
        # 1. Overall comparison: Top 3 vs Others
        top_3_maes = top_3_data['mae'].values
        other_maes = other_data['mae'].values
        
        # Perform statistical tests
        try:
            # Mann-Whitney U test (non-parametric)
            mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(
                top_3_maes, other_maes, alternative='less'
            )
            
            # Welch's t-test (assumes unequal variances)
            ttest_stat, ttest_p = stats.ttest_ind(
                top_3_maes, other_maes, equal_var=False, alternative='less'
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(top_3_maes, ddof=1) + np.var(other_maes, ddof=1)) / 2)
            cohens_d = (np.mean(top_3_maes) - np.mean(other_maes)) / pooled_std if pooled_std > 0 else 0
            
            significance_results.append({
                'comparison': 'Top 3 vs Others',
                'top_3_mae_mean': np.mean(top_3_maes),
                'top_3_mae_std': np.std(top_3_maes),
                'other_mae_mean': np.mean(other_maes),
                'other_mae_std': np.std(other_maes),
                'mae_difference': np.mean(top_3_maes) - np.mean(other_maes),
                'mannwhitney_statistic': mannwhitney_stat,
                'mannwhitney_p_value': mannwhitney_p,
                'ttest_statistic': ttest_stat,
                'ttest_p_value': ttest_p,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_effect_size(abs(cohens_d)),
                'is_significant_mannwhitney': mannwhitney_p < 0.05,
                'is_significant_ttest': ttest_p < 0.05,
                'top_3_n': len(top_3_maes),
                'other_n': len(other_maes)
            })
            
        except Exception as e:
            print(f"Warning: Statistical test failed: {e}")
        
        # 2. Bootstrap confidence intervals for difference
        try:
            n_bootstrap = 1000
            bootstrap_differences = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample from each group
                top_3_sample = np.random.choice(top_3_maes, size=len(top_3_maes), replace=True)
                other_sample = np.random.choice(other_maes, size=len(other_maes), replace=True)
                
                # Calculate difference in means
                diff = np.mean(top_3_sample) - np.mean(other_sample)
                bootstrap_differences.append(diff)
            
            bootstrap_differences = np.array(bootstrap_differences)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_differences, 2.5)
            ci_upper = np.percentile(bootstrap_differences, 97.5)
            
            # Add bootstrap results
            if significance_results:
                significance_results[0].update({
                    'bootstrap_mean_difference': np.mean(bootstrap_differences),
                    'bootstrap_ci_lower': ci_lower,
                    'bootstrap_ci_upper': ci_upper,
                    'bootstrap_significant': ci_upper < 0  # If upper bound < 0, top 3 is significantly better
                })
            
        except Exception as e:
            print(f"Warning: Bootstrap analysis failed: {e}")
        
        # 3. Individual top 3 vs others comparisons
        for idx, (_, top_combo) in enumerate(top_combinations.iterrows()):
            try:
                combo_data = df_window_arch[df_window_arch['combination'] == top_combo['combination']]
                combo_maes = combo_data['mae'].values
                
                if len(combo_maes) > 0:
                    # Compare this specific combination vs others
                    _, p_value = stats.mannwhitneyu(combo_maes, other_maes, alternative='less')
                    
                    significance_results.append({
                        'comparison': f"#{idx+1} {top_combo['combination']} vs Others",
                        'combo_mae_mean': np.mean(combo_maes),
                        'combo_mae_std': np.std(combo_maes),
                        'other_mae_mean': np.mean(other_maes),
                        'other_mae_std': np.std(other_maes),
                        'mae_difference': np.mean(combo_maes) - np.mean(other_maes),
                        'mannwhitney_p_value': p_value,
                        'is_significant': p_value < 0.05,
                        'combo_n': len(combo_maes),
                        'other_n': len(other_maes)
                    })
            except Exception as e:
                print(f"Warning: Individual comparison failed for {top_combo['combination']}: {e}")
        
        # Save significance results
        if significance_results:
            df_significance = pd.DataFrame(significance_results)
            df_significance.to_csv(output_dir / f'top_combinations_significance_{timestamp}.csv', index=False)
            
            # Print summary
            print(f"📊 Statistical Significance Results:")
            overall_result = significance_results[0]
            print(f"   Top 3 MAE: {overall_result['top_3_mae_mean']:.4f} ± {overall_result['top_3_mae_std']:.4f}")
            print(f"   Others MAE: {overall_result['other_mae_mean']:.4f} ± {overall_result['other_mae_std']:.4f}")
            print(f"   Difference: {overall_result['mae_difference']:.4f}")
            print(f"   Mann-Whitney p-value: {overall_result['mannwhitney_p_value']:.4f}")
            print(f"   T-test p-value: {overall_result['ttest_p_value']:.4f}")
            print(f"   Effect size: {overall_result['effect_size']} (Cohen's d = {overall_result['cohens_d']:.3f})")
            
            if 'bootstrap_significant' in overall_result:
                bootstrap_sig = "Yes" if overall_result['bootstrap_significant'] else "No"
                print(f"   Bootstrap significance: {bootstrap_sig}")
                print(f"   Bootstrap 95% CI: [{overall_result['bootstrap_ci_lower']:.4f}, {overall_result['bootstrap_ci_upper']:.4f}]")
        
        return significance_results if significance_results else None
    
    def _create_window_architecture_plots(self, aggregated_results: dict, top_combinations: pd.DataFrame, 
                                        output_dir: Path, timestamp: str):
        """Create visualizations for window and architecture analysis."""
        
        print("Creating window and architecture visualizations...")
        
        # 1. Window size performance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: MAE by window size
        ax = axes[0, 0]
        window_data = aggregated_results['by_window']
        ax.bar(window_data['window_size'].astype(str), window_data['mean_mae'], 
               yerr=window_data['std_mae'], capsize=5, alpha=0.7)
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Performance by Window Size')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Rank by window size
        ax = axes[0, 1]
        ax.bar(window_data['window_size'].astype(str), window_data['mean_rank'], 
               yerr=window_data['std_rank'], capsize=5, alpha=0.7, color='orange')
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Average Rank')
        ax.set_title('Average Rank by Window Size')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: MAE by architecture
        ax = axes[1, 0]
        arch_data = aggregated_results['by_architecture']
        bars = ax.bar(arch_data['architecture'], arch_data['mean_mae'], 
                      yerr=arch_data['std_mae'], capsize=5, alpha=0.7, color='green')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Performance by Architecture')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Top 3 combinations
        ax = axes[1, 1]
        top_3_labels = [f"#{i+1}: W{row['window_size']} + {row['architecture'][:4]}" 
                       for i, (_, row) in enumerate(top_combinations.iterrows())]
        colors = ['gold', 'silver', '#CD7F32']  # Gold, silver, bronze
        bars = ax.bar(top_3_labels, top_combinations['mean_mae'], 
                      yerr=top_combinations['std_mae'], capsize=5, 
                      color=colors, alpha=0.8)
        ax.set_xlabel('Top 3 Combinations')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Top 3 Window + Architecture Combinations')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on top 3 bars
        for bar, mae in zip(bars, top_combinations['mean_mae']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'window_architecture_analysis_{timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Combination heatmap
        combination_data = aggregated_results['by_combination']
        
        # Create pivot table for heatmap
        pivot_data = combination_data.pivot(index='architecture', columns='window_size', values='mean_mae')
        
        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                       ax=ax, cbar_kws={'label': 'Mean Absolute Error'})
            ax.set_title('Window Size × Architecture Performance Heatmap')
            ax.set_xlabel('Window Size')
            ax.set_ylabel('Architecture')
            
            # Highlight top 3 combinations
            for _, row in top_combinations.iterrows():
                try:
                    arch_idx = pivot_data.index.get_loc(row['architecture'])
                    window_idx = pivot_data.columns.get_loc(row['window_size'])
                    
                    # Add a rectangle around the top combinations
                    rect = plt.Rectangle((window_idx, arch_idx), 1, 1, 
                                       fill=False, edgecolor='red', linewidth=3)
                    ax.add_patch(rect)
                except:
                    pass  # Skip if combination not found in pivot
            
            plt.tight_layout()
            plt.savefig(output_dir / f'combination_heatmap_{timestamp}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print("📊 Window and architecture visualizations created")
    
    def _analyze_feature_consistency(self, df_importance: pd.DataFrame, output_dir: Path, timestamp: str):
        """Analyze feature consistency across models."""
        
        # Calculate feature statistics across models
        feature_stats = df_importance.groupby('feature').agg({
            'importance': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        feature_stats.columns = ['mean_importance', 'std_importance', 'min_importance', 'max_importance', 'model_count']
        
        # Calculate coefficient of variation
        feature_stats['cv'] = feature_stats['std_importance'] / feature_stats['mean_importance']
        
        # Sort by mean importance
        feature_stats = feature_stats.sort_values('mean_importance', ascending=False)
        
        # Save feature consistency analysis
        feature_stats.to_csv(output_dir / f'feature_consistency_analysis_{timestamp}.csv')
        
        print(f"📊 Feature consistency analysis saved")


def main():
    """Main function to run analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze saved predictions from comprehensive evaluation')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing saved prediction results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis results (default: results_dir/analysis)')
    parser.add_argument('--bootstrap-samples', type=int, default=1000,
                        help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals')
    
    args = parser.parse_args()
    
    print("🔍 Starting comprehensive prediction analysis...")
    
    try:
        # Initialize analyzer
        analyzer = PredictionAnalyzer(args.results_dir)
        
        # Run analysis
        results = analyzer.run_comprehensive_analysis(
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {results['output_dir']}")
        
        # Print key findings
        print(f"\n🏆 KEY FINDINGS:")
        performance_summary = results['performance_summary']
        best_model = performance_summary.iloc[0]
        print(f"   Best model: {best_model['model']} (MAE: {best_model['mae_mean']:.3f})")
        
        significance_results = results['significance_results']
        n_significant = len(significance_results[significance_results['is_significant']])
        print(f"   Significant differences: {n_significant}/{len(significance_results)} comparisons")
        
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS:")
        print(f"   - Top 5 features analyzed for each model")
        print(f"   - Key student modeling features highlighted:")
        print(f"     * student_learning_rate")
        print(f"     * student_ability") 
        print(f"     * avg_difficulty")
        print(f"   - Feature consistency and ranking analysis included")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()