"""
Cross-Window Analysis for Proficiency Prediction
===============================================

This script performs comprehensive analysis across multiple window sizes to understand:
1. How model performance changes with sequence length
2. Which features become more/less important with longer windows
3. Model ranking consistency across window sizes
4. Optimal window size recommendations
5. Feature importance evolution patterns

Designed to work with results from multiple separate window evaluations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CrossWindowAnalyzer:
    """Analyzes model performance and feature importance across multiple window sizes."""
    
    def __init__(self, base_dir: str, window_sizes: List[int]):
        """
        Initialize cross-window analyzer.
        
        Args:
            base_dir: Base directory containing window-specific result directories
            window_sizes: List of window sizes to analyze
        """
        self.base_dir = Path(base_dir)
        self.window_sizes = window_sizes
        self.window_results = {}
        self.performance_data = pd.DataFrame()
        self.feature_importance_data = pd.DataFrame()
        
        # Load all window results
        self._load_all_window_results()
        
    def _load_all_window_results(self):
        """Load results from all window directories."""
        print(f"🔍 Loading results from {len(self.window_sizes)} windows...")
        
        for window_size in self.window_sizes:
            window_dir = self.base_dir / f"rolling_new_proficiency_w{window_size}_all_standard_all_except_linear_regression"
            
            if window_dir.exists():
                self.window_results[window_size] = self._load_window_result(window_dir, window_size)
                print(f"✅ Window {window_size}: Loaded")
            else:
                print(f"❌ Window {window_size}: Directory not found - {window_dir}")
        
        # Consolidate performance data
        self._consolidate_performance_data()
        
        # Consolidate feature importance data
        self._consolidate_feature_importance_data()
        
        print(f"📊 Loaded {len(self.window_results)} windows with {len(self.performance_data)} performance records")
    
    def _load_window_result(self, window_dir: Path, window_size: int) -> Dict[str, Any]:
        """Load results from a single window directory."""
        result = {
            'window_size': window_size,
            'performance': {},
            'feature_importance': {},
            'analysis_exists': False
        }
        
        # Load overall results
        overall_results_file = window_dir / 'overall_results.json'
        if overall_results_file.exists():
            with open(overall_results_file, 'r') as f:
                data = json.load(f)
                result['performance'] = data.get('model_results', {})
                result['evaluation_config'] = data.get('evaluation_config', {})
        
        # Load feature importance from analysis directory
        analysis_dir = window_dir / 'analysis'
        if analysis_dir.exists():
            result['analysis_exists'] = True
            
            # Look for feature importance files
            feature_files = list(analysis_dir.glob('top_features_by_model_*.csv'))
            if feature_files:
                # Use the most recent file
                feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
                try:
                    feature_df = pd.read_csv(feature_file)
                    feature_df['window_size'] = window_size
                    result['feature_importance'] = feature_df
                except Exception as e:
                    print(f"⚠️  Warning: Could not load feature importance for window {window_size}: {e}")
        
        return result
    
    def _consolidate_performance_data(self):
        """Consolidate performance data from all windows into a single DataFrame."""
        performance_records = []
        
        for window_size, window_result in self.window_results.items():
            for model_name, model_results in window_result['performance'].items():
                if isinstance(model_results, dict) and 'mae_mean' in model_results:
                    performance_records.append({
                        'window_size': window_size,
                        'model': model_name,
                        'mae_mean': model_results.get('mae_mean', np.nan),
                        'mae_std': model_results.get('mae_std', np.nan),
                        'rmse_mean': model_results.get('rmse_mean', np.nan),
                        'r2_mean': model_results.get('r2_mean', np.nan),
                        'category': model_results.get('category', 'unknown'),
                        'training_time': model_results.get('training_time', np.nan)
                    })
        
        self.performance_data = pd.DataFrame(performance_records)
        
        # Filter out extreme outliers (MAE > 100 likely indicates convergence issues)
        if len(self.performance_data) > 0:
            initial_count = len(self.performance_data)
            self.performance_data = self.performance_data[self.performance_data['mae_mean'] <= 100]
            filtered_count = initial_count - len(self.performance_data)
            
            if filtered_count > 0:
                print(f"⚠️  Filtered {filtered_count} outlier results (MAE > 100)")
    
    def _consolidate_feature_importance_data(self):
        """Consolidate feature importance data from all windows."""
        feature_records = []
        
        for window_size, window_result in self.window_results.items():
            if isinstance(window_result.get('feature_importance'), pd.DataFrame):
                feature_df = window_result['feature_importance'].copy()
                feature_records.append(feature_df)
        
        if feature_records:
            self.feature_importance_data = pd.concat(feature_records, ignore_index=True)
            print(f"📊 Consolidated feature importance: {len(self.feature_importance_data)} records")
        else:
            print("⚠️  No feature importance data found across windows")
    
    def run_cross_window_analysis(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive cross-window analysis."""
        if output_dir is None:
            output_dir = Path("cross_window_analysis")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"CROSS-WINDOW ANALYSIS")
        print(f"{'='*80}")
        
        results = {}
        
        # 1. Window Performance Trends
        print("\n1. Analyzing window performance trends...")
        window_trends = self.analyze_window_performance_trends(output_dir, timestamp)
        results['window_trends'] = window_trends
        
        # 2. Model Ranking Consistency
        print("\n2. Analyzing model ranking consistency...")
        ranking_consistency = self.analyze_model_ranking_consistency(output_dir, timestamp)
        results['ranking_consistency'] = ranking_consistency
        
        # 3. Feature Importance Evolution
        if not self.feature_importance_data.empty:
            print("\n3. Analyzing feature importance evolution...")
            feature_evolution = self.analyze_feature_importance_evolution(output_dir, timestamp)
            results['feature_evolution'] = feature_evolution
        else:
            print("\n3. Skipping feature importance evolution (no data available)")
        
        # 4. Optimal Window Recommendations
        print("\n4. Computing optimal window recommendations...")
        optimal_windows = self.compute_optimal_window_recommendations(output_dir, timestamp)
        results['optimal_windows'] = optimal_windows
        
        # 5. Model Stability Analysis
        print("\n5. Analyzing model stability across windows...")
        stability_analysis = self.analyze_model_stability(output_dir, timestamp)
        results['stability_analysis'] = stability_analysis
        
        # 6. Comprehensive Summary Report
        print("\n6. Creating comprehensive summary report...")
        self.create_summary_report(output_dir, timestamp, results)
        
        print(f"\n✅ Cross-window analysis complete! Results saved to: {output_dir}")
        return results
    
    def analyze_window_performance_trends(self, output_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Analyze how performance changes with window size."""
        
        # Calculate summary statistics by window
        window_summary = self.performance_data.groupby('window_size').agg({
            'mae_mean': ['mean', 'std', 'min', 'max', 'count'],
            'rmse_mean': ['mean', 'std'],
            'r2_mean': ['mean', 'std']
        }).round(3)
        
        window_summary.columns = [f'{col[1]}_{col[0]}' for col in window_summary.columns]
        window_summary = window_summary.reset_index()
        
        # Find best model for each window
        best_models = []
        for window_size in self.window_sizes:
            window_data = self.performance_data[self.performance_data['window_size'] == window_size]
            if len(window_data) > 0:
                best_idx = window_data['mae_mean'].idxmin()
                best_model = window_data.loc[best_idx]
                best_models.append({
                    'window_size': window_size,
                    'best_model': best_model['model'],
                    'best_mae': best_model['mae_mean'],
                    'best_category': best_model['category']
                })
        
        best_models_df = pd.DataFrame(best_models)
        
        # Save results
        window_summary.to_csv(output_dir / f'window_performance_trends_{timestamp}.csv', index=False)
        best_models_df.to_csv(output_dir / f'best_models_by_window_{timestamp}.csv', index=False)
        
        # Create visualizations
        self._create_window_trends_plots(window_summary, best_models_df, output_dir, timestamp)
        
        return {
            'window_summary': window_summary,
            'best_models': best_models_df
        }
    
    def analyze_model_ranking_consistency(self, output_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Analyze how model rankings change across window sizes."""
        
        # Create ranking matrix
        ranking_data = []
        rank_correlations = []
        
        for window_size in self.window_sizes:
            window_data = self.performance_data[self.performance_data['window_size'] == window_size]
            if len(window_data) > 0:
                # Sort by MAE and assign ranks
                sorted_data = window_data.sort_values('mae_mean')
                for rank, (_, row) in enumerate(sorted_data.iterrows(), 1):
                    ranking_data.append({
                        'window_size': window_size,
                        'model': row['model'],
                        'rank': rank,
                        'mae': row['mae_mean'],
                        'category': row['category']
                    })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Calculate rank correlations between consecutive windows
        window_pairs = list(zip(self.window_sizes[:-1], self.window_sizes[1:]))
        
        for w1, w2 in window_pairs:
            ranks_w1 = ranking_df[ranking_df['window_size'] == w1].set_index('model')['rank']
            ranks_w2 = ranking_df[ranking_df['window_size'] == w2].set_index('model')['rank']
            
            # Find common models
            common_models = ranks_w1.index.intersection(ranks_w2.index)
            
            if len(common_models) > 1:
                corr, p_value = spearmanr(ranks_w1[common_models], ranks_w2[common_models])
                rank_correlations.append({
                    'window_pair': f"{w1}-{w2}",
                    'correlation': corr,
                    'p_value': p_value,
                    'n_models': len(common_models)
                })
        
        rank_corr_df = pd.DataFrame(rank_correlations)
        
        # Calculate overall consistency metrics
        model_consistency = []
        for model in ranking_df['model'].unique():
            model_ranks = ranking_df[ranking_df['model'] == model]['rank'].values
            if len(model_ranks) > 1:
                rank_std = np.std(model_ranks)
                rank_range = np.max(model_ranks) - np.min(model_ranks)
                consistency_score = 1 / (1 + rank_std)  # Higher = more consistent
                
                model_consistency.append({
                    'model': model,
                    'rank_std': rank_std,
                    'rank_range': rank_range,
                    'consistency_score': consistency_score,
                    'n_windows': len(model_ranks)
                })
        
        consistency_df = pd.DataFrame(model_consistency)
        consistency_df = consistency_df.sort_values('consistency_score', ascending=False)
        
        # Save results
        ranking_df.to_csv(output_dir / f'model_rankings_by_window_{timestamp}.csv', index=False)
        rank_corr_df.to_csv(output_dir / f'rank_correlations_{timestamp}.csv', index=False)
        consistency_df.to_csv(output_dir / f'model_consistency_{timestamp}.csv', index=False)
        
        # Create visualizations
        self._create_ranking_consistency_plots(ranking_df, rank_corr_df, consistency_df, output_dir, timestamp)
        
        return {
            'rankings': ranking_df,
            'rank_correlations': rank_corr_df,
            'model_consistency': consistency_df
        }
    
    def analyze_feature_importance_evolution(self, output_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Analyze how feature importance changes across window sizes."""
        
        if self.feature_importance_data.empty:
            return {}
        
        # Calculate feature importance trends
        feature_trends = []
        
        for feature in self.feature_importance_data['feature'].unique():
            feature_data = self.feature_importance_data[self.feature_importance_data['feature'] == feature]
            
            # Calculate average importance by window for this feature
            window_importances = feature_data.groupby('window_size')['importance'].mean()
            
            if len(window_importances) > 1:
                # Calculate trend metrics
                windows = window_importances.index.values
                importances = window_importances.values
                
                # Linear trend
                if len(windows) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(windows, importances)
                else:
                    slope = 0
                    r_value = 0
                    p_value = 1
                
                # Stability metrics
                importance_std = np.std(importances)
                importance_range = np.max(importances) - np.min(importances)
                cv = importance_std / np.mean(importances) if np.mean(importances) > 0 else float('inf')
                
                feature_trends.append({
                    'feature': feature,
                    'trend_slope': slope,
                    'trend_r2': r_value**2,
                    'trend_p_value': p_value,
                    'importance_std': importance_std,
                    'importance_range': importance_range,
                    'coefficient_of_variation': cv,
                    'mean_importance': np.mean(importances),
                    'n_windows': len(windows)
                })
        
        feature_trends_df = pd.DataFrame(feature_trends)
        
        if not feature_trends_df.empty:
            # Sort by different criteria
            trending_up = feature_trends_df[feature_trends_df['trend_slope'] > 0].sort_values('trend_slope', ascending=False).head(10)
            trending_down = feature_trends_df[feature_trends_df['trend_slope'] < 0].sort_values('trend_slope', ascending=True).head(10)
            most_stable = feature_trends_df.sort_values('coefficient_of_variation').head(10)
            most_variable = feature_trends_df.sort_values('coefficient_of_variation', ascending=False).head(10)
            
            # Save results
            feature_trends_df.to_csv(output_dir / f'feature_importance_trends_{timestamp}.csv', index=False)
            trending_up.to_csv(output_dir / f'features_trending_up_{timestamp}.csv', index=False)
            trending_down.to_csv(output_dir / f'features_trending_down_{timestamp}.csv', index=False)
            most_stable.to_csv(output_dir / f'most_stable_features_{timestamp}.csv', index=False)
            
            # Create visualizations
            self._create_feature_evolution_plots(feature_trends_df, output_dir, timestamp)
            
            return {
                'feature_trends': feature_trends_df,
                'trending_up': trending_up,
                'trending_down': trending_down,
                'most_stable': most_stable,
                'most_variable': most_variable
            }
        
        return {}
    
    def compute_optimal_window_recommendations(self, output_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Compute optimal window size recommendations based on various criteria."""
        
        recommendations = {}
        
        if self.performance_data.empty:
            return recommendations
        
        # 1. Best overall performance (lowest average MAE across all models)
        window_avg_mae = self.performance_data.groupby('window_size')['mae_mean'].mean()
        best_overall_window = window_avg_mae.idxmin()
        best_overall_mae = window_avg_mae.min()
        
        # 2. Best single model performance
        best_single_idx = self.performance_data['mae_mean'].idxmin()
        best_single_result = self.performance_data.loc[best_single_idx]
        
        # 3. Most consistent performance (lowest std across models)
        window_mae_std = self.performance_data.groupby('window_size')['mae_mean'].std()
        most_consistent_window = window_mae_std.idxmin()
        
        # 4. Best computational efficiency (performance vs time tradeoff)
        if 'training_time' in self.performance_data.columns:
            efficiency_scores = []
            for window_size in self.window_sizes:
                window_data = self.performance_data[self.performance_data['window_size'] == window_size]
                if len(window_data) > 0 and window_data['training_time'].notna().any():
                    avg_mae = window_data['mae_mean'].mean()
                    avg_time = window_data['training_time'].mean()
                    # Efficiency = 1 / (mae * log(time + 1))
                    efficiency = 1 / (avg_mae * np.log(avg_time + 1))
                    efficiency_scores.append({
                        'window_size': window_size,
                        'efficiency': efficiency,
                        'mae': avg_mae,
                        'time': avg_time
                    })
            
            if efficiency_scores:
                efficiency_df = pd.DataFrame(efficiency_scores)
                most_efficient_window = efficiency_df.loc[efficiency_df['efficiency'].idxmax(), 'window_size']
            else:
                most_efficient_window = None
        else:
            most_efficient_window = None
        
        recommendations = {
            'best_overall': {
                'window_size': int(best_overall_window),
                'mae': float(best_overall_mae),
                'reason': 'Lowest average MAE across all models'
            },
            'best_single_model': {
                'window_size': int(best_single_result['window_size']),
                'model': best_single_result['model'],
                'mae': float(best_single_result['mae_mean']),
                'reason': 'Best single model performance'
            },
            'most_consistent': {
                'window_size': int(most_consistent_window),
                'mae_std': float(window_mae_std.min()),
                'reason': 'Most consistent performance across models'
            }
        }
        
        if most_efficient_window is not None:
            recommendations['most_efficient'] = {
                'window_size': int(most_efficient_window),
                'reason': 'Best performance vs computational cost tradeoff'
            }
        
        # Save recommendations
        with open(output_dir / f'optimal_window_recommendations_{timestamp}.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations
    
    def analyze_model_stability(self, output_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Analyze model stability across different window sizes."""
        
        stability_metrics = []
        
        for model in self.performance_data['model'].unique():
            model_data = self.performance_data[self.performance_data['model'] == model]
            
            if len(model_data) > 1:
                mae_values = model_data['mae_mean'].values
                
                # Stability metrics
                mae_mean = np.mean(mae_values)
                mae_std = np.std(mae_values)
                cv = mae_std / mae_mean if mae_mean > 0 else float('inf')
                mae_range = np.max(mae_values) - np.min(mae_values)
                normalized_range = mae_range / mae_mean if mae_mean > 0 else float('inf')
                
                # Trend analysis
                windows = model_data['window_size'].values
                if len(windows) > 1:
                    slope, _, r_value, p_value, _ = stats.linregress(windows, mae_values)
                else:
                    slope = 0
                    r_value = 0
                    p_value = 1
                
                stability_metrics.append({
                    'model': model,
                    'category': model_data['category'].iloc[0],
                    'mae_mean': mae_mean,
                    'mae_std': mae_std,
                    'coefficient_of_variation': cv,
                    'mae_range': mae_range,
                    'normalized_range': normalized_range,
                    'trend_slope': slope,
                    'trend_r2': r_value**2,
                    'trend_p_value': p_value,
                    'n_windows': len(model_data),
                    'stability_score': 1 / (1 + cv) if cv != float('inf') else 0
                })
        
        stability_df = pd.DataFrame(stability_metrics)
        stability_df = stability_df.sort_values('stability_score', ascending=False)
        
        # Save results
        stability_df.to_csv(output_dir / f'model_stability_analysis_{timestamp}.csv', index=False)
        
        # Create visualizations
        self._create_stability_plots(stability_df, output_dir, timestamp)
        
        return {'stability_metrics': stability_df}
    
    def _create_window_trends_plots(self, window_summary: pd.DataFrame, best_models_df: pd.DataFrame, 
                                   output_dir: Path, timestamp: str):
        """Create plots showing window performance trends."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average MAE by window
        ax1 = axes[0, 0]
        ax1.errorbar(window_summary['window_size'], window_summary['mean_mae_mean'], 
                    yerr=window_summary['std_mae_mean'], marker='o', capsize=5)
        ax1.set_xlabel('Window Size')
        ax1.set_ylabel('Average MAE')
        ax1.set_title('Average Performance vs Window Size')
        ax1.grid(True, alpha=0.3)
        
        # 2. Best model MAE by window
        ax2 = axes[0, 1]
        ax2.plot(best_models_df['window_size'], best_models_df['best_mae'], 'ro-')
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('Best Model MAE')
        ax2.set_title('Best Model Performance vs Window Size')
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of successful models by window
        ax3 = axes[1, 0]
        ax3.bar(window_summary['window_size'], window_summary['count_mae_mean'])
        ax3.set_xlabel('Window Size')
        ax3.set_ylabel('Number of Successful Models')
        ax3.set_title('Model Success Rate vs Window Size')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance spread by window
        ax4 = axes[1, 1]
        ax4.errorbar(window_summary['window_size'], window_summary['mean_mae_mean'],
                    yerr=[window_summary['mean_mae_mean'] - window_summary['min_mae_mean'],
                          window_summary['max_mae_mean'] - window_summary['mean_mae_mean']],
                    marker='s', capsize=5, fmt='g-')
        ax4.set_xlabel('Window Size')
        ax4.set_ylabel('MAE Range')
        ax4.set_title('Performance Spread vs Window Size')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'window_performance_trends_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ranking_consistency_plots(self, ranking_df: pd.DataFrame, rank_corr_df: pd.DataFrame,
                                         consistency_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create plots for model ranking consistency analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Ranking heatmap
        ax1 = axes[0, 0]
        pivot_ranks = ranking_df.pivot(index='model', columns='window_size', values='rank')
        sns.heatmap(pivot_ranks, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax1, cbar_kws={'label': 'Rank'})
        ax1.set_title('Model Rankings Across Window Sizes')
        ax1.set_xlabel('Window Size')
        ax1.set_ylabel('Model')
        
        # 2. Rank correlations
        ax2 = axes[0, 1]
        if not rank_corr_df.empty:
            ax2.bar(range(len(rank_corr_df)), rank_corr_df['correlation'])
            ax2.set_xticks(range(len(rank_corr_df)))
            ax2.set_xticklabels(rank_corr_df['window_pair'], rotation=45)
            ax2.set_ylabel('Spearman Correlation')
            ax2.set_title('Ranking Correlation Between Consecutive Windows')
            ax2.grid(True, alpha=0.3)
        
        # 3. Model consistency scores
        ax3 = axes[1, 0]
        if not consistency_df.empty:
            top_consistent = consistency_df.head(10)
            ax3.barh(range(len(top_consistent)), top_consistent['consistency_score'])
            ax3.set_yticks(range(len(top_consistent)))
            ax3.set_yticklabels(top_consistent['model'])
            ax3.set_xlabel('Consistency Score')
            ax3.set_title('Top 10 Most Consistent Models')
            ax3.grid(True, alpha=0.3)
        
        # 4. Rank standard deviation
        ax4 = axes[1, 1]
        if not consistency_df.empty:
            ax4.scatter(consistency_df['rank_std'], consistency_df['consistency_score'])
            ax4.set_xlabel('Rank Standard Deviation')
            ax4.set_ylabel('Consistency Score')
            ax4.set_title('Rank Stability vs Consistency Score')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'ranking_consistency_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_evolution_plots(self, feature_trends_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create plots for feature importance evolution."""
        
        if feature_trends_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature importance trends (top trending up)
        ax1 = axes[0, 0]
        trending_up = feature_trends_df.nlargest(10, 'trend_slope')
        y_pos = np.arange(len(trending_up))
        ax1.barh(y_pos, trending_up['trend_slope'], color='green', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in trending_up['feature']])
        ax1.set_xlabel('Trend Slope (Increasing Importance)')
        ax1.set_title('Top 10 Features with Increasing Importance')
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance trends (top trending down)
        ax2 = axes[0, 1]
        trending_down = feature_trends_df.nsmallest(10, 'trend_slope')
        y_pos = np.arange(len(trending_down))
        ax2.barh(y_pos, trending_down['trend_slope'], color='red', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in trending_down['feature']])
        ax2.set_xlabel('Trend Slope (Decreasing Importance)')
        ax2.set_title('Top 10 Features with Decreasing Importance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature stability
        ax3 = axes[1, 0]
        most_stable = feature_trends_df.nsmallest(15, 'coefficient_of_variation')
        ax3.scatter(most_stable['mean_importance'], most_stable['coefficient_of_variation'])
        ax3.set_xlabel('Mean Importance')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Feature Stability (Lower CV = More Stable)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trend significance
        ax4 = axes[1, 1]
        significant_trends = feature_trends_df[feature_trends_df['trend_p_value'] < 0.05]
        if not significant_trends.empty:
            ax4.scatter(significant_trends['trend_slope'], significant_trends['trend_r2'], 
                       c=significant_trends['mean_importance'], cmap='viridis', s=50)
            ax4.set_xlabel('Trend Slope')
            ax4.set_ylabel('Trend R²')
            ax4.set_title('Significant Trends (p < 0.05)')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(ax4.collections[0], ax=ax4, label='Mean Importance')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_stability_plots(self, stability_df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Create plots for model stability analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Stability scores
        ax1 = axes[0, 0]
        top_stable = stability_df.head(10)
        ax1.barh(range(len(top_stable)), top_stable['stability_score'])
        ax1.set_yticks(range(len(top_stable)))
        ax1.set_yticklabels(top_stable['model'])
        ax1.set_xlabel('Stability Score')
        ax1.set_title('Top 10 Most Stable Models')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance vs stability
        ax2 = axes[0, 1]
        ax2.scatter(stability_df['mae_mean'], stability_df['stability_score'], 
                   c=stability_df.index, cmap='viridis', s=50)
        ax2.set_xlabel('Mean MAE')
        ax2.set_ylabel('Stability Score')
        ax2.set_title('Performance vs Stability')
        ax2.grid(True, alpha=0.3)
        
        # 3. Coefficient of variation by category
        ax3 = axes[1, 0]
        category_cv = stability_df.groupby('category')['coefficient_of_variation'].mean().sort_values()
        ax3.bar(range(len(category_cv)), category_cv.values)
        ax3.set_xticks(range(len(category_cv)))
        ax3.set_xticklabels(category_cv.index, rotation=45)
        ax3.set_ylabel('Average Coefficient of Variation')
        ax3.set_title('Stability by Model Category')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trend analysis
        ax4 = axes[1, 1]
        significant_trends = stability_df[stability_df['trend_p_value'] < 0.05]
        if not significant_trends.empty:
            colors = ['red' if slope > 0 else 'blue' for slope in significant_trends['trend_slope']]
            ax4.scatter(significant_trends['trend_slope'], significant_trends['trend_r2'], c=colors, s=50)
            ax4.set_xlabel('Performance Trend Slope')
            ax4.set_ylabel('Trend R²')
            ax4.set_title('Performance Trends Across Windows\n(Red=Degrading, Blue=Improving)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'model_stability_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, output_dir: Path, timestamp: str, results: Dict[str, Any]):
        """Create comprehensive summary report."""
        
        report_path = output_dir / f'cross_window_analysis_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Cross-Window Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write(f"## Analysis Configuration\n\n")
            f.write(f"- **Window sizes analyzed**: {self.window_sizes}\n")
            f.write(f"- **Total performance records**: {len(self.performance_data)}\n")
            f.write(f"- **Feature importance records**: {len(self.feature_importance_data)}\n")
            f.write(f"- **Base directory**: {self.base_dir}\n\n")
            
            # Window performance trends
            if 'window_trends' in results:
                window_trends = results['window_trends']
                best_models = window_trends['best_models']
                
                f.write(f"## Window Performance Trends\n\n")
                f.write(f"### Best Models by Window Size\n\n")
                for _, row in best_models.iterrows():
                    f.write(f"- **Window {row['window_size']}**: {row['best_model']} ({row['best_category']}) - MAE: {row['best_mae']:.3f}\n")
                f.write(f"\n")
            
            # Optimal window recommendations
            if 'optimal_windows' in results:
                recommendations = results['optimal_windows']
                f.write(f"## Optimal Window Recommendations\n\n")
                
                for criteria, rec in recommendations.items():
                    f.write(f"### {criteria.replace('_', ' ').title()}\n")
                    f.write(f"- **Window size**: {rec['window_size']}\n")
                    f.write(f"- **Reason**: {rec['reason']}\n")
                    if 'mae' in rec:
                        f.write(f"- **MAE**: {rec['mae']:.3f}\n")
                    f.write(f"\n")
            
            # Model consistency insights
            if 'ranking_consistency' in results:
                consistency = results['ranking_consistency']['model_consistency']
                if not consistency.empty:
                    f.write(f"## Model Consistency Insights\n\n")
                    f.write(f"### Top 5 Most Consistent Models\n\n")
                    top_5 = consistency.head(5)
                    for _, row in top_5.iterrows():
                        f.write(f"- **{row['model']}**: Consistency score {row['consistency_score']:.3f} (rank std: {row['rank_std']:.2f})\n")
                    f.write(f"\n")
            
            # Feature evolution insights
            if 'feature_evolution' in results and results['feature_evolution']:
                feature_evolution = results['feature_evolution']
                f.write(f"## Feature Importance Evolution\n\n")
                
                if 'trending_up' in feature_evolution and not feature_evolution['trending_up'].empty:
                    f.write(f"### Features with Increasing Importance\n\n")
                    trending_up = feature_evolution['trending_up'].head(5)
                    for _, row in trending_up.iterrows():
                        f.write(f"- **{row['feature']}**: Slope {row['trend_slope']:.4f}, R² {row['trend_r2']:.3f}\n")
                    f.write(f"\n")
                
                if 'trending_down' in feature_evolution and not feature_evolution['trending_down'].empty:
                    f.write(f"### Features with Decreasing Importance\n\n")
                    trending_down = feature_evolution['trending_down'].head(5)
                    for _, row in trending_down.iterrows():
                        f.write(f"- **{row['feature']}**: Slope {row['trend_slope']:.4f}, R² {row['trend_r2']:.3f}\n")
                    f.write(f"\n")
            
            # Files generated
            f.write(f"## Generated Files\n\n")
            f.write(f"### Performance Analysis\n")
            f.write(f"- `window_performance_trends_{timestamp}.csv`: Performance trends across windows\n")
            f.write(f"- `best_models_by_window_{timestamp}.csv`: Best performing models for each window\n")
            f.write(f"- `optimal_window_recommendations_{timestamp}.json`: Recommended window sizes\n")
            f.write(f"- `window_performance_trends_{timestamp}.png`: Performance visualization\n\n")
            
            f.write(f"### Model Analysis\n")
            f.write(f"- `model_rankings_by_window_{timestamp}.csv`: Model rankings for each window\n")
            f.write(f"- `rank_correlations_{timestamp}.csv`: Ranking correlations between windows\n")
            f.write(f"- `model_consistency_{timestamp}.csv`: Model consistency metrics\n")
            f.write(f"- `model_stability_analysis_{timestamp}.csv`: Stability analysis across windows\n")
            f.write(f"- `ranking_consistency_analysis_{timestamp}.png`: Ranking consistency visualization\n")
            f.write(f"- `model_stability_analysis_{timestamp}.png`: Stability analysis visualization\n\n")
            
            if not self.feature_importance_data.empty:
                f.write(f"### Feature Importance Analysis\n")
                f.write(f"- `feature_importance_trends_{timestamp}.csv`: Feature importance trends\n")
                f.write(f"- `features_trending_up_{timestamp}.csv`: Features with increasing importance\n")
                f.write(f"- `features_trending_down_{timestamp}.csv`: Features with decreasing importance\n")
                f.write(f"- `most_stable_features_{timestamp}.csv`: Most stable features across windows\n")
                f.write(f"- `feature_importance_evolution_{timestamp}.png`: Feature evolution visualization\n\n")
        
        print(f"📋 Summary report saved to: {report_path}")


def main():
    """Run cross-window analysis on proficiency prediction results."""
    
    # Configuration
    base_dir = "evaluation_outputs_after_milestone_2_single_config_with_features"
    window_sizes = [1, 6, 11, 16, 21, 26]
    output_dir = "cross_window_analysis_results"
    
    print("🔍 CROSS-WINDOW ANALYSIS FOR PROFICIENCY PREDICTION")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Window sizes: {window_sizes}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create analyzer and run analysis
    analyzer = CrossWindowAnalyzer(base_dir, window_sizes)
    results = analyzer.run_cross_window_analysis(output_dir)
    
    print(f"\n🎉 Cross-window analysis complete!")
    print(f"Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main() 