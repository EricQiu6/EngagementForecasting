"""
Comprehensive Prediction Error Analysis
Analyzes model prediction errors across different target value ranges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Import our framework
import sys
sys.path.append('../../..')
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator
import torch
from torch.utils.data import DataLoader

# Models to analyze
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
import xgboost as xgb


class PredictionErrorAnalyzer:
    """Analyzes prediction errors across different target ranges."""
    
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.results = {}
        
    def get_predictions(self, model_name, model, dataset, schema):
        """Get predictions from a model using cross-validation."""
        print(f"\nGetting predictions for {model_name}...")
        
        # Create adapter
        adapter = SKLearnAdapter(
            sklearn_model=model,
            schema=schema,
            lag_window=5
        )
        
        # Use cross-validation to get out-of-fold predictions
        cv = CrossValidator(adapter, dataset)
        
        # Store predictions from each fold
        all_predictions = []
        all_actuals = []
        all_errors = []
        
        # Get splits
        splits = dataset.get_splits(n_splits=5, test_size=1)
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"  Fold {fold_idx + 1}/5", end='', flush=True)
            
            # Get fold data
            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, train_indices),
                batch_size=32,
                shuffle=False
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, val_indices),
                batch_size=32,
                shuffle=False
            )
            
            # Train model
            adapter.fit(train_loader)
            
            # Get predictions
            y_pred = adapter.predict(val_loader)
            
            # Get actuals
            y_true = []
            for _, batch_y in val_loader:
                y_true.extend(batch_y.numpy().flatten())
            y_true = np.array(y_true)
            
            # Store results
            all_predictions.extend(y_pred)
            all_actuals.extend(y_true)
            all_errors.extend(y_pred - y_true)
            
        print(" Done!")
        
        return {
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals),
            'errors': np.array(all_errors),
            'model_name': model_name
        }
    
    def analyze_by_target_range(self, results):
        """Analyze errors by target value ranges."""
        print("\n=== RESIDUAL ANALYSIS BY TARGET RANGE ===")
        
        actuals = results['actuals']
        predictions = results['predictions']
        errors = results['errors']
        
        # Define bins
        bins = [0, 5, 15, 30, 50, float('inf')]
        labels = ['Very Low\n[0-5)', 'Low\n[5-15)', 'Medium\n[15-30)', 
                  'High\n[30-50)', 'Very High\n[50+)']
        
        # Bin the actual values
        binned = pd.cut(actuals, bins=bins, labels=labels, include_lowest=True)
        
        # Calculate metrics for each bin
        bin_metrics = []
        for label in labels:
            mask = binned == label
            if mask.sum() > 0:
                bin_errors = errors[mask]
                bin_actuals = actuals[mask]
                bin_predictions = predictions[mask]
                
                metrics = {
                    'bin': label,
                    'count': mask.sum(),
                    'mae': np.mean(np.abs(bin_errors)),
                    'rmse': np.sqrt(np.mean(bin_errors**2)),
                    'bias': np.mean(bin_errors),
                    'std_error': np.std(bin_errors),
                    'mean_actual': np.mean(bin_actuals),
                    'mean_predicted': np.mean(bin_predictions),
                    'within_5': np.mean(np.abs(bin_errors) <= 5) * 100
                }
                bin_metrics.append(metrics)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MAE by bin
        ax = axes[0, 0]
        bins_names = [m['bin'] for m in bin_metrics]
        maes = [m['mae'] for m in bin_metrics]
        bars = ax.bar(bins_names, maes, color='steelblue', alpha=0.8)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'{results["model_name"]}: MAE by Target Range')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.2f}', ha='center', va='bottom')
        
        # 2. Bias by bin
        ax = axes[0, 1]
        biases = [m['bias'] for m in bin_metrics]
        colors = ['red' if b < 0 else 'green' for b in biases]
        bars = ax.bar(bins_names, biases, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Mean Error (Bias)')
        ax.set_title('Prediction Bias by Target Range')
        ax.grid(True, alpha=0.3)
        
        # 3. Sample size by bin
        ax = axes[1, 0]
        counts = [m['count'] for m in bin_metrics]
        ax.bar(bins_names, counts, color='coral', alpha=0.8)
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution by Target Range')
        ax.grid(True, alpha=0.3)
        
        # 4. Percentage within ±5 minutes
        ax = axes[1, 1]
        within_5 = [m['within_5'] for m in bin_metrics]
        ax.bar(bins_names, within_5, color='darkgreen', alpha=0.8)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Predictions Within ±5 Minutes of Actual')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.suptitle(f'Error Analysis by Target Range - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/error_by_range_{results["model_name"].lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        
        # Print summary
        print(f"\nMetrics for {results['model_name']}:")
        print("-" * 80)
        print(f"{'Range':<15} {'Count':>8} {'MAE':>8} {'Bias':>8} {'Std':>8} {'Within±5':>10}")
        print("-" * 80)
        for m in bin_metrics:
            print(f"{m['bin']:<15} {m['count']:>8} {m['mae']:>8.2f} {m['bias']:>8.2f} "
                  f"{m['std_error']:>8.2f} {m['within_5']:>9.1f}%")
        
        return bin_metrics
    
    def create_scatter_plots(self, results):
        """Create predicted vs actual scatter plots with analysis."""
        print("\n=== SCATTER PLOT ANALYSIS ===")
        
        actuals = results['actuals']
        predictions = results['predictions']
        errors = results['errors']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Basic scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(actuals, predictions, alpha=0.5, s=20, 
                           c=np.abs(errors), cmap='YlOrRd')
        ax.plot([0, actuals.max()], [0, actuals.max()], 'k--', alpha=0.5, label='Perfect prediction')
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Predicted Minutes per Week')
        ax.set_title('Predicted vs Actual')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Absolute Error')
        
        # 2. Residual plot
        ax = axes[0, 1]
        ax.scatter(actuals, errors, alpha=0.5, s=20)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add LOESS smoothing
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(errors, actuals, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'red', linewidth=2, label='LOESS smooth')
        
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Prediction Error (Predicted - Actual)')
        ax.set_title('Residual Plot with Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax = axes[1, 0]
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)
        
        # 4. Error distribution by actual value bins
        ax = axes[1, 1]
        bins = [0, 10, 20, 30, 50, 100]
        for i in range(len(bins)-1):
            mask = (actuals >= bins[i]) & (actuals < bins[i+1])
            if mask.sum() > 0:
                ax.hist(errors[mask], bins=20, alpha=0.5, 
                       label=f'{bins[i]}-{bins[i+1]} min', density=True)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution by Actual Value Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Scatter Plot Analysis - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/scatter_analysis_{results["model_name"].lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        
        # Calculate heteroscedasticity test
        # Breusch-Pagan test
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.regression.linear_model import OLS
        
        # Prepare data for test
        X = actuals.reshape(-1, 1)
        y = errors
        model = OLS(y, X).fit()
        bp_test = het_breuschpagan(model.resid, X)
        
        print(f"\nHeteroscedasticity Test for {results['model_name']}:")
        print(f"Breusch-Pagan test statistic: {bp_test[0]:.3f}")
        print(f"p-value: {bp_test[1]:.3f}")
        print(f"Heteroscedasticity: {'Present' if bp_test[1] < 0.05 else 'Not significant'}")
    
    def analyze_extreme_values(self, results):
        """Analyze performance on extreme values."""
        print("\n=== EXTREME VALUE ANALYSIS ===")
        
        actuals = results['actuals']
        predictions = results['predictions']
        errors = results['errors']
        
        # Define extreme thresholds
        p10 = np.percentile(actuals, 10)
        p90 = np.percentile(actuals, 90)
        
        # Create masks
        bottom_10_mask = actuals <= p10
        top_10_mask = actuals >= p90
        middle_80_mask = ~(bottom_10_mask | top_10_mask)
        
        # Calculate metrics for each group
        groups = {
            'Bottom 10%': bottom_10_mask,
            'Middle 80%': middle_80_mask,
            'Top 10%': top_10_mask
        }
        
        metrics = []
        for name, mask in groups.items():
            if mask.sum() > 0:
                group_metrics = {
                    'group': name,
                    'count': mask.sum(),
                    'actual_mean': actuals[mask].mean(),
                    'actual_range': f"{actuals[mask].min():.1f}-{actuals[mask].max():.1f}",
                    'mae': mean_absolute_error(actuals[mask], predictions[mask]),
                    'rmse': np.sqrt(mean_squared_error(actuals[mask], predictions[mask])),
                    'bias': errors[mask].mean(),
                    'r2': r2_score(actuals[mask], predictions[mask])
                }
                metrics.append(group_metrics)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Box plots of errors by group
        ax = axes[0, 0]
        error_data = [errors[bottom_10_mask], errors[middle_80_mask], errors[top_10_mask]]
        ax.boxplot(error_data, labels=['Bottom 10%', 'Middle 80%', 'Top 10%'])
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution by Percentile Group')
        ax.grid(True, alpha=0.3)
        
        # 2. MAE comparison
        ax = axes[0, 1]
        groups_names = [m['group'] for m in metrics]
        maes = [m['mae'] for m in metrics]
        ax.bar(groups_names, maes, color=['red', 'green', 'blue'], alpha=0.8)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE by Percentile Group')
        ax.grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted for extremes
        ax = axes[1, 0]
        # Bottom 10%
        ax.scatter(actuals[bottom_10_mask], predictions[bottom_10_mask], 
                  alpha=0.6, label='Bottom 10%', color='red', s=30)
        # Top 10%
        ax.scatter(actuals[top_10_mask], predictions[top_10_mask], 
                  alpha=0.6, label='Top 10%', color='blue', s=30)
        ax.plot([0, actuals.max()], [0, actuals.max()], 'k--', alpha=0.5)
        ax.set_xlabel('Actual Minutes')
        ax.set_ylabel('Predicted Minutes')
        ax.set_title('Predictions for Extreme Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Bias by group
        ax = axes[1, 1]
        biases = [m['bias'] for m in metrics]
        colors = ['red' if b < 0 else 'green' for b in biases]
        ax.bar(groups_names, biases, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Mean Error (Bias)')
        ax.set_title('Systematic Bias by Percentile Group')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Extreme Value Analysis - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/extreme_analysis_{results["model_name"].lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        
        # Print summary
        print(f"\nExtreme Value Metrics for {results['model_name']}:")
        print("-" * 90)
        print(f"{'Group':<12} {'Count':>8} {'Range':<15} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'R²':>8}")
        print("-" * 90)
        for m in metrics:
            print(f"{m['group']:<12} {m['count']:>8} {m['actual_range']:<15} "
                  f"{m['mae']:>8.2f} {m['rmse']:>8.2f} {m['bias']:>8.2f} {m['r2']:>8.3f}")
    
    def analyze_zero_inflation(self, results):
        """Analyze model performance on zero and near-zero values."""
        print("\n=== ZERO-INFLATION ANALYSIS ===")
        
        actuals = results['actuals']
        predictions = results['predictions']
        
        # Define thresholds
        zero_mask = actuals == 0
        near_zero_mask = (actuals > 0) & (actuals < 5)
        low_mask = (actuals >= 5) & (actuals < 15)
        normal_mask = actuals >= 15
        
        # Calculate metrics
        groups = {
            'Zero (0)': zero_mask,
            'Near-zero (0-5)': near_zero_mask,
            'Low (5-15)': low_mask,
            'Normal (15+)': normal_mask
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion matrix for zero detection
        ax = axes[0, 0]
        zero_threshold = 2.5  # Consider predictions < 2.5 as "zero"
        
        true_zero = actuals == 0
        pred_zero = predictions < zero_threshold
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_zero, pred_zero)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Non-zero', 'Predicted Zero'],
                    yticklabels=['Actual Non-zero', 'Actual Zero'])
        ax.set_title('Zero Detection Confusion Matrix')
        
        # 2. Distribution of predictions for actual zeros
        ax = axes[0, 1]
        if zero_mask.sum() > 0:
            ax.hist(predictions[zero_mask], bins=20, alpha=0.7, color='red', 
                   label=f'Predictions for actual zeros (n={zero_mask.sum()})')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Count')
            ax.set_title('Predictions When Actual = 0')
            ax.legend()
        
        # 3. Actual vs Predicted for low values
        ax = axes[1, 0]
        low_value_mask = actuals < 15
        ax.scatter(actuals[low_value_mask], predictions[low_value_mask], alpha=0.6)
        ax.plot([0, 15], [0, 15], 'k--', alpha=0.5, label='Perfect prediction')
        ax.set_xlabel('Actual Minutes')
        ax.set_ylabel('Predicted Minutes')
        ax.set_title('Predictions for Low Values (< 15 minutes)')
        ax.set_xlim(-1, 16)
        ax.set_ylim(-1, 20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error metrics by group
        ax = axes[1, 1]
        group_metrics = []
        for name, mask in groups.items():
            if mask.sum() > 0:
                mae = mean_absolute_error(actuals[mask], predictions[mask])
                group_metrics.append({'name': name, 'mae': mae, 'count': mask.sum()})
        
        names = [m['name'] for m in group_metrics]
        maes = [m['mae'] for m in group_metrics]
        counts = [m['count'] for m in group_metrics]
        
        bars = ax.bar(names, maes, alpha=0.8, color='steelblue')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE by Engagement Level')
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'Zero-Inflation Analysis - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/zero_inflation_{results["model_name"].lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        
        # Print metrics
        print(f"\nZero-Inflation Metrics for {results['model_name']}:")
        if zero_mask.sum() > 0:
            print(f"Actual zeros: {zero_mask.sum()}")
            print(f"Mean prediction for zeros: {predictions[zero_mask].mean():.2f}")
            print(f"Predictions < 2.5 for actual zeros: {(predictions[zero_mask] < 2.5).sum()}")
    
    def analyze_bias_variance(self, results):
        """Decompose error into bias and variance components."""
        print("\n=== BIAS-VARIANCE ANALYSIS ===")
        
        actuals = results['actuals']
        predictions = results['predictions']
        errors = results['errors']
        
        # Define bins for analysis
        bins = np.linspace(0, 50, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bias_by_bin = []
        variance_by_bin = []
        mse_by_bin = []
        count_by_bin = []
        
        for i in range(len(bins)-1):
            mask = (actuals >= bins[i]) & (actuals < bins[i+1])
            if mask.sum() > 5:  # Need enough samples
                bin_errors = errors[mask]
                bias = np.mean(bin_errors)
                variance = np.var(bin_errors)
                mse = np.mean(bin_errors**2)
                
                bias_by_bin.append(bias)
                variance_by_bin.append(variance)
                mse_by_bin.append(mse)
                count_by_bin.append(mask.sum())
            else:
                bias_by_bin.append(np.nan)
                variance_by_bin.append(np.nan)
                mse_by_bin.append(np.nan)
                count_by_bin.append(0)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Bias curve
        ax = axes[0, 0]
        valid_mask = ~np.isnan(bias_by_bin)
        ax.plot(bin_centers[valid_mask], np.array(bias_by_bin)[valid_mask], 
               'o-', color='red', linewidth=2, markersize=8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Mean Error (Bias)')
        ax.set_title('Bias vs Target Value')
        ax.grid(True, alpha=0.3)
        
        # 2. Variance curve
        ax = axes[0, 1]
        ax.plot(bin_centers[valid_mask], np.array(variance_by_bin)[valid_mask], 
               'o-', color='blue', linewidth=2, markersize=8)
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Error Variance')
        ax.set_title('Variance vs Target Value')
        ax.grid(True, alpha=0.3)
        
        # 3. Bias² vs Variance
        ax = axes[1, 0]
        bias_squared = np.array(bias_by_bin)**2
        ax.bar(bin_centers[valid_mask] - 1, bias_squared[valid_mask], 
               width=2, alpha=0.7, label='Bias²', color='red')
        ax.bar(bin_centers[valid_mask] + 1, np.array(variance_by_bin)[valid_mask], 
               width=2, alpha=0.7, label='Variance', color='blue')
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Error Component')
        ax.set_title('Bias² and Variance Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Sample size by bin
        ax = axes[1, 1]
        ax.bar(bin_centers, count_by_bin, width=bins[1]-bins[0]*0.8, 
               alpha=0.7, color='green')
        ax.set_xlabel('Actual Minutes per Week')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Bias-Variance Analysis - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/bias_variance_{results["model_name"].lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        
        # Calculate overall bias-variance decomposition
        overall_bias = np.mean(errors)
        overall_variance = np.var(errors)
        overall_mse = np.mean(errors**2)
        
        print(f"\nBias-Variance Decomposition for {results['model_name']}:")
        print(f"Overall Bias: {overall_bias:.3f}")
        print(f"Overall Bias²: {overall_bias**2:.3f}")
        print(f"Overall Variance: {overall_variance:.3f}")
        print(f"Overall MSE: {overall_mse:.3f}")
        print(f"Bias² / MSE: {(overall_bias**2 / overall_mse * 100):.1f}%")
        print(f"Variance / MSE: {(overall_variance / overall_mse * 100):.1f}%")
    
    def create_summary_report(self, all_results):
        """Create a comprehensive summary report comparing all models."""
        print("\n=== CREATING SUMMARY REPORT ===")
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prepare data for all models
        model_names = []
        overall_maes = []
        low_maes = []
        high_maes = []
        zero_maes = []
        biases = []
        
        for result in all_results:
            model_name = result['model_name']
            actuals = result['actuals']
            predictions = result['predictions']
            errors = result['errors']
            
            model_names.append(model_name)
            overall_maes.append(mean_absolute_error(actuals, predictions))
            
            # Low values (< 15)
            low_mask = actuals < 15
            if low_mask.sum() > 0:
                low_maes.append(mean_absolute_error(actuals[low_mask], predictions[low_mask]))
            else:
                low_maes.append(np.nan)
            
            # High values (> 30)
            high_mask = actuals > 30
            if high_mask.sum() > 0:
                high_maes.append(mean_absolute_error(actuals[high_mask], predictions[high_mask]))
            else:
                high_maes.append(np.nan)
            
            # Zero values
            zero_mask = actuals == 0
            if zero_mask.sum() > 0:
                zero_maes.append(mean_absolute_error(actuals[zero_mask], predictions[zero_mask]))
            else:
                zero_maes.append(np.nan)
            
            biases.append(np.mean(errors))
        
        # 1. Overall MAE comparison
        ax = axes[0, 0]
        ax.bar(model_names, overall_maes, color='steelblue', alpha=0.8)
        ax.set_ylabel('MAE')
        ax.set_title('Overall Model Performance')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. MAE by range
        ax = axes[0, 1]
        x = np.arange(len(model_names))
        width = 0.25
        ax.bar(x - width, low_maes, width, label='Low (<15)', alpha=0.8)
        ax.bar(x, overall_maes, width, label='Overall', alpha=0.8)
        ax.bar(x + width, high_maes, width, label='High (>30)', alpha=0.8)
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Value Range')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Bias comparison
        ax = axes[0, 2]
        colors = ['red' if b < 0 else 'green' for b in biases]
        ax.bar(model_names, biases, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Mean Error (Bias)')
        ax.set_title('Model Bias Comparison')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4-6. Scatter plots for each model
        for i, result in enumerate(all_results[:3]):  # Show first 3 models
            ax = axes[1, i]
            actuals = result['actuals']
            predictions = result['predictions']
            
            # Subsample for clarity
            if len(actuals) > 1000:
                idx = np.random.choice(len(actuals), 1000, replace=False)
                actuals_plot = actuals[idx]
                predictions_plot = predictions[idx]
            else:
                actuals_plot = actuals
                predictions_plot = predictions
            
            ax.scatter(actuals_plot, predictions_plot, alpha=0.5, s=20)
            ax.plot([0, actuals.max()], [0, actuals.max()], 'k--', alpha=0.5)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{result["model_name"]}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Comparison Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/model_comparison_summary.png', 
                    dpi=150, bbox_inches='tight')
        
        # Create detailed summary table
        print("\n" + "=" * 100)
        print("COMPREHENSIVE ERROR ANALYSIS SUMMARY")
        print("=" * 100)
        print(f"{'Model':<20} {'Overall MAE':>12} {'Low MAE':>12} {'High MAE':>12} "
              f"{'Zero MAE':>12} {'Bias':>10}")
        print("-" * 100)
        
        for i in range(len(model_names)):
            print(f"{model_names[i]:<20} {overall_maes[i]:>12.2f} "
                  f"{low_maes[i]:>12.2f} {high_maes[i]:>12.2f} "
                  f"{zero_maes[i]:>12.2f} {biases[i]:>10.2f}")


def main():
    """Run the complete prediction error analysis."""
    
    # Create output directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Initialize analyzer
    analyzer = PredictionErrorAnalyzer(save_dir='results')
    
    # Load data
    print("Loading data...")
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../../../../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Define models to analyze
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
    }
    
    # Get predictions for each model
    all_results = []
    for model_name, model in models.items():
        results = analyzer.get_predictions(model_name, model, dataset, schema)
        all_results.append(results)
        
        # Run all analyses
        analyzer.analyze_by_target_range(results)
        analyzer.create_scatter_plots(results)
        analyzer.analyze_extreme_values(results)
        analyzer.analyze_zero_inflation(results)
        analyzer.analyze_bias_variance(results)
    
    # Create summary report
    analyzer.create_summary_report(all_results)
    
    print("\n✅ Analysis complete! Results saved in 'results' directory.")


if __name__ == "__main__":
    main()
