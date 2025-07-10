"""
Quick Prediction Error Analysis
Focuses on key insights about model errors across target ranges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our framework
import sys
sys.path.append('../../..')
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
import torch
from torch.utils.data import DataLoader

# Models to analyze
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb


def get_model_predictions(model_name, model, dataset, schema):
    """Get cross-validated predictions for a model."""
    print(f"\n{model_name}:")
    
    # Create adapter
    adapter = SKLearnAdapter(
        sklearn_model=model,
        schema=schema,
        lag_window=5
    )
    
    # Get splits
    splits = dataset.get_splits(n_splits=5, test_size=1)
    
    all_predictions = []
    all_actuals = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/5", end='', flush=True)
        
        # Get data loaders
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
        
        # Train and predict
        adapter.fit(train_loader)
        y_pred = adapter.predict(val_loader)
        
        # Get actuals
        y_true = []
        for _, batch_y in val_loader:
            y_true.extend(batch_y.numpy().flatten())
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_true)
    
    print(" Done!")
    
    return np.array(all_predictions), np.array(all_actuals)


def analyze_errors_by_range(predictions, actuals, model_name):
    """Analyze prediction errors by target value ranges."""
    
    errors = predictions - actuals
    
    # Define meaningful bins
    bins = [0, 5, 15, 30, 50, float('inf')]
    labels = ['Very Low\n[0-5)', 'Low\n[5-15)', 'Medium\n[15-30)', 
              'High\n[30-50)', 'Very High\n[50+)']
    
    # Bin the data
    binned = pd.cut(actuals, bins=bins, labels=labels, include_lowest=True)
    
    # Calculate metrics for each bin
    print(f"\n{model_name} - Error Analysis by Target Range:")
    print("-" * 70)
    print(f"{'Range':<15} {'Count':>8} {'MAE':>8} {'Bias':>8} {'Within±5':>10}")
    print("-" * 70)
    
    metrics = []
    for label in labels:
        mask = binned == label
        if mask.sum() > 0:
            bin_errors = errors[mask]
            mae = np.mean(np.abs(bin_errors))
            bias = np.mean(bin_errors)
            within_5 = np.mean(np.abs(bin_errors) <= 5) * 100
            
            print(f"{label:<15} {mask.sum():>8} {mae:>8.2f} {bias:>8.2f} {within_5:>9.1f}%")
            
            metrics.append({
                'range': label,
                'mae': mae,
                'bias': bias,
                'count': mask.sum()
            })
    
    return metrics


def create_key_visualizations(all_results, save_dir='results'):
    """Create the most important visualizations."""
    
    # 1. MAE by Target Range Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot MAE by range for each model
    ax = axes[0, 0]
    for result in all_results:
        ranges = [m['range'] for m in result['metrics']]
        maes = [m['mae'] for m in result['metrics']]
        ax.plot(ranges, maes, 'o-', label=result['model'], markersize=8, linewidth=2)
    
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('MAE by Target Value Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Bias by range
    ax = axes[0, 1]
    x = np.arange(len(ranges))
    width = 0.2
    
    for i, result in enumerate(all_results):
        biases = [m['bias'] for m in result['metrics']]
        ax.bar(x + i*width, biases, width, label=result['model'], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean Error (Bias)')
    ax.set_title('Prediction Bias by Target Range')
    ax.set_xticks(x + width)
    ax.set_xticklabels(ranges, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot - Best model
    best_result = min(all_results, key=lambda x: x['overall_mae'])
    ax = axes[1, 0]
    
    # Subsample for clarity
    idx = np.random.choice(len(best_result['actuals']), 
                          min(1000, len(best_result['actuals'])), 
                          replace=False)
    
    ax.scatter(best_result['actuals'][idx], best_result['predictions'][idx], 
              alpha=0.5, s=20)
    ax.plot([0, best_result['actuals'].max()], 
            [0, best_result['actuals'].max()], 
            'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Actual Minutes per Week')
    ax.set_ylabel('Predicted Minutes per Week')
    ax.set_title(f'Best Model: {best_result["model"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Overall comparison
    ax = axes[1, 1]
    model_names = [r['model'] for r in all_results]
    overall_maes = [r['overall_mae'] for r in all_results]
    
    bars = ax.bar(model_names, overall_maes, color='steelblue', alpha=0.8)
    ax.set_ylabel('Overall MAE')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mae in zip(bars, overall_maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Prediction Error Analysis Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run quick error analysis."""
    
    # Create output directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("Loading data...")
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../../../../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Define models
    models = {
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
    }
    
    # Analyze each model
    all_results = []
    for model_name, model in models.items():
        predictions, actuals = get_model_predictions(model_name, model, dataset, schema)
        metrics = analyze_errors_by_range(predictions, actuals, model_name)
        
        overall_mae = mean_absolute_error(actuals, predictions)
        print(f"\nOverall MAE: {overall_mae:.3f}")
        
        all_results.append({
            'model': model_name,
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics,
            'overall_mae': overall_mae
        })
    
    # Create visualizations
    create_key_visualizations(all_results)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
