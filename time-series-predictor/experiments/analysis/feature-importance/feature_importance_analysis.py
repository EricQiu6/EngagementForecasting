"""
Feature Importance Analysis for Time Series Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters.sklearn_adapter import SKLearnAdapter
from src.framework.core.base import CrossValidator
from torch.utils.data import DataLoader


def get_feature_names_from_adapter(adapter):
    """Get feature names from the adapter."""
    return adapter.get_feature_names()


def analyze_tree_based_importance():
    """Analyze feature importance for tree-based models."""
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS - TREE-BASED MODELS")
    print("=" * 80)
    
    # Setup
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Models that have built-in feature importance
    tree_models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        ),
        'extra_trees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Get all data for training
    all_data = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Create adapter to get feature names and transform data
    sample_adapter = SKLearnAdapter(
        sklearn_model=tree_models['random_forest'],
        schema=schema,
        lag_window=5
    )
    
    # Transform data
    X_transformed, y_transformed = sample_adapter._dataloader_to_arrays(all_data)
    feature_names = sample_adapter.get_feature_names()
    
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Training samples: {len(X_transformed)}")
    
    # Store results
    importance_results = {}
    
    for model_name, model in tree_models.items():
        print(f"\n{'-' * 40}")
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_transformed, y_transformed)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_results[model_name] = {
                'importances': importances,
                'type': 'built-in'
            }
            
            # Print top 10 features
            indices = np.argsort(importances)[::-1][:10]
            print(f"\nTop 10 features for {model_name}:")
            for i, idx in enumerate(indices):
                print(f"{i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")
    
    return importance_results, feature_names, X_transformed, y_transformed


def analyze_linear_model_coefficients():
    """Analyze coefficients for linear models."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS - LINEAR MODELS")
    print("=" * 80)
    
    # Setup
    schema = get_schema('time_goal_extended')
    dataset = SchemaBasedTimeSeriesDataset(
        data_path='../data-analysis/student_week_aggregations_rolling_new.csv',
        schema=schema,
        sequence_length=5,
        validate_data=False
    )
    
    # Linear models with coefficients
    linear_models = {
        'lasso': Lasso(alpha=0.1, max_iter=2000),
        'ridge': Ridge(alpha=1.0),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    }
    
    # Get data
    all_data = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Create adapter
    sample_adapter = SKLearnAdapter(
        sklearn_model=linear_models['lasso'],
        schema=schema,
        lag_window=5
    )
    
    X_transformed, y_transformed = sample_adapter._dataloader_to_arrays(all_data)
    feature_names = sample_adapter.get_feature_names()
    
    # Standardize features for fair coefficient comparison
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    
    coefficient_results = {}
    
    for model_name, model in linear_models.items():
        print(f"\n{'-' * 40}")
        print(f"Training {model_name}...")
        
        # Train on scaled data
        model.fit(X_scaled, y_transformed)
        
        # Get coefficients
        coefficients = np.abs(model.coef_)  # Absolute value for importance
        coefficient_results[model_name] = {
            'importances': coefficients,
            'type': 'coefficients'
        }
        
        # Print top 10 features
        indices = np.argsort(coefficients)[::-1][:10]
        print(f"\nTop 10 features for {model_name} (by absolute coefficient):")
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:40s} {coefficients[idx]:.4f}")
    
    return coefficient_results, feature_names


def analyze_permutation_importance(X, y, feature_names):
    """Analyze permutation importance for any model."""
    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Models to test with permutation importance
    models = {
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42),
        'mlp': None  # We'll handle this separately
    }
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    perm_results = {}
    
    for model_name, model in models.items():
        if model is None and model_name == 'mlp':
            # Skip MLP for now - would need PyTorch adapter
            continue
            
        print(f"\n{'-' * 40}")
        print(f"Permutation importance for {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        perm_results[model_name] = {
            'importances': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'type': 'permutation'
        }
        
        # Print top 10
        indices = np.argsort(perm_importance.importances_mean)[::-1][:10]
        print(f"\nTop 10 features by permutation importance:")
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:40s} "
                  f"{perm_importance.importances_mean[idx]:.4f} "
                  f"(±{perm_importance.importances_std[idx]:.4f})")
    
    return perm_results


def create_comparative_visualizations(all_results, feature_names):
    """Create visualizations comparing feature importance across models."""
    
    # 1. Top features comparison heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Get top 20 features across all models
    all_importances = []
    for model_name, result in all_results.items():
        importances = result['importances']
        # Normalize to 0-1 scale
        if np.max(importances) > 0:
            normalized = importances / np.max(importances)
        else:
            normalized = importances
        all_importances.extend(normalized)
    
    # Find features that appear in top 20 for any model
    top_features_set = set()
    for model_name, result in all_results.items():
        importances = result['importances']
        indices = np.argsort(importances)[::-1][:20]
        top_features_set.update(indices)
    
    top_features_list = sorted(list(top_features_set))[:25]  # Limit to 25
    
    # Create heatmap data
    heatmap_data = []
    model_names = []
    
    for model_name, result in all_results.items():
        importances = result['importances']
        # Normalize
        if np.max(importances) > 0:
            normalized = importances / np.max(importances)
        else:
            normalized = importances
        
        row = [normalized[idx] for idx in top_features_list]
        heatmap_data.append(row)
        model_names.append(model_name)
    
    # Create heatmap
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=model_names,
        columns=[feature_names[idx] for idx in top_features_list]
    )
    
    sns.heatmap(
        heatmap_df.T, 
        cmap='YlOrRd', 
        cbar_kws={'label': 'Normalized Importance'},
        ax=ax1
    )
    ax1.set_title('Feature Importance Heatmap (Top 25 Features)', fontsize=14)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Feature')
    
    # 2. Gap features analysis
    gap_feature_indices = [i for i, name in enumerate(feature_names) 
                          if 'gap' in name.lower()]
    
    if gap_feature_indices:
        gap_data = []
        for model_name, result in all_results.items():
            importances = result['importances']
            gap_importances = [importances[idx] for idx in gap_feature_indices]
            gap_data.append(gap_importances)
        
        gap_df = pd.DataFrame(
            gap_data,
            index=model_names,
            columns=[feature_names[idx] for idx in gap_feature_indices]
        )
        
        gap_df.plot(kind='bar', ax=ax2)
        ax2.set_title('Gap Feature Importance Across Models', fontsize=14)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Feature Importance')
        ax2.legend(title='Gap Features', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nFeature importance comparison saved as 'feature_importance_comparison.png'")
    
    # 3. Top features by category
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Categorize features
    categories = {
        'Current': [],
        'Lag': [],
        'Statistical': [],
        'Change': [],
        'Gap': [],
        'Interaction': []
    }
    
    for i, name in enumerate(feature_names):
        if 'current_' in name:
            categories['Current'].append(i)
        elif 'lag' in name:
            categories['Lag'].append(i)
        elif any(stat in name for stat in ['mean', 'std', 'sum', 'range', 'iqr', 'trend', 'acceleration']):
            categories['Statistical'].append(i)
        elif 'change' in name:
            categories['Change'].append(i)
        elif 'gap' in name:
            categories['Gap'].append(i)
        elif 'x' in name:
            categories['Interaction'].append(i)
    
    # Calculate average importance by category
    category_importance = {}
    for model_name, result in all_results.items():
        importances = result['importances']
        model_category_imp = {}
        
        for cat_name, indices in categories.items():
            if indices:
                avg_imp = np.mean([importances[idx] for idx in indices])
                model_category_imp[cat_name] = avg_imp
        
        category_importance[model_name] = model_category_imp
    
    # Create grouped bar chart
    cat_df = pd.DataFrame(category_importance).T
    cat_df.plot(kind='bar', ax=ax)
    ax.set_title('Average Feature Importance by Category', fontsize=14)
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Importance')
    ax.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance_by_category.png', dpi=150, bbox_inches='tight')
    print("Feature importance by category saved as 'feature_importance_by_category.png'")


def main():
    """Run complete feature importance analysis."""
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # 1. Tree-based models
    tree_results, feature_names, X, y = analyze_tree_based_importance()
    
    # 2. Linear models
    linear_results, _ = analyze_linear_model_coefficients()
    
    # 3. Permutation importance
    perm_results = analyze_permutation_importance(X, y, feature_names)
    
    # Combine all results
    all_results = {**tree_results, **linear_results, **perm_results}
    
    # 4. Create visualizations
    create_comparative_visualizations(all_results, feature_names)
    
    # 5. Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: MOST IMPORTANT FEATURES ACROSS ALL MODELS")
    print("=" * 80)
    
    # Aggregate importance across models
    feature_scores = np.zeros(len(feature_names))
    model_count = 0
    
    for model_name, result in all_results.items():
        importances = result['importances']
        # Normalize to 0-1
        if np.max(importances) > 0:
            normalized = importances / np.max(importances)
            feature_scores += normalized
            model_count += 1
    
    # Average across models
    feature_scores /= model_count
    
    # Print top 15 overall
    indices = np.argsort(feature_scores)[::-1][:15]
    print("\nTop 15 features by average normalized importance:")
    for i, idx in enumerate(indices):
        print(f"{i+1:2d}. {feature_names[idx]:40s} {feature_scores[idx]:.4f}")
    
    # Gap features analysis
    print("\n" + "-" * 40)
    print("GAP FEATURES ANALYSIS:")
    gap_indices = [i for i, name in enumerate(feature_names) if 'gap' in name.lower()]
    
    for idx in gap_indices:
        avg_rank = 0
        appearances = 0
        for model_name, result in all_results.items():
            importances = result['importances']
            rank = len(importances) - np.argsort(importances).argsort()[idx]
            avg_rank += rank
            appearances += 1
        avg_rank /= appearances
        print(f"{feature_names[idx]:30s} - Average rank: {avg_rank:.1f}/53")
    
    return all_results, feature_names


if __name__ == "__main__":
    results, features = main()
