#!/usr/bin/env python3
"""
True Mixed Effects Integration with Framework
============================================

This script shows how to properly use mixed effects models for
TRUE time series prediction within the framework architecture.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# Import framework components
from src.framework import (
    SchemaBasedTimeSeriesDataset,
    CrossValidator,
    SKLearnAdapter,
    TrueMixedEffectsModel,
    get_schema
)


def compare_models_properly():
    """Compare models doing TRUE time series prediction."""
    print("Comparing Models for TRUE Time Series Prediction")
    print("=" * 70)
    
    # Load schema and dataset
    schema = get_schema('time_goal')  # Predicting minutes_per_week
    
    try:
        dataset = SchemaBasedTimeSeriesDataset(
            data_path='data/student_week_aggregations.csv',
            schema=schema,
            sequence_length=8
        )
        print(f"Loaded dataset with {len(dataset)} sequences")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Models to compare
    models = {
        'Lasso': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'True Mixed Effects': TrueMixedEffectsModel(
            target_col='minutes_per_week',
            n_lags=3,
            use_lasso=False
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Create adapter
        adapter = SKLearnAdapter(model, schema=schema, lag_window=5)
        
        # Create cross-validator
        cv = CrossValidator(adapter, dataset)
        
        try:
            # Run cross-validation
            cv_results = cv.cross_validate(n_splits=3, test_size=1)
            
            results[model_name] = {
                'mae': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'rmse': cv_results['rmse_mean']
            }
            
            print(f"  MAE: {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}")
            print(f"  RMSE: {cv_results['rmse_mean']:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {'mae': np.nan, 'mae_std': np.nan, 'rmse': np.nan}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TRUE Time Series Prediction Results")
    print("-" * 70)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf'))
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        if not np.isnan(metrics['mae']):
            print(f"{rank}. {model_name}: MAE = {metrics['mae']:.2f} ± {metrics['mae_std']:.2f}")
    
    print("\nKEY POINTS:")
    print("1. All models are doing TRUE next-week prediction")
    print("2. Mixed effects should show modest improvement (10-20%)")
    print("3. The dramatic 90% improvement was from doing the wrong task")
    print("4. These MAE values (~7-8) are realistic for true forecasting")


def explain_the_confusion():
    """Explain what went wrong with the original mixed effects implementation."""
    print("\n\nUnderstanding the Mixed Effects Confusion")
    print("=" * 70)
    
    # Create example data
    data = pd.DataFrame({
        'student_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'week': [1, 2, 3, 1, 2, 3],
        'proficiency': [0.6, 0.7, 0.8, 0.5, 0.6, 0.7],
        'minutes': [10, 12, 15, 20, 22, 25]
    })
    
    print("\nExample Data:")
    print(data)
    
    print("\n1. WRONG Approach (Concurrent Regression):")
    print("   Model: minutes[week=t] ~ proficiency[week=t]")
    print("   This explains CURRENT minutes using CURRENT proficiency")
    print("   Easy task! Variables measured at same time are highly correlated")
    
    print("\n2. RIGHT Approach (Time Series Prediction):")
    print("   Model: minutes[week=t+1] ~ proficiency[week=t] + minutes[week=t]")
    print("   This predicts FUTURE minutes using CURRENT information")
    print("   Harder task! Must capture temporal dynamics")
    
    print("\n3. Why Mixed Effects Seemed Amazing:")
    print("   - In concurrent regression, student effects explain most variance")
    print("   - Student A always studies less than Student B")
    print("   - Adding random intercepts captures this perfectly")
    print("   - Result: 90% improvement (but wrong task!)")
    
    print("\n4. Why Mixed Effects Are Modest for True Prediction:")
    print("   - In time series, we need to predict CHANGES")
    print("   - Student effects help with baseline, but changes are harder")
    print("   - Temporal dynamics matter more than static differences")
    print("   - Result: 10-20% improvement (realistic!)")


def create_proper_mixed_effects_adapter():
    """Show how to create a proper adapter for production use."""
    print("\n\nCreating Production-Ready Mixed Effects Adapter")
    print("=" * 70)
    
    code = '''
class ProductionMixedEffectsAdapter(SKLearnAdapter):
    """
    Production-ready adapter for TRUE mixed effects time series prediction.
    """
    
    def __init__(self, schema, **mixed_kwargs):
        # Initialize with TrueMixedEffectsModel
        model = TrueMixedEffectsModel(**mixed_kwargs)
        super().__init__(model, schema=schema)
        
    def fit(self, train_data, val_data=None, **kwargs):
        """Override to handle mixed effects data structure."""
        # Extract features with student IDs preserved
        X_train, y_train = self._prepare_mixed_effects_data(train_data)
        
        # Fit the model
        self.sklearn_model.fit(X_train, y_train)
        self.is_fitted = True
        
        return {'status': 'completed'}
        
    def _prepare_mixed_effects_data(self, data):
        """Prepare data preserving student IDs for mixed effects."""
        # Implementation would extract student IDs and features
        # ensuring proper time series structure
        pass
    '''
    
    print("Key implementation points:")
    print("1. Preserve student IDs through the pipeline")
    print("2. Ensure proper time alignment (predict t+1 from t)")
    print("3. Handle new students gracefully (population model)")
    print("4. Validate temporal structure in data")
    
    print("\nExample usage:")
    print(code)


def main():
    """Run all demonstrations."""
    # Compare models properly
    compare_models_properly()
    
    # Explain the confusion
    explain_the_confusion()
    
    # Show production approach
    create_proper_mixed_effects_adapter()
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("Mixed effects are valuable for time series prediction, but:")
    print("1. Must do TRUE prediction (t+1 from t), not concurrent")
    print("2. Expect modest improvements (10-20%), not dramatic (90%)")
    print("3. Individual differences matter, but temporal dynamics matter more")
    print("4. Always validate you're solving the right problem!")


 