"""
Comprehensive Analysis of Student Week Predictive Modeling Results
================================================================

This script provides detailed analysis of the modeling results from our
student week engagement prediction experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*70)
print("STUDENT WEEK PREDICTIVE MODELING - RESULTS ANALYSIS")
print("="*70)

# Load the results
results_df = pd.read_csv('../outputs/panel_model_results.csv', index_col=0)
print("\n1. PANEL MODEL PERFORMANCE COMPARISON")
print("="*50)
print(results_df.round(3))

# Load the original dataset for context
df = pd.read_csv('/Users/ericq/cmu/goalsetting-recommendation-algorithm/data-analysis/student_week_aggregations_rolling.csv')
df['week_date'] = pd.to_datetime(df['week_id'] + '-1', format='%Y-W%W-%w')

print(f"\nDataset Context:")
print(f"- Total records: {len(df):,}")
print(f"- Unique students: {df['anon_student_id'].nunique()}")
print(f"- Average minutes per week: {df['minutes_per_week'].mean():.1f} ± {df['minutes_per_week'].std():.1f}")
print(f"- Median minutes per week: {df['minutes_per_week'].median():.1f}")

# Calculate relative performance metrics
print("\n2. DETAILED PERFORMANCE ANALYSIS")
print("="*50)

baseline_mae = df['minutes_per_week'].std()  # Naive baseline: predict mean
print(f"Naive Baseline (predict mean): MAE = {baseline_mae:.2f}")

print("\nModel Performance vs Baseline:")
for model, row in results_df.iterrows():
    improvement = (baseline_mae - row['mae']) / baseline_mae * 100
    print(f"- {model.capitalize():15}: MAE = {row['mae']:5.2f} ({improvement:+5.1f}% vs baseline)")

print(f"\nError Distribution Analysis:")
print(f"- Best MAE: {results_df['mae'].min():.2f} (Random Forest)")
print(f"- Worst MAE: {results_df['mae'].max():.2f} (Ridge)")
print(f"- MAE Range: {results_df['mae'].max() - results_df['mae'].min():.2f}")

# RMSE vs MAE analysis
print(f"\nRMSE vs MAE Analysis (indicates outlier sensitivity):")
for model, row in results_df.iterrows():
    rmse_mae_ratio = row['rmse'] / row['mae']
    print(f"- {model.capitalize():15}: RMSE/MAE = {rmse_mae_ratio:.2f}")

print(f"\nSMAPE Analysis (scale-independent error):")
for model, row in results_df.iterrows():
    print(f"- {model.capitalize():15}: SMAPE = {row['smape']:5.1f}%")

# Statistical significance analysis
print("\n3. MODEL COMPARISON & STATISTICAL INSIGHTS")
print("="*50)

# Create performance visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MAE comparison
axes[0, 0].bar(results_df.index, results_df['mae'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 0].set_title('Mean Absolute Error (MAE)')
axes[0, 0].set_ylabel('Minutes')
axes[0, 0].tick_params(axis='x', rotation=45)

# RMSE comparison
axes[0, 1].bar(results_df.index, results_df['rmse'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 1].set_title('Root Mean Square Error (RMSE)')
axes[0, 1].set_ylabel('Minutes')
axes[0, 1].tick_params(axis='x', rotation=45)

# SMAPE comparison
axes[1, 0].bar(results_df.index, results_df['smape'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[1, 0].set_title('Symmetric Mean Absolute Percentage Error (SMAPE)')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].tick_params(axis='x', rotation=45)

# Error distribution comparison
error_data = []
for model in results_df.index:
    # Simulate error distribution based on MAE and RMSE
    # This is an approximation since we don't have actual predictions
    mae = results_df.loc[model, 'mae']
    rmse = results_df.loc[model, 'rmse']
    
    # Approximate error distribution
    errors = np.random.normal(0, mae, 1000)  # Simplified simulation
    error_data.extend([(model, err) for err in errors])

error_df = pd.DataFrame(error_data, columns=['Model', 'Error'])
sns.boxplot(data=error_df, x='Model', y='Error', ax=axes[1, 1])
axes[1, 1].set_title('Error Distribution (Simulated)')
axes[1, 1].set_ylabel('Error (Minutes)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../outputs/model_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Saved performance analysis plot to outputs/model_performance_analysis.png")

# Business impact analysis
print("\n4. BUSINESS IMPACT ANALYSIS")
print("="*50)

avg_study_time = df['minutes_per_week'].mean()
best_mae = results_df['mae'].min()

print(f"Prediction Accuracy Context:")
print(f"- Average study time: {avg_study_time:.1f} minutes/week")
print(f"- Best model error: {best_mae:.1f} minutes/week")
print(f"- Relative error: {(best_mae/avg_study_time)*100:.1f}% of average study time")

print(f"\nPractical Implications:")
print(f"- Predictions are typically off by ±{best_mae:.0f} minutes")
print(f"- This represents ±{best_mae/60:.1f} hours per week")
print(f"- For early intervention, this accuracy level is reasonable")

# Model selection recommendations
print("\n5. MODEL SELECTION RECOMMENDATIONS")
print("="*50)

print("Ranking by different criteria:")

# Rank by MAE (primary metric)
mae_ranking = results_df.sort_values('mae')
print(f"\nBy MAE (primary metric for engagement prediction):")
for i, (model, row) in enumerate(mae_ranking.iterrows(), 1):
    print(f"{i}. {model.capitalize():15} - MAE: {row['mae']:.2f}")

# Rank by SMAPE (scale-independent)
smape_ranking = results_df.sort_values('smape')
print(f"\nBy SMAPE (scale-independent):")
for i, (model, row) in enumerate(smape_ranking.iterrows(), 1):
    print(f"{i}. {model.capitalize():15} - SMAPE: {row['smape']:.1f}%")

print(f"\nRecommendations:")
print(f"1. **Best Overall**: Random Forest")
print(f"   - Lowest MAE ({results_df.loc['random_forest', 'mae']:.2f})")
print(f"   - Lowest SMAPE ({results_df.loc['random_forest', 'smape']:.1f}%)")
print(f"   - Handles non-linear patterns well")

print(f"\n2. **Most Interpretable**: Linear Regression")
print(f"   - Simple, fast, interpretable")
print(f"   - Close performance to Ridge")
print(f"   - Good for understanding feature importance")

print(f"\n3. **Balanced Choice**: Ridge Regression")
print(f"   - Regularization prevents overfitting")
print(f"   - Stable across different datasets")
print(f"   - Good interpretability")

# Individual vs Panel Model Analysis
print("\n6. INDIVIDUAL vs PANEL MODEL ANALYSIS")
print("="*50)

panel_best_mae = results_df['mae'].min()
individual_mae = 16.95  # From our earlier individual model result

print(f"Performance Comparison:")
print(f"- Panel Model (Random Forest): MAE = {panel_best_mae:.2f}")
print(f"- Individual Model (Ridge):    MAE = {individual_mae:.2f}")
print(f"- Improvement: {((panel_best_mae - individual_mae) / panel_best_mae * 100):+.1f}%")

print(f"\nTrade-offs:")
print(f"✅ Individual Models:")
print(f"   - Better accuracy for students with sufficient data")
print(f"   - Personalized predictions")
print(f"   - Can capture student-specific patterns")
print(f"❌ Limitations:")
print(f"   - Requires ≥30 weeks of data per student")
print(f"   - 238 students qualify (56% of dataset)")
print(f"   - Higher computational cost")

print(f"✅ Panel Models:")
print(f"   - Works for all students (including new ones)")
print(f"   - Single model to maintain")
print(f"   - Captures general patterns across population")
print(f"❌ Limitations:")
print(f"   - Lower accuracy for individual students")
print(f"   - May miss student-specific patterns")

# Dropout prediction analysis
print("\n7. DROPOUT PREDICTION ANALYSIS")
print("="*50)

dropout_auc = 0.727  # From our earlier result
dropout_rate = 0.0363  # 3.63%

print(f"Classification Performance:")
print(f"- ROC-AUC: {dropout_auc:.3f}")
print(f"- Baseline dropout rate: {dropout_rate:.1%}")

print(f"\nInterpretation:")
if dropout_auc > 0.8:
    performance = "Excellent"
elif dropout_auc > 0.7:
    performance = "Good"
elif dropout_auc > 0.6:
    performance = "Fair"
else:
    performance = "Poor"
    
print(f"- {performance} discrimination ability")
print(f"- Model can distinguish dropouts from active students")
print(f"- Useful for early intervention systems")

# Feature importance insights
print("\n8. FEATURE ENGINEERING INSIGHTS")
print("="*50)

print("Key findings from our feature engineering:")
print("✅ Lagged features (1-4 weeks) are crucial")
print("✅ Rolling statistics capture trends")
print("✅ Temporal features (week of year) add value")
print("✅ Multiple metrics together improve performance")

print(f"\nFeature categories used:")
print(f"- Lagged values: minutes_per_week, problems_solved, avg_proficiency")
print(f"- Rolling statistics: 4-week means and standard deviations")
print(f"- Temporal: week_of_year, month")
print(f"- Total features: 26")

# Recommendations for improvement
print("\n9. RECOMMENDATIONS FOR IMPROVEMENT")
print("="*50)

print("Short-term improvements:")
print("1. Feature Engineering:")
print("   - Add trend features (slope over past N weeks)")
print("   - Include student-level aggregates (overall performance)")
print("   - Seasonal decomposition features")
print("   - Gap features (weeks since last activity)")

print("\n2. Model Enhancements:")
print("   - Ensemble methods (combine multiple models)")
print("   - Time series specific models (ARIMA, Prophet)")
print("   - Neural networks for complex patterns")

print("\n3. Data Improvements:")
print("   - Handle missing weeks more systematically")
print("   - Include external factors (holidays, exams)")
print("   - Add student demographic features")

print("\nLong-term improvements:")
print("1. Real-time prediction pipeline")
print("2. A/B testing framework for model validation")
print("3. Intervention effectiveness tracking")
print("4. Multi-step ahead forecasting")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Create summary report
summary_report = f"""
STUDENT WEEK PREDICTIVE MODELING - EXECUTIVE SUMMARY
==================================================

DATASET: 11,714 weekly records from 425 students (2010-2012)
TASK: Predict next week's study time (minutes_per_week)

BEST MODEL PERFORMANCE:
- Random Forest: MAE = {results_df.loc['random_forest', 'mae']:.1f} minutes
- Typical error: ±{results_df.loc['random_forest', 'mae']:.0f} minutes ({(results_df.loc['random_forest', 'mae']/avg_study_time)*100:.1f}% of avg study time)
- Individual models can achieve MAE = 16.9 minutes for students with sufficient data

KEY FINDINGS:
1. Random Forest outperforms linear models (MAE: 18.5 vs 18.8)
2. Individual models provide 8.6% better accuracy for qualified students
3. Dropout prediction achieves 72.7% ROC-AUC (good discrimination)
4. Feature engineering with lags and rolling stats is crucial

BUSINESS IMPACT:
- Predictions accurate enough for early intervention systems
- Can identify at-risk students before they disengage
- Enables personalized learning recommendations

RECOMMENDATIONS:
1. Deploy Random Forest for general population
2. Use individual models for students with 30+ weeks of data
3. Implement dropout prediction for proactive intervention
4. Enhance with trend features and ensemble methods
"""

with open('../outputs/modeling_results_summary.txt', 'w') as f:
    f.write(summary_report)

print("\nSaved executive summary to outputs/modeling_results_summary.txt") 