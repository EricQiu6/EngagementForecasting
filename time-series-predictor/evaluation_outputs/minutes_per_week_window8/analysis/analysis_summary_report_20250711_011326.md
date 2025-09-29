# Comprehensive Model Analysis Report

Generated on: 2025-07-11 01:13:30

## Evaluation Configuration

- **schema_name**: time_goal_extended
- **window_size**: 8
- **target_type**: minutes_per_week
- **n_splits**: 5
- **test_size**: 1
- **timestamp**: 2025-07-11T01:07:57.480994

## Model Performance Summary

### Top 5 Models by MAE

            model category  mae_mean  mae_std  rmse_mean  r2_mean
            lasso   linear  8.173302 1.094754  11.591448 0.222883
    random_forest     tree  8.186875 1.222290  11.580135 0.224400
            ridge   linear  8.234377 1.124880  11.660659 0.213576
linear_regression   linear  8.262415 1.133264  11.704485 0.207653
          xgboost     tree  8.399708 1.180951  11.956013 0.173232

## Statistical Significance Testing

Total pairwise comparisons: 120
Significant differences (p < 0.05): 86

### Top 5 Most Significant Differences

- **adams_baseline_70** vs **lasso**: p = 0.0000, effect size = medium
- **adams_baseline_60** vs **lasso**: p = 0.0000, effect size = medium
- **adams_baseline_50** vs **lasso**: p = 0.0000, effect size = medium
- **random_forest** vs **adams_baseline_70**: p = 0.0000, effect size = medium
- **random_forest** vs **adams_baseline_60**: p = 0.0000, effect size = medium

## Performance by Category

### Baseline
- Number of models: 5
- Average MAE: 10.157 ± 1.182
- Best MAE: 9.602
- Worst MAE: 12.270

### Goal_Based
- Number of models: 3
- Average MAE: 13.453 ± 0.000
- Best MAE: 13.453
- Worst MAE: 13.453

### Linear
- Number of models: 3
- Average MAE: 8.223 ± 0.046
- Best MAE: 8.173
- Worst MAE: 8.262

### Mixed_Effects
- Number of models: 1
- Average MAE: 8.524 ± nan
- Best MAE: 8.524
- Worst MAE: 8.524

### Neural
- Number of models: 2
- Average MAE: 9.563 ± 0.372
- Best MAE: 9.300
- Worst MAE: 9.826

### Tree
- Number of models: 2
- Average MAE: 8.293 ± 0.150
- Best MAE: 8.187
- Worst MAE: 8.400

## Recommendations

1. **Best Overall Model**: lasso (linear)
   - MAE: 8.173 ± 1.095
   - RMSE: 11.591
   - R²: 0.223

2. **Most Stable Model**: median_all (baseline)
   - MAE: 9.619 ± 1.070
   - Consistency across folds is highest

## Generated Files

- `model_performance_summary.csv`: Detailed performance metrics
- `significance_testing.csv`: Statistical significance test results
- `bootstrap_confidence_intervals.csv`: Bootstrap confidence intervals
- `predicted_vs_actual.png`: Scatter plots of predictions vs actuals
- `error_distributions.png`: Error distribution histograms
- `residual_plots.png`: Residual analysis plots
- `model_comparison.png`: Performance comparison charts
- `performance_by_category.png`: Category-wise performance analysis
