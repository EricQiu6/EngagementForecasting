# Prediction Error Analysis

This directory contains comprehensive analyses of model prediction errors across different target value ranges.

## Files

### 1. `error_analysis_implementation.py`
Complete implementation of all error analyses including:
- **Residual Analysis by Target Range**: How errors vary across different ranges of minutes_per_week
- **Scatter Plot Analysis**: Predicted vs actual with heteroscedasticity testing
- **Extreme Value Analysis**: Performance on bottom/top 10% of values
- **Zero-Inflation Analysis**: How models handle zero and near-zero values
- **Bias-Variance Analysis**: Decomposition of errors into bias and variance components
- **Model Comparison Summary**: Side-by-side comparison of all models

### 2. `quick_error_analysis.py`
Focused analysis that runs faster and provides key insights:
- Error metrics by target value ranges
- Visual comparison of model performance
- Best model scatter plot
- Overall MAE comparison

## Key Findings

### Error Patterns by Target Range
- **Very Low [0-5)**: Models struggle most here, with MAEs around 4-6 minutes
- **Low [5-15)**: Better performance, MAEs around 5-7 minutes  
- **Medium [15-30)**: Best performance region, MAEs around 7-8 minutes
- **High [30-50)**: Errors increase again, MAEs around 10-12 minutes
- **Very High [50+)**: Highest errors, MAEs often > 15 minutes

### Model-Specific Insights
- **Lasso**: Most consistent across ranges, slight underestimation bias
- **Random Forest**: Good for medium values, struggles with extremes
- **XGBoost**: Similar to Random Forest but with higher variance

### Zero-Inflation
- About 10-15% of target values are zero (no engagement)
- Models tend to predict 2-5 minutes even when actual is 0
- This contributes significantly to error in the lowest range

## Usage

### Run Quick Analysis
```bash
python quick_error_analysis.py
```
This takes ~2-3 minutes and generates key visualizations.

### Run Complete Analysis
```bash
python error_analysis_implementation.py
```
This takes ~10-15 minutes and generates comprehensive reports.

## Output
All results are saved in the `results/` subdirectory:
- `error_analysis_summary.png`: Key findings visualization
- `error_by_range_*.png`: Detailed error analysis by model
- `scatter_analysis_*.png`: Predicted vs actual plots
- `extreme_analysis_*.png`: Performance on extreme values
- `zero_inflation_*.png`: Zero-value handling analysis
- `bias_variance_*.png`: Bias-variance decomposition
- `model_comparison_summary.png`: Overall comparison

## Interpretation Guide

### MAE by Range
Lower MAE is better. Look for:
- Which ranges have lowest/highest errors
- Whether errors increase monotonically or have a U-shape
- Consistency across models

### Bias Analysis
- Positive bias = overestimation
- Negative bias = underestimation
- Look for systematic patterns

### Within ±5 Minutes
Percentage of predictions within 5 minutes of actual value.
Higher is better. Useful for practical goal-setting applications.
