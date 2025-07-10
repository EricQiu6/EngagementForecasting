# Prediction Error Investigation Plan

## Overview
This document outlines approaches to investigate whether models consistently predict certain ranges of `minutes_per_week` values less accurately than others.

## Target Variable Distribution

Based on our data analysis:
- **Mean**: 16.94 minutes/week
- **Std**: 13.80 minutes/week
- **Range**: 0.00 - 176.71 minutes/week

### Key Percentiles
- 10%: 2.62 minutes (very low engagement)
- 25%: 6.52 minutes (low engagement)
- 50%: 13.45 minutes (median)
- 75%: 24.25 minutes (high engagement)
- 90%: 35.99 minutes (very high engagement)
- 99%: 54.81 minutes (extreme engagement)

## Recommended Analysis Approaches

### 1. Residual Analysis by Target Range (Start Here)

**Approach**: Bin target values and analyze prediction errors for each bin

**Implementation**:
```python
bins = [0, 5, 15, 30, 50, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
```

**Metrics to Calculate**:
- MAE per bin
- Mean error (bias) per bin
- Error variance per bin
- Percentage of predictions within ±5 minutes

**Expected Insights**:
- Whether models struggle with extreme values
- If there's systematic under/over prediction in certain ranges

### 2. Scatter Plot Analysis

**Approach**: Visual inspection of predicted vs actual values

**Key Elements**:
- Diagonal reference line (perfect predictions)
- Color coding by error magnitude
- LOESS smoothing to show trends
- Separate plots for each model

**What to Look For**:
- Systematic deviations from diagonal
- Heteroscedasticity (changing error variance)
- Clustering patterns

### 3. Zero and Near-Zero Analysis

**Approach**: Special focus on zero/low engagement predictions

**Analysis**:
- Binary classification metrics for zero detection
- Error distribution for values < 5 minutes
- False positive rate (predicting activity when none)
- False negative rate (missing actual activity)

**Why Important**:
- Zero values may represent different phenomena (dropouts, breaks)
- Models might struggle with discontinuous behavior

### 4. Extreme Value Analysis

**Approach**: Separate analysis for distribution tails

**Categories**:
- Bottom 10%: < 2.62 minutes
- Top 10%: > 35.99 minutes
- Middle 80%: 2.62 - 35.99 minutes

**Comparisons**:
- Error metrics by category
- Feature importance differences
- Model rankings by category

### 5. Bias-Variance Decomposition

**Approach**: Analyze systematic vs random errors

**Calculate**:
- Bias²: (mean(predicted - actual))²
- Variance: var(predicted - actual)
- Total MSE: Bias² + Variance

**By Range**: Calculate for each target value bin

## Hypotheses to Test

### H1: Mean Reversion Bias
**Hypothesis**: Models predict values closer to mean (16.94) than actual extremes
**Test**: Compare predicted range vs actual range

### H2: Zero-Inflation Challenge
**Hypothesis**: Models struggle with zero values (complete disengagement)
**Test**: Separate analysis for zero vs non-zero predictions

### H3: High Engagement Underestimation
**Hypothesis**: Models underpredict students with >50 minutes/week
**Test**: Bias analysis for top 5% of values

### H4: Heteroscedastic Errors
**Hypothesis**: Prediction errors increase with target value
**Test**: Plot absolute error vs target value

### H5: Student-Specific Patterns
**Hypothesis**: Some students are consistently harder to predict
**Test**: Aggregate errors by student, identify outliers

## Implementation Priority

1. **Quick Win**: Create predicted vs actual scatter plots
2. **Core Analysis**: Residual analysis by binned ranges
3. **Deep Dive**: Zero-inflation and extreme value analysis
4. **Advanced**: Student-specific error patterns

## Expected Outcomes

This analysis will reveal:
- Whether certain engagement levels are inherently harder to predict
- If we need specialized models for different ranges
- Whether feature engineering should differ by engagement level
- If ensemble methods combining range-specific models would help

## Next Steps

After identifying patterns:
1. Implement range-specific model adjustments
2. Consider ensemble approaches
3. Develop range-aware evaluation metrics
4. Create monitoring for production systems
