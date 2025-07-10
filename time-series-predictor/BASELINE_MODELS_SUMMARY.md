# Baseline Models Summary

## Verification Results

I have successfully verified and **added** the following four baseline models to the `time-series-predictor/src/framework/models/baselines.py` file:

## New Baseline Models

### 1. MedianPredictor
- **Purpose**: Predicts the median of past target values
- **Behavior**: Uses all historical values (including zeros) to compute the median
- **Fallback**: Uses global median of training data when no historical data is available
- **Compatible**: Works with SKLearnAdapter and follows the same interface as other baseline models

### 2. MedianWithoutZeroPredictor  
- **Purpose**: Predicts the median of past target values, ignoring zeros
- **Behavior**: Filters out zero values from historical data before computing median
- **Fallback**: Returns 0 if no non-zero past target values are available
- **Compatible**: Works with SKLearnAdapter and follows the same interface as other baseline models

### 3. MeanPredictor
- **Purpose**: Predicts the mean of past target values
- **Behavior**: Uses all historical values (including zeros) to compute the mean
- **Fallback**: Uses global mean of training data when no historical data is available
- **Compatible**: Works with SKLearnAdapter and follows the same interface as other baseline models

### 4. MeanWithoutZeroPredictor
- **Purpose**: Predicts the mean of past target values, ignoring zeros
- **Behavior**: Filters out zero values from historical data before computing mean
- **Fallback**: Returns 0 if no non-zero past target values are available
- **Compatible**: Works with SKLearnAdapter and follows the same interface as other baseline models

## Implementation Details

All four models:
- ✅ **Support feature metadata**: Can automatically detect lag features using metadata from SKLearnAdapter
- ✅ **Fallback handling**: Use hardcoded lag positions (9-13) when metadata is not available
- ✅ **Sklearn compatibility**: Implement required methods (`fit`, `predict`, `get_params`)
- ✅ **Robust error handling**: Include proper validation and fallback mechanisms
- ✅ **Tested and verified**: All models have been tested with synthetic data and work correctly

## Existing Models (Already Present)

The following baseline models were already present in the codebase:
1. **AveragePredictor** - Predicts the training mean for all samples
2. **DLinearWrapper** - Wrapper for DLinear model compatible with SKLearnAdapter
3. **NaiveForecast** - Uses the last observed value
4. **LinearTrend** - Fits a linear trend to the historical sequence

## Usage

These models can be used anywhere in the codebase where baseline predictors are needed. They are fully compatible with the existing framework and can be instantiated and used like any other baseline model:

```python
from framework.models.baselines import MedianPredictor, MedianWithoutZeroPredictor

# Example usage
model = MedianPredictor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Testing

All four new models have been verified to work correctly with:
- ✅ Proper prediction shapes
- ✅ Non-negative predictions
- ✅ Finite predictions
- ✅ Reasonable prediction ranges
- ✅ Correct handling of zero values (where applicable)