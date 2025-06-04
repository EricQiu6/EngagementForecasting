# Time-Series Predictor for Student Goal Recommendation

This project implements a baseline forecasting pipeline for predicting student performance in skills-mastered tasks, as specified in the baseline specification document.

## Project Overview

The goal is to create a learning algorithm that can recommend weekly practice goals for students based on their historical performance data. This baseline implementation focuses on predicting skills-mastered performance using autoregressive modeling.

## Implementation Progress

### ✅ Completed Steps (1-3)

#### Step 1: Choose the target series

- **Target variable**: Skills-mastered (using `proficient` column)
- **Rationale**: Focus on skill proficiency as the key performance metric

#### Step 2: Load & tidy the data

- **Input**: `exp-static-flexible-anon-2025-05-29.csv` (1,893 rows, 16 columns)
- **Output**: `data_tidied.csv` (1,893 rows, 3 columns: name, week, proficient)
- **Processing steps**:
  - Selected only required columns: `name`, `week`, `proficient`
  - Sorted by (name, week) and removed duplicates
  - Analyzed missing weeks within student spans (none found)
  - Forward-filled missing proficient values (3 missing values handled)
  - Initial missing values filled with 0 (no skills mastered initially)

#### Step 3: Split chronologically ✨ NEW

- **Per student**: Keep last K=3 weeks as test set, everything before as training
- **Intelligent K selection**: Analyzed data to recommend K=3 (27.3% test data, ≥5 training weeks)
- **Results**:
  - 183 students processed (1 student had insufficient data ≤3 weeks)
  - Training: 1,341 samples (7.3±1.2 weeks per student)
  - Testing: 549 samples (exactly 3 weeks per student)
- **Model-agnostic output**: Clean dictionary format for any ML model
- **Data integrity**: Zero train/test overlap, perfect chronological separation

### 📊 Data Summary

- **Students**: 184 unique students (183 with sufficient data for split)
- **Time range**: Week -2 to Week 8 (11 weeks total)
- **Target variable statistics**:
  - Mean: 1.32 skills mastered per week
  - Range: 0-26 skills mastered
  - Most common: 0-2 skills per week
- **Split characteristics**:
  - Training weeks: -2 to 5 (per student basis)
  - Test weeks: 2 to 8 (last 3 weeks per student)
  - No temporal leakage between train/test

### 🚧 Next Steps (4-6)

#### Step 4: Add learnable model

- Implement AR(p) autoregressive model via linear regression
- Fit on stacked training pairs from all students
- Optimize using MSE (ordinary least squares)

#### Step 5: Evaluate on held-out weeks

- Compute per-student metrics then average:
  - MAE (robust to outliers)
  - RMSE (penalizes large errors)
  - SMAPE (scale-free interpretability)

#### Step 6: Package into ready-to-use function

- Save fitted coefficients
- Create prediction function: given last t data points, return ŷ\_{t+1}

## Usage

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn
```

### Run Steps 1-3

```bash
# Run complete data processing pipeline (Steps 1-3)
python data_processing.py

# Or test the split functionality
python test_data_split.py
```

### Use the split data for modeling

```python
from data_processing import split_chronologically

# Get the chronologically split data
result = split_chronologically()

# Extract components for model training
train_data = result['train_data']
test_data = result['test_data']
target_column = result['target_column']  # 'proficient'

# Use for any machine learning model
X_train = train_data[['week']]  # Can add more features/lags
y_train = train_data[target_column]
X_test = test_data[['week']]
y_test = test_data[target_column]
```

## Files

- `baseline_specification.MD` - Original specification document
- `experiment_design.tex` - Detailed experimental design and results
- `exp-static-flexible-anon-2025-05-29.csv` - Raw experimental data
- `data_processing.py` - **Main implementation script (Steps 1-3)**
- `test_data_split.py` - Test script demonstrating split functionality
- `data_tidied.csv` - Processed, cleaned dataset (generated)
- `README.md` - This documentation

## Architecture

The implementation follows the "minimal-input baseline" approach:

- **Input**: Only past weekly performance per student (no goals, demographics, etc.)
- **Objective**: Exact prediction accuracy via MSE optimization
- **Model**: Simple AR(p) autoregressive linear regression
- **Evaluation**: Standard time-series prediction metrics

### Data Processing Pipeline

1. **Load & Tidy**: Clean raw CSV → standardized format
2. **Chronological Split**: Per-student train/test split preserving temporal order
3. **Model-Agnostic Output**: Ready for any ML algorithm

This provides a clean baseline for comparison with more sophisticated goal-setting algorithms.

### Key Features

- ✅ **No temporal leakage**: Strict chronological separation
- ✅ **Per-student splitting**: Respects individual student timelines
- ✅ **Intelligent K selection**: Data-driven test set size
- ✅ **Model agnostic**: Works with any ML framework
- ✅ **Data integrity checks**: Automated verification
- ✅ **Comprehensive documentation**: Example usage included
