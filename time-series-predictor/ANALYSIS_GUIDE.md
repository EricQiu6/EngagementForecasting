# Comprehensive Evaluation Analysis Guide

## Overview

The `comprehensive_evaluation_analysis.py` script analyzes saved predictions from model evaluation runs and provides:

1. **Top 5 Features Analysis** (automatically included)
2. Statistical significance testing
3. Predicted vs actual plots
4. Error analysis and model comparison
5. Bootstrap confidence intervals

## 🎯 Key Feature: Top 5 Features Analysis

The script **automatically analyzes the top 5 features** for each model. This includes:

- **Feature Rankings**: Top 5 most important features for each model
- **Cross-Model Comparison**: Which features are consistently important across models
- **Feature Importance Scores**: Numerical importance values for each feature
- **CSV Export**: Results saved as `top_features_by_model_{timestamp}.csv`

## Prerequisites

1. **Python Environment**: Activate the virtual environment
```bash
source time-series-predictor/venv_analysis/bin/activate
```

2. **Required Packages** (already installed):
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scipy
   - scikit-learn

## Usage

### Basic Usage

```bash
# Activate environment
source time-series-predictor/venv_analysis/bin/activate

# Run analysis (basic)
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir /path/to/your/evaluation/results

# Run analysis with custom output directory
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir /path/to/your/evaluation/results \
    --output-dir /path/to/analysis/output
```

### Advanced Usage

```bash
# Run with custom bootstrap settings
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir /path/to/your/evaluation/results \
    --bootstrap-samples 2000 \
    --confidence-level 0.99
```

### Example with Real Paths

```bash
# Example if you have results in time-series-predictor/results/
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir time-series-predictor/results/comprehensive_evaluation_2024

# Example with custom output
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir time-series-predictor/results/comprehensive_evaluation_2024 \
    --output-dir time-series-predictor/analysis_output
```

## Expected Results Directory Structure

Your `--results-dir` should contain:

```
results_directory/
├── evaluation_config.json          # (optional) Evaluation configuration
├── overall_results.json            # (optional) Overall results summary
├── model1_name/                    # Individual model directories
│   ├── fold_0_predictions.json    # Predictions for fold 0
│   ├── fold_1_predictions.json    # Predictions for fold 1
│   ├── fold_2_predictions.json    # Predictions for fold 2
│   └── summary.json               # Model summary (contains feature importance)
├── model2_name/
│   ├── fold_0_predictions.json
│   ├── fold_1_predictions.json
│   └── summary.json
└── ...
```

## 🏆 Top 5 Features Analysis Output

The script automatically generates:

### 1. CSV Files
- `top_features_by_model_{timestamp}.csv` - **Main output** with top 5 features per model
- `model_performance_summary_{timestamp}.csv` - Performance metrics
- `significance_testing_{timestamp}.csv` - Statistical significance tests

### 2. Visualizations
- `predicted_vs_actual_{timestamp}.png` - Scatter plots
- `error_distributions_{timestamp}.png` - Error histograms
- `model_comparison_{timestamp}.png` - Performance comparison charts

### 3. Summary Report
- `analysis_summary_report_{timestamp}.md` - Comprehensive markdown report

## Top 5 Features CSV Format

The main output file `top_features_by_model_{timestamp}.csv` contains:

| Column | Description |
|--------|-------------|
| model | Model name |
| rank | Feature rank (1-5) |
| feature | Feature name |
| importance | Feature importance score |

Example:
```csv
model,rank,feature,importance
random_forest,1,student_learning_rate,0.245
random_forest,2,avg_difficulty,0.189
random_forest,3,current_minutes_per_week,0.156
random_forest,4,lag_1_minutes,0.142
random_forest,5,student_ability,0.098
xgboost,1,student_ability,0.267
xgboost,2,student_learning_rate,0.234
...
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--results-dir` | **Required**. Directory with saved prediction results | - |
| `--output-dir` | Output directory for analysis results | `results_dir/analysis` |
| `--bootstrap-samples` | Number of bootstrap samples for confidence intervals | 1000 |
| `--confidence-level` | Confidence level for bootstrap intervals | 0.95 |

## Feature Importance Data Requirements

For the **Top 5 Features Analysis** to work, your model evaluation must save feature importance data in one of these formats:

### Option 1: In Model Summary
```json
{
  "feature_importance": {
    "student_learning_rate": 0.245,
    "avg_difficulty": 0.189,
    "current_minutes_per_week": 0.156,
    ...
  }
}
```

### Option 2: In Overall Results
```json
{
  "model_results": {
    "random_forest": {
      "category": "ensemble",
      "feature_importance": {
        "student_learning_rate": 0.245,
        ...
      }
    }
  }
}
```

## Troubleshooting

### 1. "No feature importance data found"
- Check that your model evaluation saved feature importance data
- Verify the JSON format matches the expected structure
- Some models (like neural networks) may not have feature importance

### 2. "Results directory not found"
- Verify the path to your results directory is correct
- Use absolute paths if relative paths don't work
- Check directory permissions

### 3. Missing dependencies
```bash
# Reactivate environment and install packages
source time-series-predictor/venv_analysis/bin/activate
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Example Complete Workflow

```bash
# 1. Navigate to project directory
cd /path/to/your/project

# 2. Activate environment
source time-series-predictor/venv_analysis/bin/activate

# 3. Run analysis
python3 time-series-predictor/comprehensive_evaluation_analysis.py \
    --results-dir time-series-predictor/results/my_evaluation_results \
    --output-dir time-series-predictor/analysis_output \
    --bootstrap-samples 1000

# 4. View results
ls time-series-predictor/analysis_output/
cat time-series-predictor/analysis_output/top_features_by_model_*.csv
```

## Key Features Highlighted

The analysis automatically focuses on these key student modeling features:

- **student_learning_rate** - Student's learning progression rate
- **student_ability** - Current student proficiency level  
- **avg_difficulty** - Average difficulty of practice content
- **Gap features** - Temporal discontinuity indicators
- **Lag features** - Historical behavior patterns

## Output Interpretation

### Top 5 Features Results
1. **Higher importance scores** = more influential features
2. **Consistent top features** across models = robust predictors
3. **Model-specific patterns** = different algorithms prefer different features
4. **Feature categories** help understand what drives predictions

### Performance Metrics
- **MAE (Mean Absolute Error)** - Lower is better
- **RMSE (Root Mean Square Error)** - Lower is better  
- **R² Score** - Higher is better (closer to 1.0)

---

**Note**: The script automatically analyzes the top 5 features for each model - no additional configuration needed!