# Enhanced Feature Importance Analysis

## Overview

The `comprehensive_evaluation_analysis.py` script has been enhanced with comprehensive feature importance analysis capabilities that focus on identifying the most important features for each model, with special attention to three key student modeling features.

## Key Features

### 🎯 Top 5 Features Analysis
- **Identifies top 5 most important features for each model**
- **Generates horizontal bar charts** showing feature importance rankings
- **Highlights key student modeling features** in red for easy identification
- **Saves detailed CSV summaries** with feature rankings and importance scores

### 🔍 Focus on Key Student Modeling Features
The analysis specifically highlights and analyzes these three critical features:

1. **`student_learning_rate`** - Student's learning progression rate
2. **`student_ability`** - Current student proficiency level  
3. **`avg_difficulty`** - Average difficulty of practice content

### 📊 Comprehensive Visualizations

#### 1. Top Features by Model (`top_features_by_model_*.png`)
- **Subplot grid** showing top 5 features for each model
- **Color-coded bars**: Red for key features, blue for others
- **Value labels** showing exact importance scores
- **Horizontal bar charts** for easy feature name reading

#### 2. Feature Importance Heatmap (`feature_importance_heatmap_*.png`)
- **Cross-model comparison** of feature importance
- **Top 15 features** across all models
- **Color-coded labels**: Red for key student modeling features
- **Annotated values** showing exact importance scores

#### 3. Key Features Ranking (`key_features_ranking_*.png`)
- **Line plot** showing how the three key features rank across models
- **Inverted y-axis** (rank 1 at top)
- **Clear comparison** of feature consistency across models

#### 4. Specific Features Importance (`specific_features_importance_*.png`)
- **Dedicated analysis** of the three key features
- **Error bars** showing importance variability
- **Value labels** for precise comparisons

### 📈 Analysis Outputs

#### CSV Files Generated:
- **`top_features_by_model_*.csv`** - Complete ranking of top features by model
- **`feature_consistency_analysis_*.csv`** - Feature statistics across models
- **`specific_features_analysis_*.csv`** - Detailed analysis of key features

#### Key Metrics Calculated:
- **Mean importance** across models
- **Standard deviation** of importance
- **Coefficient of variation** for consistency
- **Model count** (how many models use each feature)
- **Min/max importance** values

## Usage

### Basic Usage
```bash
python comprehensive_evaluation_analysis.py --results-dir /path/to/results
```

### With Custom Output Directory
```bash
python comprehensive_evaluation_analysis.py --results-dir /path/to/results --output-dir /path/to/analysis
```

### Testing the Feature Importance Analysis
```bash
# Activate virtual environment
source venv_analysis/bin/activate

# Run the test
python test_feature_importance_analysis.py
```

## Data Requirements

The analysis expects saved model results with feature importance data in one of these formats:

### Option 1: Model Summary Files
Each model directory should contain a `summary.json` file with:
```json
{
  "model_name": "random_forest",
  "category": "ensemble",
  "feature_importance": {
    "student_learning_rate": 0.145,
    "student_ability": 0.118,
    "avg_difficulty": 0.057,
    "other_feature": 0.032
  }
}
```

### Option 2: Overall Results File
The `overall_results.json` file can contain feature importance:
```json
{
  "model_results": {
    "random_forest": {
      "feature_importance": {
        "student_learning_rate": 0.145,
        "student_ability": 0.118,
        "avg_difficulty": 0.057
      }
    }
  }
}
```

## Example Output

### Top Features Summary
```
  model  rank                feature  importance
  lasso     1   avg_proficiency_lag1    0.414410
  lasso     2          streak_length    0.216692
  lasso     3            day_of_week    0.113713
  lasso     4  minutes_per_week_lag2    0.090835
  lasso     5    total_problems_lag1    0.044017
xgboost     1  minutes_per_week_lag1    0.206528
xgboost     2                week_id    0.122920
xgboost     3        completion_rate    0.091979
xgboost     4 problem_difficulty_std    0.078691
xgboost     5            day_of_week    0.070028
```

### Specific Features Analysis
```
       target_feature         model  mean_importance
student_learning_rate         lasso           0.0157
student_learning_rate random_forest           0.1450
student_learning_rate       xgboost           0.0183
      student_ability         lasso           0.0126
      student_ability random_forest           0.1175
      student_ability       xgboost           0.0568
       avg_difficulty         lasso           0.0105
       avg_difficulty random_forest           0.0568
       avg_difficulty       xgboost           0.0438
```

## Analysis Insights

### 🏆 Model Comparison
- **Random Forest** tends to give higher importance to student modeling features
- **LASSO** shows more sparse feature selection
- **XGBoost** provides balanced feature importance distribution

### 📊 Feature Patterns
- **student_learning_rate** typically ranks highest among the three key features
- **student_ability** shows consistent importance across models
- **avg_difficulty** varies more significantly between model types

### 🔍 Consistency Analysis
- **Coefficient of variation** helps identify which features are consistently important
- **Model count** shows how many models actually use each feature
- **Cross-model ranking** reveals feature stability

## Integration with Existing Analysis

The feature importance analysis is fully integrated into the existing comprehensive analysis pipeline:

1. **Runs automatically** as part of the comprehensive analysis
2. **Generates both visualizations and data files**
3. **Includes results in the summary report**
4. **Maintains compatibility** with existing analysis outputs

## Dependencies

The enhanced analysis requires:
- `pandas` >= 1.0.0
- `numpy` >= 1.18.0
- `matplotlib` >= 3.0.0
- `seaborn` >= 0.11.0
- `scipy` >= 1.4.0
- `scikit-learn` >= 0.22.0

## Error Handling

The analysis gracefully handles:
- **Missing feature importance data** - skips analysis with clear message
- **Inconsistent feature names** - uses string matching for robustness
- **Empty or malformed data** - provides informative error messages
- **Visualization failures** - continues with data analysis even if plots fail

## Future Enhancements

Potential improvements include:
- **Interactive visualizations** using Plotly
- **Feature correlation analysis**
- **Feature importance stability over time**
- **Automated feature selection recommendations**
- **Model-specific feature importance extraction**