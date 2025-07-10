# Comprehensive Evaluation Analysis Tools

This directory contains tools for comprehensive evaluation and analysis of time series prediction models, separated into prediction generation and analysis phases for efficiency.

## Overview

The analysis pipeline is split into two main components:

1. **Prediction Generation** (`comprehensive_evaluation_with_saved_predictions.py`)
   - Runs model evaluation and saves all predictions to disk
   - Enables faster iteration on analysis without re-running expensive model training

2. **Analysis & Visualization** (`comprehensive_evaluation_analysis.py`)
   - Loads saved predictions and performs comprehensive analysis
   - Generates plots, statistical tests, and summary reports

## Quick Start

### Option 1: Run Complete Pipeline
```bash
# Run both evaluation and analysis in one command
python run_analysis_demo.py --target-type minutes_per_week --window-size 8
```

### Option 2: Run Separately
```bash
# Step 1: Generate predictions (slow)
python comprehensive_evaluation_with_saved_predictions.py

# Step 2: Analyze predictions (fast)
python comprehensive_evaluation_analysis.py --results-dir evaluation_outputs/minutes_per_week_window8
```

### Option 3: Just Analysis (if you have existing results)
```bash
python run_analysis_demo.py --skip-evaluation --results-dir evaluation_outputs/minutes_per_week_window8
```

## Features

### Prediction Generation
- **Model Evaluation**: Tests multiple algorithms (baselines, classical ML, neural networks)
- **Cross-Validation**: Proper time series cross-validation with configurable folds
- **Prediction Saving**: Saves individual fold predictions and metadata
- **Schema Support**: Uses the schema-based framework for flexible data handling

### Analysis & Visualization

#### Statistical Analysis
- **Performance Metrics**: MAE, RMSE, R², with confidence intervals
- **Significance Testing**: Pairwise statistical tests between models
- **Bootstrap Confidence Intervals**: Robust uncertainty quantification
- **Effect Size Analysis**: Cohen's d for practical significance

#### Visualizations
- **Predicted vs Actual Plots**: Scatter plots for top models
- **Error Distributions**: Histograms of prediction errors
- **Residual Analysis**: Residual plots for model diagnostics
- **Model Comparison**: Bar charts comparing performance metrics
- **Category Analysis**: Performance grouped by model type

#### Reports
- **Summary Report**: Comprehensive markdown report with key findings
- **CSV Exports**: Detailed performance metrics and statistical test results
- **Recommendations**: Automated model selection guidance

## Command Line Options

### Evaluation Script
```bash
python comprehensive_evaluation_with_saved_predictions.py
```

### Analysis Script
```bash
python comprehensive_evaluation_analysis.py --help
```

Options:
- `--results-dir`: Directory containing saved predictions (required)
- `--output-dir`: Output directory for analysis results (optional)
- `--bootstrap-samples`: Number of bootstrap samples (default: 1000)
- `--confidence-level`: Confidence level for intervals (default: 0.95)

### Demo Script
```bash
python run_analysis_demo.py --help
```

Options:
- `--target-type`: Target variable (minutes_per_week or avg_proficiency)
- `--window-size`: Window size for sequences (default: 8)
- `--skip-evaluation`: Skip evaluation step and use existing results
- `--results-dir`: Directory with existing results (if skipping evaluation)

## Output Structure

When you run the analysis, it creates the following structure:

```
evaluation_outputs/
└── minutes_per_week_window8/
    ├── evaluation_config.json              # Evaluation parameters
    ├── overall_results.json                # Summary of all models
    ├── model_name_1/                       # Individual model results
    │   ├── fold_0_predictions.json         # Fold predictions
    │   ├── fold_1_predictions.json
    │   └── summary.json                     # Model summary
    └── analysis/                           # Analysis outputs
        ├── analysis_summary_report_*.md    # Main report
        ├── model_performance_summary_*.csv # Performance metrics
        ├── significance_testing_*.csv      # Statistical tests
        ├── bootstrap_confidence_intervals_*.csv
        ├── predicted_vs_actual_*.png       # Scatter plots
        ├── error_distributions_*.png       # Error histograms
        ├── residual_plots_*.png            # Residual analysis
        ├── model_comparison_*.png          # Performance comparison
        └── performance_by_category_*.png   # Category analysis
```

## Analysis Components

### 1. Performance Summary
- Overall performance metrics for each model
- Cross-validation statistics (mean ± std)
- Model rankings by different metrics
- Number of predictions and folds

### 2. Statistical Significance Testing
- Pairwise comparisons between all models
- Multiple test types (paired t-test, independent t-test, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Significance thresholds and interpretation

### 3. Bootstrap Analysis
- Non-parametric confidence intervals
- Robust uncertainty quantification
- Configurable confidence levels
- Bootstrap distribution analysis

### 4. Visualization Suite
- **Predicted vs Actual**: Scatter plots with perfect prediction lines
- **Error Distributions**: Histograms showing error patterns
- **Residual Plots**: Diagnostic plots for model assumptions
- **Performance Comparison**: Bar charts with error bars
- **Category Analysis**: Grouped performance by model type

### 5. Summary Report
- Automated markdown report generation
- Key findings and recommendations
- Statistical test summaries
- Model performance rankings
- File inventory and locations

## Extending the Analysis

### Adding New Metrics
```python
# In comprehensive_evaluation_analysis.py
def create_performance_summary(self):
    # Add new metrics here
    new_metric = calculate_new_metric(y_true, y_pred)
    summary_data.append({
        'new_metric': new_metric,
        # ... existing metrics
    })
```

### Adding New Plots
```python
# In comprehensive_evaluation_analysis.py
def create_new_plot(self, output_dir, timestamp):
    # Your plotting code here
    plt.savefig(output_dir / f'new_plot_{timestamp}.png')
```

### Custom Analysis
```python
from comprehensive_evaluation_analysis import PredictionAnalyzer

# Load your results
analyzer = PredictionAnalyzer('path/to/results')

# Access the prediction DataFrame
df = analyzer.prediction_df

# Your custom analysis
custom_analysis = analyze_predictions(df)
```

## Best Practices

1. **Separate Evaluation from Analysis**: Always save predictions first, then analyze
2. **Use Bootstrap**: For reliable confidence intervals and significance testing
3. **Check Assumptions**: Review residual plots and error distributions
4. **Compare Categories**: Look at performance by model type, not just individual models
5. **Document Results**: The summary report captures key findings automatically

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check that evaluation completed successfully
3. **Empty Results**: Verify that models generated predictions
4. **Memory Issues**: Reduce bootstrap samples for large datasets

### Performance Tips

1. **Skip Re-evaluation**: Use `--skip-evaluation` for faster iteration
2. **Subset Analysis**: Analyze only top models for quick insights
3. **Parallel Processing**: Analysis steps can be parallelized for large datasets

## Integration with Existing Code

This analysis framework integrates with the existing task requirements:

- ✅ **Separate prediction and analysis**: Achieved through two-script architecture
- ✅ **Bootstrap significance testing**: Implemented with configurable parameters
- ✅ **Predicted vs actual plots**: Generated automatically for top models
- ✅ **Feature importance**: Framework ready (requires model-specific implementation)
- ✅ **MAE rank aggregation**: Performed by window and architecture categories

## Next Steps

1. **Run the demo**: Start with `python run_analysis_demo.py`
2. **Explore results**: Check the generated analysis folder
3. **Customize analysis**: Modify scripts for your specific needs
4. **Integrate with workflow**: Use as part of your model development pipeline