# Time Series Predictor - Project Structure

This document explains the reorganized structure of the time-series-predictor project.

## Directory Structure

```
time-series-predictor/
├── src/                          # Main source code
│   └── framework/               # Production framework (formerly framework_v2)
│       ├── core/               # Core abstractions
│       │   ├── base.py        # Base classes (TimeSeriesModel, etc.)
│       │   └── data.py        # Data pipeline (StudentTimeSeriesDataset)
│       ├── adapters/          # Model adapters
│       │   ├── sklearn_adapter.py  # SKLearn model wrapper
│       │   └── pytorch_adapter.py  # PyTorch model wrapper
│       ├── models/            # Model implementations
│       │   ├── neural_nets.py      # Neural network models
│       │   ├── baselines.py        # Baseline models
│       │   ├── student_ability_model.py  # Student-specific models
│       │   ├── zero_inflated_model.py    # Zero-inflated models
│       │   └── DLinear.py          # DLinear implementation
│       └── utils/             # Utilities
│           └── device.py      # GPU device management
│
├── legacy/                      # Archived legacy framework
│   └── model/                  # Original framework (deprecated)
│       ├── framework.py       # Old framework implementation
│       ├── models.py          # Old model implementations
│       └── train_evaluate.py  # Old training script
│
├── experiments/                 # Analysis and experiments
│   ├── demos/                  # Demonstration scripts
│   │   ├── demo_framework_v2.py        # Main framework demo
│   │   └── demo_student_ability_model.py  # Student model demo
│   ├── analysis/               # Analysis scripts
│   │   ├── analyze_student_week_data.py     # Initial data exploration
│   │   ├── student_week_predictive_modeling.py  # Various modeling approaches
│   │   ├── analyze_modeling_results.py      # Model results analysis
│   │   ├── diagnose_model_issues.py         # Performance diagnostics
│   │   ├── prediction_quality_analysis.py   # Prediction quality analysis
│   │   ├── final_modeling_analysis.py       # Final comprehensive analysis
│   │   └── test_improved_models.py          # Test improved models
│   └── outputs/                # Generated outputs
│       ├── *.png              # Plots and visualizations
│       ├── *.csv              # Result data files
│       └── *.txt              # Reports and summaries
│
├── docs/                        # Documentation
│   ├── STUDENT_ABILITY_MODEL_README.md  # Model documentation
│   └── migrate_to_v2.py       # Migration guide
│
├── data/                        # Data directory
│   ├── data_processing.py     # Data processing pipeline
│   ├── tidy_data.py          # Data cleaning
│   └── test/                 # Data tests and notebooks
│
├── tests/                       # Unit tests (to be populated)
│
├── README.md                    # Main project README
├── baseline_specification.MD    # Original baseline specification
└── experiment_design.tex        # Experiment design document
```

## Key Changes from Previous Structure

1. **Framework Consolidation**: The production framework (formerly `framework_v2`) is now in `src/framework/`
2. **Legacy Preservation**: The old framework is archived in `legacy/model/`
3. **Experiment Organization**: All analysis scripts and outputs are organized under `experiments/`
4. **Clean Root**: The root directory now only contains essential files and directories

## Using the New Structure

### Running Demos

```bash
cd experiments/demos
python demo_framework_v2.py
python demo_student_ability_model.py
```

### Running Analysis

```bash
cd experiments/analysis
python analyze_student_week_data.py
python final_modeling_analysis.py
```

### Importing the Framework

In your Python scripts:

```python
from src.framework import (
    StudentTimeSeriesDataset,
    SKLearnAdapter,
    PyTorchAdapter,
    CrossValidator
)
```

## Migration Notes

- All import paths have been updated to use `src.framework` instead of `framework_v2`
- The demos and analysis scripts now use relative imports with proper path adjustments
- All generated outputs (plots, CSVs, reports) are now in `experiments/outputs/`
