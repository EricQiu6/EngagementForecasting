## Engagement Forecasting — End-to-End Pipeline Guide

This document explains how data flow, training/evaluation, and analysis fit together to produce some of the paper artifacts and figures.

### 1) Data preparation (R → CSVs)

- `data-analysis/combined_rolling_window.Rmd`: dataset processing step and creating AFM features

### 2) Modeling framework (Python)

- Core abstractions (`time-series-predictor/src/framework/core/`):
  - `data.py`: sequence construction and dataset utilities (student-week → sliding windows).
  - `schema.py`: feature/target schema definitions.
  - `base.py`: base model interfaces with type hints.
- Models (`time-series-predictor/src/framework/models/`):
  - `baselines.py`: median/mean/no-zeros, naive, averages.
  - `neural_nets.py`. neural network implementations.

### 3) Comprehensive evaluation runs (train once, save everything)

- Entrypoints:
  - `time-series-predictor/comprehensive_evaluation_with_saved_predictions.py` (saves predictions, metrics, summaries per window/fold)
- What a run defines: dataset, target, window size W, feature set, CV (GroupKFold by student), model list.
- Outputs per window directory (under `time-series-predictor/evaluation_outputs*/...`):
  - `evaluation_config.json` (config used)
  - `predictions_*.csv` (per-fold predictions)
  - `performance_summary.csv` (MAE/RMSE per model)
  - Model summaries with raw feature importances as JSON

### 4) Single-window analysis (no retraining)

- `time-series-predictor/comprehensive_evaluation_analysis.py`:
  - Loads saved predictions/summaries from a window directory.
  - Normalizes feature importances on load (sum-to-one across features per model) via internal helpers.

How to run (example with result outputs from running evaluation script in step 3):

```
cd time-series-predictor
python comprehensive_evaluation_analysis.py --results_dir evaluation_outputs/.../rolling_new_minutes_w6_all_standard_all
```

### 5) Cross-window analysis (aggregate across window)

- `time-series-predictor/cross_window_analysis.py`:
  - Reads each window’s `analysis/top_features_by_model_*.csv` and performance summaries.
  - Applies the same sum-to-one normalization on importances at load time.
  - **Produces RQ3's figure 2: feature-importance ranking** (`cross_window_aggregated_feature_ranking_*.png`/CSV) and summary report in the specified output directory.

How to run (example):

```
cd time-series-predictor
python cross_window_analysis.py --base_dir evaluation_outputs_with_features/rolling_new_minutes_* --output_dir cross_window_analysis_results
```

### 6) Additional paper artifacts (tables, JSON, figures)

- `time-series-predictor/paper_artifacts_rq.py` consolidates saved outputs (no re-train) to produce:
  - `artifacts/metrics/rq1_family_tests.json`: pairwise family comparisons (Wilcoxon, 95% CI, Cliff’s δ) per target.
  - `artifacts/metrics/rq1_family_tests_new.json`: family-level MAE summaries per target and window.
  - **combined gives RQ1's table 3: family comparisons**

Run:

```
cd time-series-predictor
python paper_artifacts_rq.py
```

### 7) Goal-based teacher-forcing evaluation (standalone)

- `paper-evaluations/goal_based_autoregressive_evaluation_new_designs_adam.py`: **produces RQ2's fig 1 and table 4:** refined Adams rules (P50/60/70) + XGBoost, with grey ±1 std-dev ribbons and detailed diagnostics.

Example:

```
python paper-evaluations/goal_based_autoregressive_evaluation_new_designs_adam.py \
  --csv data.csv \
  --max_weeks 30 --train_window 5 --save_plots paper-evaluations/rq2_adams.png
```

### 9) Conventions and guardrails

- Splits: GroupKFold by `student_id` to prevent leakage.
- Ordering: Chronological within each student; sequences length W predict next-week target.
- Normalization: Feature importances normalized in analysis only (sum-to-one per model) to enable cross-family comparisons; raw evaluation outputs are untouched.
- average of multiple baselines used for Δ%.

### 10) Typical workflow

1. Run comprehensive evaluations to generate predictions per window.
2. Run single-window analysis to compute normalized feature importance and charts.
3. Run cross-window analysis to aggregate importances and rank features across W.
4. Generate paper artifacts via `paper_artifacts_rq.py` (tables/JSON/figures).
