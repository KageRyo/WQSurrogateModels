# Statistical Analysis

This document separates descriptive feature-score analysis from predictive model reliability analysis.

## Feature-Score Correlation

Pearson and Spearman correlation coefficients can be computed between each raw water-quality indicator and the calculated WQI5 `Score`:

- `DO` vs `Score`
- `BOD` vs `Score`
- `NH3N` vs `Score`
- `EC` vs `Score`
- `SS` vs `Score`

This analysis is descriptive because `Score` is constructed from the same five indicators.

It should therefore be interpreted as:

- a summary of how the raw indicators relate to the constructed WQI5 index
- not causal evidence
- not evidence of temporal prediction ability

## Model Reliability Analysis

Model reliability should be evaluated on held-out or otherwise unseen prediction outputs using metrics such as:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Residual Mean`
- `Residual Std.`
- `Mean Relative Accuracy`
- `Std. of Relative Accuracy`
- `Category Accuracy`
- `Macro-F1`

Residuals are defined as:

```text
residual_i = y_i - ŷ_i
```

where `y_i` is the reference WQI5 score and `ŷ_i` is the model-estimated WQI5 score.

## Statistical Coverage

The current statistics workflow covers three main analysis families:

### Confidence Intervals

- repeated-run `95% t-intervals` for subset-benchmark metrics
- bootstrap `95% confidence intervals` for the `10714` hold-out prediction metrics

Covered metrics include:

- `R²`
- `MAE`
- `RMSE`
- `Category Accuracy`
- `Macro-F1`

### Significance Testing

- paired `Wilcoxon signed-rank tests` for model-to-model comparisons
- `Holm` correction for multiple comparisons
- bootstrap confidence intervals for paired mean differences
- rank-biserial effect size for paired comparisons

This is applied to:

- repeated subset-benchmark validation metrics
- hold-out absolute-error comparisons on the `10714` rows

### Robustness Analysis

- sample-size sensitivity across `100 / 1000 / 5000 / 10000 / 20000 / 50000`
- distribution-shift checks between each subset and the full `60714`-row dataset
- category-level error analysis on the `10714` hold-out rows
- residual diagnostics including:
  - residual mean
  - residual standard deviation
  - skewness
  - kurtosis
  - KS-based normality check

## Repository Workflow

The repository currently contains a statistics workspace under [`statistics/`](../statistics/README.md):

- `statistics/statistical_analysis_from_xlsx.py`
  - analyzes the archived workbook and committed CSV datasets
  - writes generated tables into `statistics/outputs/`
- `scripts/reproduce_holdout_10714.py`
  - reconstructs the `10714` hold-out rows from `data/dataV1.csv` and `data/dataV1_50000.csv`
  - validates the hold-out source rows against the Excel `10714筆測試` sheet
- `scripts/generate_residual_plots.py`
  - reads `statistics/outputs/test_predictions_long.csv`
  - generates per-model residual figures and overview panels under `statistics/outputs/figures/`

Generated outputs under `statistics/outputs/` are derived artifacts and should normally be regenerated locally rather than versioned. If a narrative markdown summary is needed, it should be prepared separately from the public analysis script.

## Generated Result Tables

Running `python statistics/statistical_analysis_from_xlsx.py` produces structured tables under `statistics/outputs/`.

Key outputs:

- `metric_ci_by_runs.csv`
  - repeated-run confidence intervals by `sample_size`, `model`, and metric
- `paired_tests_by_runs.csv`
  - paired significance tests for repeated benchmark runs
- `test_prediction_metrics.csv`
  - point estimates on the `10714` hold-out rows
- `test_bootstrap_ci.csv`
  - bootstrap confidence intervals for hold-out metrics
- `test_paired_error_tests.csv`
  - paired hold-out absolute-error significance tests
- `category_level_metrics.csv`
  - hold-out metrics stratified by actual WQI category
- `residual_diagnostics.csv`
  - residual mean, std, skewness, kurtosis, and KS normality statistics
- `dataset_distribution_robustness.csv`
  - subset-vs-full distribution shift summaries
- `sample_size_stability.csv`
  - monotonic trend summaries across sample sizes
- `statistical_analysis_outputs.xlsx`
  - workbook export containing the same generated tables

The hold-out reconstruction workflow also produces:

- `holdout_reproduction/holdout_10714.csv`
  - reproducible extraction of the `10714` tail rows
- `holdout_reproduction/holdout_prediction_reproduction_comparison.csv`
  - comparison between Excel-recorded predictions and reproduced predictions from saved model artifacts
- `holdout_reproduction/holdout_reproduction_summary.csv`
  - compact reproduction summary by model

Residual plotting workflow:

```bash
python scripts/generate_residual_plots.py
```

This generates:

- `figures/residual_overview.png`
- `figures/residual_qq_overview.png`
- `figures/residual_lightgbm.png`
- `figures/residual_xgboost.png`
- `figures/residual_rf.png`
- `figures/residual_svm.png`
- `figures/residual_mpr.png`
- `figures/residual_lr.png`
