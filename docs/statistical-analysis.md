# Statistical Analysis

This document separates descriptive feature-score analysis from predictive model reliability analysis.

## Feature-Score Correlation

Pearson and Spearman correlation coefficients can be computed between each raw water-quality indicator and the calculated WQI5 `Score`:

- `DO` vs `Score`
- `BOD` vs `Score`
- `NH3N` vs `Score`
- `EC` vs `Score`
- `SS` vs `Score`

This analysis is descriptive because `Score` is constructed from the same five indicators. It summarizes feature-index relationships within the processed dataset rather than temporal forecasting behavior.

## Model Reliability Analysis

Model reliability is evaluated on held-out or otherwise unseen prediction outputs using metrics such as:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Mean Predictive Accuracy (MPA)`
- `Residual Mean`
- `Residual Std.`

Residuals are defined as:

```text
residual_i = y_i - ŷ_i
```

where `y_i` is the reference WQI5 score and `ŷ_i` is the model-estimated WQI5 score.

## Statistical Coverage

The current statistics workflow covers three main analysis families:

### Confidence Intervals

Two confidence-interval estimates are reported because the available data have two levels of granularity:

- run-level `95%` intervals summarize repeated subset-benchmark metric logs
- row-level bootstrap `95%` intervals summarize the `10714` hold-out prediction records

Covered metrics include:

- `R²`
- `MAE`
- `RMSE`
- `Mean Predictive Accuracy (MPA)`

The hold-out bootstrap intervals cover the continuous regression metrics listed above.

### Interval Definitions

Run-level intervals in `metric_ci_by_runs.csv` summarize repeated benchmark runs. For each `sample_size`, `model`, and metric, the interval is computed from the run-level metric values:

```text
mean +/- t_(0.975, n-1) * sample_std / sqrt(n)
```

These intervals describe variation across repeated runs in the archived benchmark logs. When only one run is available, the point estimate is reported and the interval bounds are left empty.

Hold-out metric intervals in `test_bootstrap_ci.csv` summarize prediction performance on the `10714` hold-out rows. For each model, rows are resampled with replacement, the metric is recomputed on each bootstrap sample, and the 2.5th and 97.5th percentiles of the bootstrap metrics are reported.

Paired-difference intervals in `paired_tests_by_runs.csv` and `test_paired_error_tests.csv` summarize model-to-model differences. For hold-out error comparisons, the paired difference is:

```text
diff_i = |y_i - yhat_A_i| - |y_i - yhat_B_i|
```

The paired-difference interval is obtained by bootstrapping the paired differences and taking the 2.5th and 97.5th percentiles of the bootstrap mean differences. Intervals that include zero indicate that the observed average difference is small relative to its bootstrap uncertainty.

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
- WQI-band error analysis on the `10714` hold-out rows
- residual diagnostics including:
  - residual mean
  - residual standard deviation
  - skewness
  - kurtosis
  - KS-based normality check

## Repository Workflow

The repository currently contains a statistics workspace under [`statistics/`](../statistics/README.md):

- `statistics/statistical_analysis_from_xlsx.py`
  - post-processes archived experiment records and committed CSV datasets
  - writes generated tables into `statistics/outputs/`
  - does not retrain model artifacts
- `scripts/reproduce_holdout_10714.py`
  - reconstructs the `10714` hold-out rows from `data/dataV1.csv` and `data/dataV1_50000.csv`
  - validates the hold-out source rows against the Excel `10714筆測試` sheet
- `scripts/generate_residual_plots.py`
  - reads `statistics/outputs/test_predictions_long.csv`
  - generates per-model residual figures and overview panels under `statistics/outputs/figures/`
- `statistics/generate_statistical_report.py`
  - reads generated CSV tables
  - writes `statistics/outputs/statistical_analysis_report.md`

The repository tracks the generated statistical summary tables, markdown report, and PNG residual figures used for public inspection. Large or local-only generated artifacts, including the workbook export, row-level prediction table, local Excel source workbook, and hold-out reproduction row dumps, are regenerated locally.

## Generated Result Tables

Running `python statistics/statistical_analysis_from_xlsx.py` produces structured statistical tables under `statistics/outputs/` from archived experiment records.

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
  - paired absolute-error tests on the `10714`-sample inference evaluation set
- `error_by_wqi_band.csv`
  - hold-out regression errors stratified by actual WQI band
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
- `figures/residual_diagnostics_lightgbm.png`
- `figures/residual_xgboost.png`
- `figures/residual_diagnostics_xgboost.png`
- `figures/residual_rf.png`
- `figures/residual_diagnostics_rf.png`
- `figures/residual_svm.png`
- `figures/residual_diagnostics_svm.png`
- `figures/residual_mpr.png`
- `figures/residual_diagnostics_mpr.png`
- `figures/residual_lr.png`
- `figures/residual_diagnostics_lr.png`

## Residual Figures

### Overview

![Residual overview](../statistics/outputs/figures/residual_overview.png)

![Residual Q-Q overview](../statistics/outputs/figures/residual_qq_overview.png)

### Model Diagnostics

![LightGBM residual diagnostics](../statistics/outputs/figures/residual_diagnostics_lightgbm.png)

![XGBoost residual diagnostics](../statistics/outputs/figures/residual_diagnostics_xgboost.png)

![RF residual diagnostics](../statistics/outputs/figures/residual_diagnostics_rf.png)

![SVM residual diagnostics](../statistics/outputs/figures/residual_diagnostics_svm.png)

![MPR residual diagnostics](../statistics/outputs/figures/residual_diagnostics_mpr.png)

![LR residual diagnostics](../statistics/outputs/figures/residual_diagnostics_lr.png)

### Residual Histograms

![LightGBM residual histogram](../statistics/outputs/figures/residual_lightgbm.png)

![XGBoost residual histogram](../statistics/outputs/figures/residual_xgboost.png)

![RF residual histogram](../statistics/outputs/figures/residual_rf.png)

![SVM residual histogram](../statistics/outputs/figures/residual_svm.png)

![MPR residual histogram](../statistics/outputs/figures/residual_mpr.png)

![LR residual histogram](../statistics/outputs/figures/residual_lr.png)

## Reporting Conventions

`Mean Predictive Accuracy (MPA)` is defined as:

```text
MPA (%) = mean_i [(1 - |y_i - ŷ_i| / y_i) * 100]
```

For positive WQI5 reference scores, this is equivalent to `100% - MAPE(%)`. It is retained only as an interpretable companion to `R²`, `MAE`, and `RMSE`.

WQI-band summaries use the backend category configuration: `Excellent`, `Good`, `Fair`, `Poor`, `Bad`, and `Terrible`.

Very small p-values may underflow to zero in floating-point calculations. Generated public tables report those values as `<1e-300` rather than `0`.
