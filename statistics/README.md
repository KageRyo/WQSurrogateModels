# Statistics Workspace

This directory mixes three different kinds of assets:

## Track In Git

- `statistical_analysis_from_xlsx.py`
  - statistical post-processing script
  - reads the local archived experiment workbook and `data/*.csv`
  - writes derived tables into `statistics/outputs/`
  - does not retrain model artifacts
- `generate_statistical_report.py`
  - creates the public markdown summary from generated CSV outputs
  - reports regression metrics, confidence intervals, p-values, residual diagnostics, and WQI-band error summaries
- `outputs/*.csv`
  - selected generated CSV tables used by the public statistical summary
- `outputs/statistical_analysis_report.md`
  - generated public statistical summary
- `outputs/figures/*.png`
  - generated residual figures for public inspection
- `../scripts/reproduce_holdout_10714.py`
  - reconstructs the `10714` hold-out split as `dataV1.csv - dataV1_50000.csv`
  - verifies that the hold-out `Score` matches the Excel `10714筆測試` sheet
  - reproduces hold-out predictions for the saved `50000` artifacts that exist in `models/`
- `../scripts/generate_residual_plots.py`
  - reads generated hold-out prediction tables
  - writes residual figures into `statistics/outputs/figures/`
  - supports visual inspection of residual distribution, residual spread, and Q-Q behavior
- `README.md`
  - this inventory and tracking policy

## Do Not Track In Git

- `整理.xlsx`
  - local archived experiment workbook used to regenerate the statistical outputs
  - excluded from source control because it is an experiment record, not repository source code
- untracked generated outputs
  - `outputs/statistical_analysis_outputs.xlsx`
  - `outputs/test_predictions_long.csv`
  - `outputs/long_metric_logs.csv`
  - `outputs/holdout_reproduction/*.csv`
  - superseded generated tables such as `outputs/category_level_metrics.csv`

These files are useful for local inspection and result packaging, and are regenerated locally.

## Reproduction Flow

Generate the statistical analysis package from archived experiment records:

```bash
python statistics/statistical_analysis_from_xlsx.py
```

Generate the public markdown summary:

```bash
python statistics/generate_statistical_report.py
```

Reconstruct and validate the `10714` hold-out set:

```bash
python scripts/reproduce_holdout_10714.py
```

Generate residual figures:

```bash
python scripts/generate_residual_plots.py
```

This writes per-model residual diagnostics and overview panels into `statistics/outputs/figures/`.

## What The Statistics Outputs Cover

The generated tables and figures are intended to support questions about:

- confidence intervals
  - run-level `95%` intervals for repeated subset-benchmark metric logs
  - row-level bootstrap `95%` intervals for the `10714` hold-out prediction records
- significance testing
  - paired Wilcoxon signed-rank tests
  - Holm-adjusted p-values
  - paired bootstrap difference intervals
- residual statistics
  - residual mean
  - residual standard deviation
  - residual skewness
  - residual kurtosis
  - KS-based normality statistics
- robustness analysis
  - sample-size sensitivity
  - subset-vs-full distribution shift
  - WQI-band error summaries

The main analysis treats WQI5 score estimation as a regression task. WQI-band summaries use the backend category configuration used by WaterMirror: `Excellent`, `Good`, `Fair`, `Poor`, `Bad`, and `Terrible`.

## Interval Interpretation

- `metric_ci_by_runs.csv`: run-level intervals computed as `mean +/- t_(0.975, n-1) * sample_std / sqrt(n)` from repeated benchmark metric logs.
- `test_bootstrap_ci.csv`: row-level bootstrap intervals computed by resampling the `10714` hold-out prediction rows and recomputing `R²`, `MAE`, `RMSE`, and `MPA`.
- `paired_tests_by_runs.csv`: paired model-difference intervals computed from repeated-run metric differences.
- `test_paired_error_tests.csv`: paired model-difference intervals computed from absolute-error differences on the `10714`-sample inference evaluation set, `|y_i - yhat_A_i| - |y_i - yhat_B_i|`.

Intervals describe uncertainty in the reported estimate under the available archived runs or hold-out rows. Paired-difference intervals that include zero indicate that the average model-to-model difference is small relative to its bootstrap uncertainty.

Relevant generated files:

- `outputs/metric_ci_by_runs.csv`
- `outputs/paired_tests_by_runs.csv`
- `outputs/test_bootstrap_ci.csv`
- `outputs/test_paired_error_tests.csv`
- `outputs/residual_diagnostics.csv`
- `outputs/sample_size_stability.csv`
- `outputs/dataset_distribution_robustness.csv`
- `outputs/error_by_wqi_band.csv`
- `outputs/statistical_analysis_report.md`
- `outputs/figures/*.png`
- `outputs/figures/residual_overview.png`
- `outputs/figures/residual_qq_overview.png`

The rendered report includes the tracked figures directly:

![Residual overview](outputs/figures/residual_overview.png)

![Residual Q-Q overview](outputs/figures/residual_qq_overview.png)

## Notes

- The `10714` hold-out rows are the tail rows of `data/dataV1.csv` after the first `50000` rows contained in `data/dataV1_50000.csv`.
- The hold-out source rows are reproducible from committed CSV files.
- Hold-out reconstruction and prediction-check workflows are provided by the committed scripts in this repository.
