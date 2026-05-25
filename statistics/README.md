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

## Public Outputs

The main result package is [`outputs/statistical_analysis_report.md`](outputs/statistical_analysis_report.md). It includes the summary tables, confidence intervals, pairwise tests, residual diagnostics, WQI-band error summaries, and rendered figures.

The method definitions are documented in [`../docs/statistical-analysis.md`](../docs/statistical-analysis.md). Generated CSV files under `outputs/` provide the table-level data used by the report.

Tracked generated outputs:

- `outputs/statistical_analysis_report.md`
- `outputs/metric_ci_by_runs.csv`
- `outputs/paired_tests_by_runs.csv`
- `outputs/test_prediction_metrics.csv`
- `outputs/test_bootstrap_ci.csv`
- `outputs/test_paired_error_tests.csv`
- `outputs/residual_diagnostics.csv`
- `outputs/sample_size_stability.csv`
- `outputs/dataset_distribution_robustness.csv`
- `outputs/error_by_wqi_band.csv`
- `outputs/figures/*.png`

## Notes

- The `10714` hold-out rows are the tail rows of `data/dataV1.csv` after the first `50000` rows contained in `data/dataV1_50000.csv`.
- The hold-out source rows are reproducible from committed CSV files.
- Hold-out reconstruction and prediction-check workflows are provided by the committed scripts in this repository.
