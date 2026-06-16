# Statistics Workspace

This directory contains the statistical post-processing workflow and result package for the WQI5 surrogate-regression experiments.

## Results

- Main report: [`outputs/statistical_analysis_report.md`](outputs/statistical_analysis_report.md)
- Method notes: [`../docs/statistical-analysis.md`](../docs/statistical-analysis.md)
- Figures: [`outputs/figures/`](outputs/figures/)

The result files used for manuscript updates are:

- [`outputs/complete_input_performance.csv`](outputs/complete_input_performance.csv)
- [`outputs/missing_indicator_robustness.csv`](outputs/missing_indicator_robustness.csv)
- [`outputs/cpu_only_timing.csv`](outputs/cpu_only_timing.csv)
- [`outputs/stress107_summary.csv`](outputs/stress107_summary.csv)
- [`outputs/feature_score_correlations.csv`](outputs/feature_score_correlations.csv)
- [`outputs/bootstrap_ci.csv`](outputs/bootstrap_ci.csv)
- [`outputs/paired_error_tests.csv`](outputs/paired_error_tests.csv)
- [`outputs/sample_size_sensitivity.csv`](outputs/sample_size_sensitivity.csv)
- [`outputs/sample_size_metrics_by_fold.csv`](outputs/sample_size_metrics_by_fold.csv)

The report includes summary metrics, confidence intervals, complete-input GPU
paired MAE tests, descriptive feature-score correlations, Stress107 summaries,
CPU-only timing, and rendered residual figures.

## Reproduce Results

Prepare the manuscript-facing outputs from the organized local result bundle:

```bash
python scripts/prepare_statistics_outputs.py \
  --bundle-dir results/manuscript_package \
  --complete-input-gpu-dir results/complete_input_gpu \
  --output-dir statistics/outputs
```

Create residual figures from prediction rows:

```bash
python scripts/generate_residual_plots.py \
  --input-csv results/missing_indicator_robustness/predictions/predictions_long.csv \
  --source external_10714 \
  --experiment full_reference \
  --missing-set complete \
  --output-dir statistics/outputs/figures
```

The statistical scripts post-process recorded experiment outputs and committed
datasets. They do not retrain model artifacts. The current statistics workflow
does not depend on local Excel workbooks; it reads the organized result bundle
and writes the manuscript tables.

Prepare the manuscript-facing sample-size outputs from the consolidated local
sample-size metrics:

```bash
python scripts/prepare_sample_size_outputs.py \
  --metrics-dir results/sample_size_experiments/metrics \
  --output-dir statistics/outputs
```
