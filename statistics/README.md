# Statistics Workspace

This directory contains the statistical post-processing workflow and result package for the WQI5 surrogate-regression experiments.

## Results

- Main report: [`outputs/statistical_analysis_report.md`](outputs/statistical_analysis_report.md)
- Method notes: [`../docs/statistical-analysis.md`](../docs/statistical-analysis.md)
- Figures: [`outputs/figures/`](outputs/figures/)

The revision result tables used for manuscript updates are:

- [`outputs/revision_table6_complete_input_performance.csv`](outputs/revision_table6_complete_input_performance.csv)
- [`outputs/revision_table7_missing_indicator_robustness.csv`](outputs/revision_table7_missing_indicator_robustness.csv)
- [`outputs/revision_table8_cpu_only_timing.csv`](outputs/revision_table8_cpu_only_timing.csv)
- [`outputs/revision_table9_stress107_summary.csv`](outputs/revision_table9_stress107_summary.csv)
- [`outputs/revision_bootstrap_ci.csv`](outputs/revision_bootstrap_ci.csv)
- [`outputs/revision_paired_error_tests.csv`](outputs/revision_paired_error_tests.csv)

The report includes revision summary metrics, confidence intervals, pairwise
error tests, Stress107 summaries, CPU-only timing, and rendered residual
figures.

## Reproduce Results

Prepare the revision tables from the frozen local result bundle:

```bash
python scripts/prepare_revision_outputs.py \
  --bundle-dir results_20260614_stress \
  --output-dir statistics/outputs
```

Create residual figures from revision prediction rows:

```bash
python scripts/generate_residual_plots.py \
  --input-csv results_20260614_stress/raw/results_missing_indicator_robustness_20260613_gpu_v2/predictions/predictions_long.csv \
  --source external_10714 \
  --experiment full_reference \
  --missing-set complete \
  --output-dir statistics/outputs/figures
```

Archived workbook-based post-processing remains available for older benchmark
records:

```bash
python statistics/statistical_analysis_from_xlsx.py
python statistics/generate_statistical_report.py
```

Reconstruct the `10714`-record inference evaluation set:

```bash
python scripts/reproduce_inference_10714.py
```

The statistical scripts post-process recorded experiment outputs and committed datasets. They do not retrain model artifacts.
