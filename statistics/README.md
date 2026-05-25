# Statistics Workspace

This directory contains the statistical post-processing workflow and generated result package for the WQI5 surrogate-regression experiments.

## Results

- Main report: [`outputs/statistical_analysis_report.md`](outputs/statistical_analysis_report.md)
- Method notes: [`../docs/statistical-analysis.md`](../docs/statistical-analysis.md)
- Figures: [`outputs/figures/`](outputs/figures/)

The report includes summary metrics, confidence intervals, pairwise error tests, residual diagnostics, WQI-band error summaries, and rendered residual figures.

## Regeneration

Generate the statistical tables from the archived experiment records:

```bash
python statistics/statistical_analysis_from_xlsx.py
```

Generate the markdown report:

```bash
python statistics/generate_statistical_report.py
```

Generate residual figures:

```bash
python scripts/generate_residual_plots.py
```

Reconstruct the `10714`-record inference evaluation set:

```bash
python scripts/reproduce_holdout_10714.py
```

The statistical scripts post-process recorded experiment outputs and committed datasets. They do not retrain model artifacts.
