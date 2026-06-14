# Revision Statistical Summary

## Scope

This report freezes the 2026-06-14 missing-indicator robustness, Stress107, and CPU-only timing outputs.
It replaces the earlier MPA-centered manuscript statistics with R2, MAE, RMSE, Macro-F1, bootstrap confidence intervals, and paired model tests.

The task remains WQI5 surrogate regression, not future water-quality forecasting. Direct WQI5 computation remains the reference method when all five indicators are available.

## Main Findings

- Complete-input external hold-out best model: `xgboost` with R2=0.9993, MAE=0.2504, RMSE=0.4072.
- Missing NH3N reduced retraining remains useful as an auxiliary setting: `lightgbm` with R2=0.9494, MAE=2.3694.
- DO/EC/SS-only reduced retraining is not reliable on the external hold-out: `rf` with R2=-0.1401, MAE=14.7246.
- Stress107 uses 107 sequential event windows, not 107-fold cross-validation.
- CPU-only timing is the deployment-oriented inference-time reference; GPU/multicore acceleration is acceptable for experiment reproduction.

## Output Tables

- `revision_table6_complete_input_performance.csv`
- `revision_table7_missing_indicator_robustness.csv`
- `revision_table8_cpu_only_timing.csv`
- `revision_table9_stress107_summary.csv`
- `revision_bootstrap_ci.csv`
- `revision_paired_error_tests.csv`

## Reporting Boundary

Stress107 reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.
