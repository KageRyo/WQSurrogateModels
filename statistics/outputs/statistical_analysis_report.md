# Statistical Summary

## Scope

This report summarizes the missing-indicator robustness, Stress107, and CPU-only timing outputs.
It replaces the earlier percentage-agreement tables with R2, MAE, RMSE, Macro-F1, bootstrap confidence intervals, and paired model tests.

The task remains WQI5 surrogate regression, not future water-quality forecasting. Direct WQI5 computation remains the reference method when all five indicators are available.

## Main Findings

- Complete-input external hold-out best model: `xgboost` with R2=0.9993, MAE=0.2504, RMSE=0.4072.
- Missing NH3N reduced retraining remains useful as an auxiliary setting: `lightgbm` with R2=0.9494, MAE=2.3694.
- DO/EC/SS-only reduced retraining is not reliable on the external hold-out: `rf` with R2=-0.1401, MAE=14.7246.
- Stress107 uses 107 sequential event windows, not 107-fold cross-validation.
- CPU-only timing is the deployment-oriented inference-time reference; GPU/multicore acceleration is acceptable for experiment reproduction.
- Pairwise model comparisons use seed-level external hold-out MAE with Holm-adjusted exact two-sided Wilcoxon signed-rank p-values within each experiment family.

## Output Files

- `complete_input_performance.csv`
- `missing_indicator_robustness.csv`
- `cpu_only_timing.csv`
- `stress107_summary.csv`
- `feature_score_correlations.csv`
- `bootstrap_ci.csv`
- `paired_error_tests.csv`
- `sample_size_sensitivity.csv`
- `sample_size_metrics_by_fold.csv`

## Sample-Size Sensitivity

The sample-size experiment evaluates six surrogate models using 1,000, 10,000, and 50,000 rows under stratified 80/20 splits. The summary output is `sample_size_sensitivity.csv`, and fold-level results are provided in `sample_size_metrics_by_fold.csv`.

## Pairwise Error Tests

Complete-input full-reference comparisons use seed-level external hold-out MAE from the five repeated model runs. Negative A-B values favor model A; positive values favor model B. Holm correction is applied within this 15-comparison model family.

| Comparison | Mean MAE difference (A - B) | Holm-adjusted p-value | Result | Significant |
| --- | --- | --- | --- | --- |
| lightgbm vs lr | -6.1616993752526552 | 0.9375 | not significant; lower mean absolute error: lightgbm | no |
| lightgbm vs mpr | -4.4464900302408079 | 0.9375 | not significant; lower mean absolute error: lightgbm | no |
| lightgbm vs rf | -0.11404429540742474 | 0.9375 | not significant; lower mean absolute error: lightgbm | no |
| lightgbm vs svm | -1.8316971531781541 | 0.9375 | not significant; lower mean absolute error: lightgbm | no |
| lightgbm vs xgboost | 0.041571368131353689 | 0.9375 | not significant; lower mean absolute error: xgboost | no |
| lr vs mpr | 1.7152093450118475 | 0.9375 | not significant; lower mean absolute error: mpr | no |
| lr vs rf | 6.0476550798452298 | 0.9375 | not significant; lower mean absolute error: rf | no |
| lr vs svm | 4.3300022220745005 | 0.9375 | not significant; lower mean absolute error: svm | no |
| lr vs xgboost | 6.2032707433840084 | 0.9375 | not significant; lower mean absolute error: xgboost | no |
| mpr vs rf | 4.3324457348333834 | 0.9375 | not significant; lower mean absolute error: rf | no |
| mpr vs svm | 2.6147928770626536 | 0.9375 | not significant; lower mean absolute error: svm | no |
| mpr vs xgboost | 4.4880613983721611 | 0.9375 | not significant; lower mean absolute error: xgboost | no |
| rf vs svm | -1.7176528577707295 | 0.9375 | not significant; lower mean absolute error: rf | no |
| rf vs xgboost | 0.15561566353877843 | 0.9375 | not significant; lower mean absolute error: xgboost | no |
| svm vs xgboost | 1.8732685213095077 | 0.9375 | not significant; lower mean absolute error: xgboost | no |

## Reporting Boundary

The p-values test paired seed-level MAE differences across five repeated model runs. With `n=5` and a two-sided exact Wilcoxon test, the smallest possible raw p-value is `0.0625`, so the current repeated-run design does not support claiming pairwise statistical significance at alpha 0.05.

Stress107 reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.
