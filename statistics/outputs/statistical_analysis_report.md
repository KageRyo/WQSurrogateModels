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
- Pairwise model comparisons use external hold-out row absolute-error differences with Wilcoxon signed-rank tests and Holm correction.

## Output Files

- `complete_input_performance.csv`
- `missing_indicator_robustness.csv`
- `cpu_only_timing.csv`
- `stress107_summary.csv`
- `feature_score_correlations.csv`
- `bootstrap_ci.csv`
- `paired_error_tests.csv`

## Pairwise Error Tests

Complete-input full-reference comparisons use external hold-out row absolute-error differences. Negative A-B values favor model A; positive values favor model B.

| Comparison | Mean absolute-error difference (A - B) | 95% CI | Holm p | Result | Significant |
| --- | --- | --- | --- | --- | --- |
| lightgbm vs lr | -6.1617 | [-6.20165, -6.12175] | <1e-300 | lightgbm has significantly lower mean absolute error | yes |
| lightgbm vs mpr | -4.44649 | [-4.47651, -4.41647] | <1e-300 | lightgbm has significantly lower mean absolute error | yes |
| lightgbm vs rf | -0.114044 | [-0.118721, -0.109368] | 1.2783932569282541e-94 | lightgbm has significantly lower mean absolute error | yes |
| lightgbm vs svm | -1.8317 | [-1.84738, -1.81602] | <1e-300 | lightgbm has significantly lower mean absolute error | yes |
| lightgbm vs xgboost | 0.0415714 | [0.0383202, 0.0448225] | 1.8925998025308348e-151 | xgboost has significantly lower mean absolute error | yes |
| lr vs mpr | 1.71521 | [1.67381, 1.75661] | <1e-300 | mpr has significantly lower mean absolute error | yes |
| lr vs rf | 6.04766 | [6.00806, 6.08725] | <1e-300 | rf has significantly lower mean absolute error | yes |
| lr vs svm | 4.33 | [4.28689, 4.37311] | <1e-300 | svm has significantly lower mean absolute error | yes |
| lr vs xgboost | 6.20327 | [6.16351, 6.24303] | <1e-300 | xgboost has significantly lower mean absolute error | yes |
| mpr vs rf | 4.33245 | [4.30249, 4.3624] | <1e-300 | rf has significantly lower mean absolute error | yes |
| mpr vs svm | 2.61479 | [2.58379, 2.6458] | <1e-300 | svm has significantly lower mean absolute error | yes |
| mpr vs xgboost | 4.48806 | [4.45816, 4.51796] | <1e-300 | xgboost has significantly lower mean absolute error | yes |
| rf vs svm | -1.71765 | [-1.73331, -1.702] | <1e-300 | rf has significantly lower mean absolute error | yes |
| rf vs xgboost | 0.155616 | [0.150987, 0.160244] | <1e-300 | xgboost has significantly lower mean absolute error | yes |
| svm vs xgboost | 1.87327 | [1.85767, 1.88887] | <1e-300 | xgboost has significantly lower mean absolute error | yes |

## Reporting Boundary

Stress107 reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.
