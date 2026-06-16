# Statistical Analysis

This document describes the statistics used for the missing-indicator
robustness, Stress107, and CPU-only timing results.

## Interpretation Boundaries

The task is WQI5 surrogate regression. The target `Score` is computed from the
same five indicators used under complete-input conditions, so complete-input
results are WQI5 approximation results, not future water-quality forecasting.

Direct WQI5 computation remains the reference method when `DO`, `BOD`, `NH3N`,
`EC`, and `SS` are all available. Reduced-input, inference-dropout, and
indicator-reconstruction experiments evaluate auxiliary behavior under
deployment constraints.

## Primary Metrics

Primary manuscript reporting uses:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- WQI-band `Accuracy`
- WQI-band `Macro-F1`
- residual mean and residual standard deviation where useful

Earlier percentage-agreement metrics are not used as primary manuscript metrics
in the current result package. They remain only as archived context for earlier exploratory
outputs outside the current result package.

## Confidence Intervals

The missing-indicator robustness workflow runs five stratified seeds over
`data/dataV1_50000.csv`:

- `40,000` training rows per seed
- `10,000` internal-test rows per seed
- fixed `10,714` external hold-out rows from `data/dataV1.csv`

Seed-level bootstrap `95%` confidence intervals are reported for:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Accuracy`
- `Macro-F1`

The confidence-interval output is:

```text
statistics/outputs/bootstrap_ci.csv
```

The complete-input performance output is:

```text
statistics/outputs/complete_input_performance.csv
```

The missing-indicator robustness output is:

```text
statistics/outputs/missing_indicator_robustness.csv
```

## Significance Testing

Pairwise model comparisons report Holm-adjusted exact two-sided Wilcoxon
signed-rank p-values over seed-level external hold-out MAE values. Holm
correction is applied within each source, missing-indicator setting, experiment
mode, and experiment family.

For each compared model pair within the same source, missing-indicator setting,
experiment mode, experiment, and seed, the paired difference is:

```text
diff_seed = MAE_A_seed - MAE_B_seed
```

Negative values favor model A. Positive values favor model B. Public reporting
writes `p_value` as the Holm-adjusted p-value with full double-precision
formatting rather than rounded display values.

The paired-test output is:

```text
statistics/outputs/paired_error_tests.csv
```

The current paired-test output uses `external_10714` metrics only. Each
comparison uses the five seed-specific hold-out MAE values, so each complete
pair has `5` paired differences. With `n=5` and a two-sided exact Wilcoxon
test, the smallest possible raw p-value is `0.0625`.

## Stress107 Analysis

Stress107 is a controlled synthetic event-window stress test. The external
`10,714`-row hold-out is divided into `107` consecutive non-overlapping windows.
Each window is perturbed under 30%, 100%, and 300% severity levels for five
pollution-like scenarios.

Stress107 should not be called `107-fold cross-validation`. The windows are
event locations, not model-training folds.

The Stress107 summary is:

```text
statistics/outputs/stress107_summary.csv
```

Stress107 reduces concern that the stress-test conclusion depends on a single
selected middle window. It does not prove absence of all sampling bias and does
not replace validation on real timestamped pollution-event data.

## Feature-Score Correlation

Feature-score correlations are descriptive because WQI5 `Score` is constructed
from the same five indicators. They are included only to document the processed
dataset relationships used by the surrogate-regression experiments.

The correlation output is:

```text
statistics/outputs/feature_score_correlations.csv
```

## CPU-Only Timing

GPU and multicore CPU acceleration may be used to reproduce model-effect
experiments efficiently. Deployment-oriented timing is reported separately using
CPU-only inference from saved model artifacts.

The CPU-only timing output is:

```text
statistics/outputs/cpu_only_timing.csv
```

This output is a conservative inference-time reference. It should not be
described as direct validation on a low-end edge device unless such hardware is
actually tested.

## Report

The statistical summary is:

```text
statistics/outputs/statistical_summary.md
```
