# Statistical Analysis

This document describes the revision 2026-06-14 statistics used for the
missing-indicator robustness, Stress107, and CPU-only timing results.

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

`Mean Predictive Accuracy (MPA)` is not used as a primary manuscript metric in
the revision. It may remain only as archived context for earlier
exploratory outputs.

## Confidence Intervals

The revision missing-indicator robustness workflow runs five stratified seeds over
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

The revision CI table is:

```text
statistics/outputs/revision_bootstrap_ci.csv
```

Table 6 uses:

```text
statistics/outputs/revision_table6_complete_input_performance.csv
```

Table 7 uses:

```text
statistics/outputs/revision_table7_missing_indicator_robustness.csv
```

## Significance Testing

Pairwise model comparisons use paired Wilcoxon signed-rank tests over per-seed
`MAE`, followed by Holm correction.

The revision p-value table is:

```text
statistics/outputs/revision_paired_error_tests.csv
```

Because there are five seeds, p-values should be interpreted as supporting
evidence rather than as a standalone proof of model superiority.

## Stress107 Analysis

Stress107 is a controlled synthetic event-window stress test. The external
`10,714`-row hold-out is divided into `107` consecutive non-overlapping windows.
Each window is perturbed under 30%, 100%, and 300% severity levels for five
pollution-like scenarios.

Stress107 should not be called `107-fold cross-validation`. The windows are
event locations, not model-training folds.

The Stress107 summary is:

```text
statistics/outputs/revision_table9_stress107_summary.csv
```

Stress107 reduces concern that the stress-test conclusion depends on a single
selected middle window. It does not prove absence of all sampling bias and does
not replace validation on real timestamped pollution-event data.

## CPU-Only Timing

GPU and multicore CPU acceleration may be used to reproduce model-effect
experiments efficiently. Deployment-oriented timing is reported separately using
CPU-only inference from saved model artifacts.

The CPU-only timing table is:

```text
statistics/outputs/revision_table8_cpu_only_timing.csv
```

This table is a conservative inference-time reference. It should not be
described as direct validation on a low-end edge device unless such hardware is
actually tested.

## Revision Report

The frozen revision summary is:

```text
statistics/outputs/revision_statistical_summary.md
```
