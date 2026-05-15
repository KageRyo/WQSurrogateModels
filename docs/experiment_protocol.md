# Experiment Protocol

## Task Definition

This repository evaluates `current-state WQI5 surrogate regression` from five water indicators:

- `DO`
- `BOD`
- `NH3N`
- `EC`
- `SS`

The task is not forecasting. The committed dataset has no timestamp field.

## Split Strategy

- Validation type: `cross-sectional`
- Split method: `StratifiedShuffleSplit`
- Stratification target: WQI5 category derived from `Score`
- Seeds: `0, 1, 2, 3, 4`
- Test size: `20%`

## Models

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

`direct_wqi5` is the non-ML baseline and should always be reported beside ML surrogates.

## Metrics

Regression metrics:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`

Category metrics:

- `Accuracy`
- `Macro-F1`

Operational metrics:

- training/inference runtime
- residual mean
- residual standard deviation

## Outputs

Running `scripts/reproduce_results.py` writes:

- `results/metrics_summary.csv`
- `results/repeated_split_results.csv`
- `results/residual_statistics.csv`
- `results/category_metrics.csv`

These files are intended to support reproducible regeneration of the manuscript tables after final hyperparameters are locked.
