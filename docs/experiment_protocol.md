# Revised Experiment Protocol

This document describes the revised reproducibility workflow.

It is intended to support future reproducible runs and robustness checks. The archived manuscript experiments used predefined fixed-size subsets and `80/20` train-test splits; see [original-benchmark-protocol.md](original-benchmark-protocol.md).

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
- Default split counts on the committed `60,714`-row dataset:
  - train: `48,571`
  - test: `12,143`
- No timestamp, station-sequence, or lag feature is used in the current protocol.

## Models

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

`direct_wqi5` is the non-ML baseline and should always be reported beside ML surrogates.

If `xgboost` or `lightgbm` is missing from the runtime environment, treat that as an environment setup failure rather than silently dropping the model from the configured experiment.

By default, the reproducibility script runs on CPU to preserve portability. GPU
execution can be enabled for supported gradient boosting models only:

```bash
python scripts/reproduce_results.py \
  --config configs/experiment_config.yaml \
  --output-dir results_gpu \
  --compute-device gpu \
  --gpu-id 0 \
  --overwrite
```

`xgboost` uses CUDA through `device="cuda:<gpu_id>"` and `tree_method="hist"`.
The installed LightGBM package should be verified before use; this environment
supports `device_type="gpu"` through the OpenCL backend, while
`device_type="cuda"` requires a LightGBM build compiled with CUDA support.
The scikit-learn models in this workflow remain CPU-based.

## Reduced-Indicator Experiment

The reduced-indicator experiment is the primary revision experiment for
addressing why a machine-learning surrogate is useful when WQI5 is deterministic.
It evaluates surrogate regressors when one or more indicators are removed to
simulate incomplete, delayed, or low-cost sensing.

Run:

```bash
python scripts/reproduce_reduced_indicators.py \
  --config configs/reduced_indicator_config.yaml \
  --output-dir results_reduced_indicators
```

See [reduced-indicator-analysis.md](reduced-indicator-analysis.md). Direct WQI5
is treated as the complete-input reference and is not reported for reduced-input
scenarios.

## Metrics

Regression metrics:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Mean Predictive Accuracy (MPA)`

Operational metrics:

- training/inference runtime
- residual mean
- residual standard deviation

Use [metrics.md](metrics.md) for the definition of `Mean Predictive Accuracy (MPA)` and for guidance on separating regression metrics from WQI-band summaries.

## Outputs

Running `scripts/reproduce_results.py` writes:

- `results/metrics_summary.csv`
- `results/repeated_split_results.csv`
- `results/residual_statistics.csv`
- `results/category_metrics.csv`

These files are intended to support reproducible regeneration of the manuscript tables after final hyperparameters are locked.

For verification runs, prefer a separate output directory so archived experiment outputs are not overwritten:

```bash
python scripts/reproduce_results.py --config configs/experiment_config.yaml --output-dir results_verification
```

Use `--overwrite` only when you intentionally want to replace an existing result directory.
