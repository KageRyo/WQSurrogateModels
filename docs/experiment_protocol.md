# Experiment Protocol

This document describes the reproducibility workflow.

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

`direct_wqi5` is the deterministic reference when all five indicators are
available. Manuscript tables comparing learned surrogates should state clearly
that the models approximate the reference WQI5 score rather than replace the
deterministic formula.

If `xgboost` or `lightgbm` is missing from the runtime environment, treat that as an environment setup failure rather than silently dropping the model from the configured experiment.

By default, the reproducibility script runs on CPU to preserve portability. GPU
execution can be enabled for supported gradient boosting models only:

```bash
python scripts/reproduce_results.py \
  --config configs/experiment_config.yaml \
  --output-dir results/complete_input_gpu_run \
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

The reduced-indicator experiment was an earlier analysis component for
addressing why a machine-learning surrogate may be useful when WQI5 is
deterministic. It evaluates surrogate regressors when one or more indicators are
removed to simulate incomplete or delayed sensing.

The current manuscript-facing result package is the missing-indicator
robustness suite, including single-indicator missing settings, two-stage
indicator reconstruction, Stress107, and CPU-only inference timing.

Run:

```bash
python scripts/reproduce_reduced_indicators.py \
  --config configs/reduced_indicator_config.yaml \
  --output-dir results/reduced_indicator_run
```

See [reduced-indicator-analysis.md](reduced-indicator-analysis.md). Direct WQI5
is treated as the complete-input reference and is not reported for reduced-input
scenarios.

## Missing-Indicator Core Suite

The missing-indicator core suite uses `data/dataV1_50000.csv` for
stratified `80/20` training and internal testing, then evaluates the trained
models on the `10,714` rows in `data/dataV1.csv` that are not present in the
50,000-row subset.

Run:

```bash
python scripts/run_missing_indicator_experiments.py \
  --config configs/missing_indicator_config.yaml \
  --output-dir results/missing_indicator_core_run \
  --compute-device gpu \
  --gpu-id 0
```

See [missing-indicator-core-experiments.md](missing-indicator-core-experiments.md).

## Missing-Indicator Robustness Suite

The robustness suite extends the missing-indicator core workflow by adding
single-indicator missing settings, Stress107, and CPU-only inference timing.
It evaluates:

- `missing_bod`: BOD unavailable.
- `missing_nh3n`: NH3N unavailable.
- `missing_bod_nh3n`: BOD and NH3N unavailable.

For each missing setting, the workflow reports:

- `inference_dropout`: complete-input model evaluated with selected indicators
  set to missing at inference time.
- `reduced_retraining`: model trained and evaluated only with available
  indicators.
- `indicator_reconstruction`: missing indicator(s) reconstructed from available
  indicators before WQI5 surrogate inference.

The earlier localized middle-window stress block is retained only as a disabled
backward-compatible configuration. The current workflow uses Stress107, which
divides the external `10,714`-row hold-out into `107` consecutive
non-overlapping event windows and applies 30%, 100%, and 300% synthetic
perturbations. The suite also includes a separate CPU-only inference timing
workflow from saved artifacts.

Run:

```bash
python scripts/run_missing_indicator_robustness_experiments.py \
  --config configs/missing_indicator_robustness_config.yaml \
  --output-dir results/missing_indicator_robustness_run
```

Measure CPU-only inference timing:

```bash
python scripts/measure_missing_indicator_cpu_timing.py \
  --output-dir results/missing_indicator_robustness_run
```

Export the workbook:

```bash
python scripts/export_missing_indicator_robustness_excel.py \
  --output-dir results/missing_indicator_robustness_run
```

See [missing-indicator-robustness-experiments.md](missing-indicator-robustness-experiments.md).

## Metrics

Regression metrics:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- WQI-band `Accuracy`
- WQI-band `Macro-F1`

Operational metrics:

- training/inference runtime
- residual mean
- residual standard deviation

Earlier percentage-agreement metrics are not used as primary metrics in the
manuscript tables. See [metrics.md](metrics.md) for metric definitions and for
guidance on separating regression metrics from WQI-band summaries.

## Outputs

Running `scripts/reproduce_results.py` writes:

- `results/metrics_summary.csv`
- `results/repeated_split_results.csv`
- `results/residual_statistics.csv`
- `results/category_metrics.csv`

These files are intended to support reproducible regeneration of the manuscript tables after the experiment hyperparameters are locked.

For verification runs, prefer a separate output directory so archived experiment outputs are not overwritten:

```bash
python scripts/reproduce_results.py --config configs/experiment_config.yaml --output-dir results/verification_run
```

Use `--overwrite` only when you intentionally want to replace an existing result directory.
