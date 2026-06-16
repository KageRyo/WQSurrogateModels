# Sample-Size Experiments

This workflow evaluates how the six learned WQI5 surrogate models behave under
different training data volumes.

## Scope

Sample sizes:

- `1,000` rows: `data/dataV1_1000.csv`
- `5,000` rows: `data/dataV1_5000.csv`
- `10,000` rows: `data/dataV1_10000.csv`
- `50,000` rows: `data/dataV1_50000.csv`

All four sample sizes have completed runs in the consolidated local output.

## Models

The sample-size workflow runs the six learned surrogate models:

- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

It does not include `direct_wqi5`, because this experiment compares trained
surrogate models rather than the deterministic formula baseline.

## Split Protocol

Each dataset uses `StratifiedKFold` with:

- `5` folds
- WQI5 category stratification derived from `Score`
- `80%` training rows and `20%` test rows per fold
- `shuffle=True`
- `random_state=0`

For every dataset, fold, model, and split, the workflow reports:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- WQI-band `Accuracy`
- WQI-band `Macro-F1`
- fit and prediction runtime
- residual mean and residual standard deviation

## GPU Use

GPU execution is optional. When `--compute-device gpu` is used:

- `xgboost` uses CUDA through the shared model builder.
- `lightgbm` uses `device_type="gpu"` by default.
- `lr`, `mpr`, `svm`, and `rf` remain CPU-based scikit-learn models.

## Run Commands

Run the `1,000`-row experiment:

```bash
python scripts/run_sample_size_experiments.py \
  --datasets data/dataV1_1000.csv \
  --compute-device gpu \
  --gpu-id 0 \
  --lightgbm-gpu-backend gpu
```

Run the `10,000`- and `50,000`-row datasets together:

```bash
python scripts/run_sample_size_experiments.py \
  --datasets data/dataV1_10000.csv data/dataV1_50000.csv \
  --compute-device gpu \
  --gpu-id 0 \
  --lightgbm-gpu-backend gpu
```

Run all completed sample sizes:

```bash
python scripts/run_sample_size_experiments.py \
  --datasets \
    data/dataV1_1000.csv \
    data/dataV1_5000.csv \
    data/dataV1_10000.csv \
    data/dataV1_50000.csv \
  --compute-device gpu \
  --gpu-id 0 \
  --lightgbm-gpu-backend gpu
```

By default, the script writes to the stable directories
`results/sample_size_experiments` and `models/sample_size_experiments`. It
refuses to write into a non-empty directory; pass explicit `--output-dir` and
`--model-dir` values for separate verification runs.

## Outputs

Each run writes:

- `manifest.json`
- `metrics/metrics_by_fold.csv`
- `metrics/metrics_summary.csv`
- `splits/split_indices.csv`

Manuscript-facing sample-size outputs are available at:

- `statistics/outputs/sample_size_sensitivity.csv`
- `statistics/outputs/sample_size_metrics_by_fold.csv`

These committed CSV files contain the teacher-requested `1,000`, `10,000`, and
`50,000` row settings. The local consolidated run also includes `5,000` rows,
but that intermediate size is not included in the main manuscript-facing table.

Model artifacts are saved as:

```text
models/sample_size_experiments/<Model>/<dataset>/fold_<fold>/<model>.pkl
```

Examples:

```text
models/sample_size_experiments/LR/dataV1_1000/fold_1/lr.pkl
models/sample_size_experiments/XGBoost/dataV1_50000/fold_5/xgboost.pkl
```

The current consolidated local outputs are:

```text
results/sample_size_experiments
models/sample_size_experiments
```

These consolidated outputs include the `1,000`, `5,000`, `10,000`, and `50,000`
row sample-size runs.
