# WQSurrogateModels

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org)
[![CI](https://github.com/KageRyo/WQSurrogateModels/actions/workflows/ci.yml/badge.svg)](https://github.com/KageRyo/WQSurrogateModels/actions/workflows/ci.yml)

WQSurrogateModels is a FastAPI backend and reproducibility repository for WQI5-based current-state water quality assessment.

> Scope: this repository performs WQI5-based current-state water quality assessment.
> It does not perform temporal forecasting because the committed dataset does not contain timestamps.

It provides:

- a `direct_wqi5` baseline
- surrogate regression models
- `/api/v2/*` endpoints for WaterMirror and other HTTP clients
- reproducibility scripts and experiment documentation

## Relationship with the Companion Repository

This project is part of a two-repository system:

- `WaterMirror`: cross-platform mobile frontend for data entry, CSV upload, and result visualization
- `WQSurrogateModels`: FastAPI backend and model/reproducibility repository for WQI5-based current-state water quality assessment

WaterMirror depends on the API contract exposed by this repository. `WQSurrogateModels` can also be used independently through `curl`, Postman, or custom scripts.

## What This Repository Does

- serves a FastAPI backend for WQI5 assessment
- supports a `direct_wqi5` formula baseline
- supports surrogate regression models: `lr`, `mpr`, `svm`, `rf`, `xgboost`, `lightgbm`
- provides reproducibility scripts and experiment configuration
- keeps compatibility with legacy endpoints while treating `/api/v2/*` as the primary contract

## Architecture

```mermaid
flowchart LR
    A[WaterMirror user input or CSV upload] --> B[WaterMirror frontend]
    B --> C[POST /api/v2/assessment or /api/v2/assessment/csv/summary]
    C --> D[WQSurrogateModels FastAPI service]
    D --> E[Input validation and assessment warnings]
    E --> F{Model selection}
    F --> G[direct_wqi5 baseline]
    F --> H[Surrogate regressors: lr mpr svm rf xgboost lightgbm]
    G --> I[WQI5 score category rating range]
    H --> I
    I --> J[Result payload]
    J --> B
```

## Environment

Copy `.env.example` to `.env` and adjust values if needed.

```bash
cp .env.example .env
```

Key variables:

- `MODEL_DIR=models`
- `DEFAULT_MODEL=direct_wqi5`
- `API_HOST=0.0.0.0`
- `API_PORT=8001`
- `AUTO_PORT=false`

## Install

```bash
pip install .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

Local or externally provided scikit-learn surrogate artifacts should be loaded
with the compatible scikit-learn version used during export. Production
artifacts are not committed to Git; see `models/production_model_manifest.json`
for the expected local paths.

To also enable the full set of surrogate models (`xgboost`, `lightgbm`):

```bash
pip install -e ".[dev,models]"
```

## Production Model Artifacts

Production model binaries are local artifacts and are not committed to Git.
The current model package exports one complete-input API artifact for each surrogate
model:

```text
models/LightGBM/modelLGBMVer.2.0-50000-seed0.pkl
models/LR/modelLRVer.2.0-50000-seed0.pkl
models/MPR/modelMPRVer.2.0-50000-seed3.pkl
models/RF/modelRFVer.2.0-50000-seed0.pkl
models/SVM/modelSVMVer.2.0-50000-seed3.pkl
models/XGBoost/modelXGBVer.2.0-50000-seed2.pkl
```

The committed manifest is:

```text
models/production_model_manifest.json
```

Each artifact is extracted from the complete-input `full_reference`
result with the lowest external `10,714`-row hold-out MAE for that model type.
These artifacts remain complete-input WQI5 surrogates and require:

```text
DO, BOD, NH3N, EC, SS
```

They should not be interpreted as missing-indicator replacement models. Legacy
production artifacts are kept locally under `models/archive/legacy_v1/` for
traceability, and experiment bundles remain under ignored `results_*` folders.

## Run

```bash
python main.py
```

If `API_PORT` is already occupied, the default behavior is to fail fast with a clearer error message. For local development, you can opt in to automatic fallback ports:

```env
AUTO_PORT=true
```

With `AUTO_PORT=true`, the server tries `API_PORT` first and then scans upward (`8002`, `8003`, ...) until it finds a free port.

## API

Primary endpoints live under `/api/v2/*`.

### Quick example

`POST /api/v2/assessment`

```json
{ "DO": 7.2, "BOD": 2.1, "NH3N": 0.3, "EC": 450, "SS": 12, "model_type": "lightgbm" }
```

Legacy compatibility endpoints such as `POST /predict`, `POST /score/total/`, and `GET /status` are retained but deprecated.

## Documentation

- [WaterMirror Integration](docs/watermirror-integration.md)
- [API Reference](docs/api-reference.md)
- [Full-Stack Local Run](docs/fullstack-local-run.md)
- [WQI5 Formula](docs/wqi5-formula.md)
- [Metrics](docs/metrics.md)
- [Data Preparation](docs/data_preparation.md)
- [Original Benchmark Protocol](docs/original-benchmark-protocol.md)
- [Experiment Protocol](docs/experiment_protocol.md)
- [Earlier Missing-Indicator Core Experiments](docs/missing-indicator-core-experiments.md)
- [Missing-Indicator Robustness Experiments](docs/missing-indicator-robustness-experiments.md)
- [Statistical Analysis](docs/statistical-analysis.md)
- [Statistics Workspace Notes](statistics/README.md)
- [Model Hyperparameters](docs/model-hyperparameters.md)
- [Model Card](docs/model_card.md)
- [Limitations](docs/limitations.md)

## Reproducibility

Run:

```bash
pip install -e ".[dev]"
python scripts/reproduce_results.py --config configs/experiment_config.yaml --output-dir results/verification_run
```

If you use the local `WQI` conda environment and want to run the full experiment (all models including xgboost/lightgbm):

```bash
conda activate WQI
pip install -e ".[models]"
python scripts/reproduce_results.py --config configs/experiment_config.yaml --output-dir results/verification_run
```

To protect archived manuscript outputs, the script now refuses to overwrite an existing results directory unless `--overwrite` is passed explicitly.

Run the missing-indicator core experiments:

```bash
python scripts/run_missing_indicator_experiments.py \
  --config configs/missing_indicator_config.yaml \
  --output-dir results/missing_indicator_core_run \
  --compute-device gpu \
  --gpu-id 0
```

This workflow saves model artifacts, internal-test predictions, external
`10,714`-row inference predictions, summary metrics, confidence intervals,
paired tests, and stress-scenario summaries into the selected output directory.

Run the missing-indicator robustness workflow with single-indicator missing
settings, event-window stress testing, Stress107 sequential event-window stress,
and CPU-only timing support:

```bash
python scripts/run_missing_indicator_robustness_experiments.py \
  --config configs/missing_indicator_robustness_config.yaml \
  --output-dir results/missing_indicator_robustness_run

python scripts/measure_missing_indicator_cpu_timing.py \
  --output-dir results/missing_indicator_robustness_run

python scripts/run_stress107_event_windows.py \
  --artifact-dir results/missing_indicator_robustness_run \
  --output-dir results/stress107_run

python scripts/export_missing_indicator_robustness_excel.py \
  --output-dir results/stress107_run
```

Stress107 divides the external `10,714`-row hold-out into `107` consecutive
event windows and applies 30%, 100%, and 300% synthetic perturbations. It should
not be described as `107-fold cross-validation`; these are event locations, not
training-validation folds.

Prepare manuscript result tables and production model artifacts from the
organized result bundle:

```bash
python scripts/prepare_statistics_outputs.py \
  --bundle-dir results/manuscript_package \
  --output-dir statistics/outputs \
  --update-production-model \
  --archive-legacy-50000-artifacts
```

Manuscript-facing tables are written to:

- `statistics/outputs/table6_complete_input_performance.csv`
- `statistics/outputs/table7_missing_indicator_robustness.csv`
- `statistics/outputs/table8_cpu_only_timing.csv`
- `statistics/outputs/table9_stress107_summary.csv`
- `statistics/outputs/bootstrap_ci.csv`
- `statistics/outputs/paired_error_tests.csv`

GPU and multicore CPU acceleration may be used for model-effect reproduction.
CPU-only timing is reported separately as a deployment-oriented inference-time
reference.

### Local Result Archive

Large experiment outputs are organized under `results/` and are not committed
to Git. The current local layout is:

- `results/complete_input_cpu/`: complete-input repeated validation on CPU.
- `results/complete_input_gpu/`: complete-input repeated validation with GPU
  acceleration for supported models.
- `results/reduced_indicator_cpu/`: reduced-indicator experiment on CPU.
- `results/reduced_indicator_gpu/`: reduced-indicator experiment with GPU
  acceleration for supported models.
- `results/missing_indicator_core/`: core missing-indicator experiment with
  saved models, predictions, metrics, confidence intervals, paired tests, and
  stress summaries.
- `results/missing_indicator_robustness/`: single-indicator and combined
  missing-indicator robustness results, including CPU-only timing outputs.
- `results/stress107/`: 107 sequential event-window stress-test outputs.
- `results/manuscript_package/`: organized CSV files and Excel workbooks for
  manuscript tables and discussion.

Model binaries under `models/*/*.pkl` are also local artifacts and are not
committed. `models/production_model_manifest.json` records the expected local
paths and source experiment artifacts for the six supported model families.

### Reproducibility Hyperparameters

The table below describes the current reproducibility workflow. Archived exploratory scripts may use `GridSearchCV` and library defaults; see [docs/original-benchmark-protocol.md](docs/original-benchmark-protocol.md).

| Model | Library | Preprocessing | Key Hyperparameters |
| --- | --- | --- | --- |
| `direct_wqi5` | formula baseline | none | direct WQI5 equation |
| `lr` | scikit-learn | mean imputation + standard scaling | default `LinearRegression()` |
| `mpr` | scikit-learn | mean imputation + polynomial features + standard scaling | `degree=2`, `include_bias=False` |
| `svm` | scikit-learn | mean imputation + standard scaling | `kernel=rbf`, `C=10.0`, `epsilon=0.1` |
| `rf` | scikit-learn | mean imputation | `n_estimators=300`, `random_state=0`, `n_jobs=-1` |
| `xgboost` | xgboost | mean imputation | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `random_state=0` |
| `lightgbm` | lightgbm | mean imputation | `n_estimators=300`, `learning_rate=0.05`, `random_state=0` |

Repeated validation uses stratified random splits over WQI5 categories with seeds `0, 1, 2, 3, 4`.

## Project Structure

- `data/`: processed datasets and subsets
- `models/`: production model manifest and local artifact paths; model binaries are not committed
- `src/`: API and reusable backend logic
- `scripts/`: reproducibility runners
- `configs/`: experiment settings
- `tests/`: pytest suite

## License

Apache License 2.0. See `LICENSE`.
