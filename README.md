# WQSurrogateModels

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org)

`WQSurrogateModels` is the backend and model repository for `WaterMirror`.

It supports `WQI5-based current-state water quality assessment`, not future forecasting. The committed dataset does not contain timestamps, so this project must be described as `cross-sectional surrogate regression` and `current-state assessment`.

## What This Repository Does

- serves a FastAPI backend for WQI5 assessment
- supports a `direct_wqi5` formula baseline
- supports surrogate regression models: `lr`, `mpr`, `svm`, `rf`, `xgboost`, `lightgbm`
- provides reproducibility scripts and experiment configuration
- keeps compatibility with the legacy CSV upload endpoint used by `WaterMirror`

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

## Install

```bash
pip install .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Run

```bash
python main.py
```

## API

### `POST /predict`

```json
{
  "DO": 7.2,
  "BOD": 2.1,
  "NH3N": 0.3,
  "EC": 450,
  "SS": 12,
  "model_type": "lightgbm"
}
```

Response:

```json
{
  "score": 82.5,
  "category": "Good",
  "rating_range": "70 < WQI5 ≤ 85",
  "model_type": "lightgbm",
  "latency_ms": 12.4,
  "assessment": {
    "DO": "Good",
    "BOD": "Fair",
    "NH3N": "Fair",
    "EC": "Fair",
    "SS": "Fair"
  },
  "warnings": []
}
```

### Other endpoints

- `GET /status`
- `GET /models`
- `GET /percentile?score=82.5`
- `GET /categories`
- `POST /score/total/` for legacy CSV mean-score compatibility
- `POST /score/all/` for legacy CSV per-row scores

## Reproducibility

Run:

```bash
python scripts/reproduce_results.py --config configs/experiment_config.yaml
```

Outputs are written to `results/`.

Supporting documentation:

- [docs/data_preparation.md](/mnt/8tb_hdd/ryo/WQSurrogateModels/docs/data_preparation.md)
- [docs/experiment_protocol.md](/mnt/8tb_hdd/ryo/WQSurrogateModels/docs/experiment_protocol.md)

## Project Structure

- `data/`: processed datasets and subsets
- `models/`: persisted surrogate model artifacts
- `src/`: API and reusable backend logic
- `scripts/`: reproducibility runners
- `configs/`: experiment settings
- `tests/`: pytest suite

## Limitations

- no timestamp column is available in the committed dataset
- no claim of temporal forecasting should be made
- direct raw-data provenance from the upstream `87,005` records still needs a versioned audit trail in the repo
- optional dependencies such as `xgboost` and `lightgbm` must exist in the runtime environment to retrain those models

## License

Apache License 2.0. See `LICENSE`.
