# Model Card

## Task Definition

`WQSurrogateModels` performs WQI5-based current-state water quality assessment from five input indicators.

This repository does not perform temporal forecasting.

`Score` denotes a dimensionless WQI5 index on a `0-100` scale.

## Inputs

- `DO`
- `BOD`
- `NH3N`
- `EC`
- `SS`

## Outputs

- `score`
- `category`
- `rating_range`
- `assessment`
- `warnings`

## Supported Models

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

## Current Local Inference Artifacts

Model binaries are local artifacts and are not committed to Git. The model
package exports one complete-input API artifact for each surrogate model:

```text
models/LightGBM/modelLGBMVer.2.0-50000-seed0.pkl
models/LR/modelLRVer.2.0-50000-seed0.pkl
models/MPR/modelMPRVer.2.0-50000-seed3.pkl
models/RF/modelRFVer.2.0-50000-seed0.pkl
models/SVM/modelSVMVer.2.0-50000-seed3.pkl
models/XGBoost/modelXGBVer.2.0-50000-seed2.pkl
```

The committed manifest is `models/production_model_manifest.json`. Each entry is
selected as the lowest-MAE complete-input seed artifact for that model type on
the fixed external `10,714`-row hold-out in the organized result bundle.

The local inference artifacts are complete-input WQI5 surrogates:

```text
DO, BOD, NH3N, EC, SS -> WQI5 score
```

They are not missing-indicator models. Missing-indicator experiment artifacts are
kept as reproducibility outputs under ignored `results_*` folders and should not
be copied into `models/` unless the API is explicitly extended to route those
settings.

## Intended Use

- backend assessment for `WaterMirror`
- API-based batch or single-record WQI5 assessment
- reproducibility and comparison of direct and surrogate approaches

## Not Intended Use

- temporal forecasting
- causal inference
- unsupported water quality indices beyond the documented WQI5 framing
- replacing deterministic WQI5 when all five indicators are available
- treating reduced-indicator models as reliable substitutes for complete-input WQI5
- describing the 107-window stress test as real pollution-event validation
- decision-making without reviewing domain-specific limitations and data provenance constraints

## Limitations

- The committed dataset does not contain timestamps.
- The processed dataset is versioned, but exact upstream extraction and intermediate cleaning logs are not yet fully recoverable.
- Optional model families such as `xgboost` and `lightgbm` require their corresponding runtime dependencies.
- External hold-out results show that `BOD` is a critical indicator; reduced-input settings without `BOD` should be interpreted conservatively.
- CPU-only timing is a rough inference-time reference, not direct proof of performance on a low-end edge device.
