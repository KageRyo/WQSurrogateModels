# Model Card

## Task Definition

`WQSurrogateModels` performs WQI5-based current-state water quality assessment from five input indicators.

This repository does not perform temporal forecasting.

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

## Intended Use

- backend assessment for `WaterMirror`
- API-based batch or single-record WQI5 assessment
- reproducibility and reviewer-facing comparison of direct and surrogate approaches

## Not Intended Use

- temporal forecasting
- causal inference
- unsupported water quality indices beyond the documented WQI5 framing
- decision-making without reviewing domain-specific limitations and data provenance constraints

## Limitations

- The committed dataset does not contain timestamps.
- The processed dataset is versioned, but exact upstream extraction and intermediate cleaning logs are not yet fully recoverable.
- Optional model families such as `xgboost` and `lightgbm` require their corresponding runtime dependencies.
