# Limitations

- The committed dataset does not contain timestamps, so this repository should not be described as a temporal forecasting system.
- The released repository versions the processed modeling dataset, but not the full upstream extraction and intermediate cleaning audit trail.
- The current framing is cross-sectional WQI5 assessment rather than time-dependent prediction.
- Optional surrogate model families such as `xgboost` and `lightgbm` require additional runtime dependencies when training or loading those artifacts.
