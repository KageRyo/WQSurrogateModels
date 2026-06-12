# Limitations

- The committed dataset does not contain timestamps, so this repository should not be described as a temporal forecasting system.
- The released repository versions the processed modeling dataset, but not the full upstream extraction and intermediate cleaning audit trail.
- The current framing is cross-sectional WQI5 assessment rather than time-dependent prediction.
- Optional surrogate model families such as `xgboost` and `lightgbm` require additional runtime dependencies when training or loading those artifacts.
- Direct WQI5 computation remains the reference method under complete-input conditions; surrogate models should not be described as mathematically superior to the deterministic formula.
- Reduced-indicator experiments estimate the reference WQI5 score under incomplete-input scenarios. They support deployment-oriented analysis, not claims of future water-quality forecasting.
