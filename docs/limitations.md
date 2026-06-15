# Limitations

- The committed dataset does not contain timestamps, so this repository should not be described as a temporal forecasting system.
- The released repository versions the processed modeling dataset, but not the full upstream extraction and intermediate cleaning audit trail.
- The current framing is cross-sectional WQI5 assessment rather than time-dependent prediction.
- Optional surrogate model families such as `xgboost` and `lightgbm` require additional runtime dependencies when training or loading those artifacts.
- Direct WQI5 computation remains the reference method under complete-input conditions; surrogate models should not be described as mathematically superior to the deterministic formula.
- Reduced-indicator experiments estimate the reference WQI5 score under incomplete-input scenarios. They support deployment-oriented analysis, not claims of future water-quality forecasting.
- The external hold-out results show that missing `NH3N` can still support useful auxiliary estimation, but missing `BOD` and especially `BOD + NH3N` have weak external generalization for reduced-input replacement.
- Stress107 is a controlled synthetic event-window stress test, not real typhoon, rainfall, or pollution-event validation.
- Stress107 reduces dependence on one selected event window; it does not prove the absence of all sampling bias.
- CPU-only inference timing is a deployment-oriented timing reference measured from saved artifacts. It is not direct validation on a low-end edge device unless that hardware is explicitly tested.
- GPU and multicore CPU acceleration may be used for repeated model-effect experiments, but GPU timing should not be used as edge-device deployment evidence.
