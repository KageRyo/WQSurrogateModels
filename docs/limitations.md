# Limitations

- This repository does not include the complete dataset, processed experiment
  table, or experiment subsets. Monitoring data can be downloaded from the
  official source described in [data_preparation.md](data_preparation.md).
- The prepared experiment input has no timestamps, so the supported task is
  cross-sectional WQI5 assessment.
- The repository documents only the aggregate row counts, six-column schema,
  and preprocessing operations needed to prepare a compatible local input.
- Optional surrogate model families such as `xgboost` and `lightgbm` require their exact recorded runtime dependencies when loading production artifacts; see [Production Artifact and ARM64 Runtime](edge-runtime.md).
- With complete inputs, the deterministic WQI5 formula is the reference method.
- Reduced-indicator experiments measure how closely surrogate models estimate the reference WQI5 score when indicators are missing.
- The external hold-out results show that missing `NH3N` can still support useful auxiliary estimation, but missing `BOD` and especially `BOD + NH3N` have weak external generalization as reduced-input substitutes.
- The 107-window stress test uses controlled synthetic event windows and reduces dependence on any single selected window. Real typhoon, rainfall, and pollution events have not been tested.
- CPU-only inference timing measured from saved artifacts is a rough reference. Measure edge-device performance on the target hardware.
- GPU and multicore CPU timings are for repeated model-comparison experiments only.
