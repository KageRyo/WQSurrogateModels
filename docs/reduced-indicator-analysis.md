# Reduced-Indicator Analysis

This analysis addresses the reviewer concern that the five-input surrogate
models may simply reconstruct a deterministic WQI5 index.

## Framing

Direct WQI5 computation remains the reference method when all five indicators
are available. The reduced-indicator experiment does not claim that machine
learning is more mathematically correct than the WQI5 formula. Instead, it tests
whether surrogate regressors remain useful when deployment constraints make some
indicators unavailable, delayed, noisy, or costly to collect.

This reframes the machine-learning component as a deployable WQI5 surrogate
assessment layer for small-scale IoT scenarios.

## Scenarios

The configured scenarios are:

| Scenario | Features | Deployment interpretation |
| --- | --- | --- |
| `full` | `DO`, `BOD`, `NH3N`, `EC`, `SS` | Complete-input reference setting |
| `no_bod` | `DO`, `NH3N`, `EC`, `SS` | BOD unavailable or delayed |
| `no_nh3n` | `DO`, `BOD`, `EC`, `SS` | NH3N unavailable or delayed |
| `no_bod_no_nh3n` | `DO`, `EC`, `SS` | Reduced sensing without BOD and NH3N |
| `low_cost_core` | `DO`, `EC`, `SS` | Low-cost sensing subset |

`direct_wqi5` is reported only for the complete-input scenario because the
deterministic WQI5 formula requires the full indicator set.

## Run

```bash
python scripts/reproduce_reduced_indicators.py \
  --config configs/reduced_indicator_config.yaml \
  --output-dir results/reduced_indicator_run
```

GPU execution can be enabled for supported gradient boosting models:

```bash
python scripts/reproduce_reduced_indicators.py \
  --config configs/reduced_indicator_config.yaml \
  --output-dir results/reduced_indicator_gpu_run \
  --compute-device gpu \
  --gpu-id 0
```

The script refuses to overwrite existing CSV results unless `--overwrite` is
passed explicitly.

## Outputs

- `reduced_indicator_results.csv`: per-seed, per-scenario model metrics.
- `reduced_indicator_summary.csv`: mean and standard deviation across seeds.
- `reduced_indicator_category_metrics.csv`: category-stratified MAE and RMSE.
- `best_surrogate_by_scenario.csv`: lowest-mean-MAE surrogate for each scenario,
  excluding `direct_wqi5`.

## Interpretation

The full-input experiment evaluates approximation to the reference WQI5 score
when the deterministic formula can be applied. The reduced-indicator scenarios
evaluate whether a surrogate model can retain practical utility when the formula
cannot be directly applied because part of the indicator set is unavailable.
