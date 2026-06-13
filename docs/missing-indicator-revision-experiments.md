# Missing-Indicator Revision Experiments

This workflow evaluates WQI5 surrogate behavior under missing-indicator and
deployment-constrained settings. It is designed for reviewer-response analysis,
not temporal water-quality forecasting.

## Data Split

The experiment uses:

- `data/dataV1_50000.csv` as the training and internal-test source.
- `data/dataV1.csv` rows after the first 50,000 records as the external
  inference set.

The script checks that `dataV1_50000.csv` is an exact prefix of `dataV1.csv`.
The external inference set therefore contains `10,714` rows.

For each seed, `dataV1_50000.csv` is split with stratified `80/20` sampling by
WQI5 category:

- train: `40,000` rows
- internal test: `10,000` rows
- external inference: `10,714` rows

## Experiments

| Experiment | Purpose |
| --- | --- |
| `full_reference` | Complete five-indicator model evaluated with complete five-indicator input. |
| `full_inference_dropout` | Complete five-indicator model evaluated with `BOD` and `NH3N` set to missing at inference time. |
| `reduced_retraining` | Model trained and evaluated using only `DO`, `EC`, and `SS`. |
| `two_stage_reconstruction` | Reconstructs `BOD` and `NH3N` from `DO`, `EC`, and `SS`, then evaluates WQI5 with a full-input surrogate. |
| `stress_scenarios` | Scenario-based perturbation tests for pollution-like parameter shifts. |

`full_inference_dropout` and `full_reference` use the same trained full-input
model. The dropout condition simulates unexpected deployment-time loss of BOD
and NH3N measurements.

The two-stage reconstruction workflow trains the indicator-reconstruction models
only on the training split. The second-stage WQI surrogate is trained on the
training rows after `BOD` and `NH3N` have been reconstructed, so the
second-stage model sees the same reconstructed-feature structure during
training and evaluation.

```text
DO / EC / SS -> BOD
DO / EC / SS -> NH3N
DO / reconstructed BOD / reconstructed NH3N / EC / SS -> WQI5 surrogate
```

## Run

```bash
python scripts/run_revision_missing_indicator_experiments.py \
  --config configs/revision_missing_indicator_config.yaml \
  --output-dir results_revision_missing_indicators_20260613_gpu \
  --compute-device gpu \
  --gpu-id 0
```

The script refuses to overwrite an output directory that already contains files
unless `--overwrite` is passed explicitly.

## Outputs

The output directory contains:

- `manifest.json`: run configuration and dataset summary.
- `hardware.json`: local hardware snapshot.
- `versions.json`: library versions.
- `models/`: saved joblib model artifacts by seed, experiment, and model.
- `splits/split_indices.csv`: train/test split row indices for each seed.
- `predictions/predictions_long.csv`: row-level internal-test and external
  inference predictions.
- `metrics/metrics_by_seed.csv`: per-seed metrics.
- `metrics/metrics_summary.csv`: mean and standard deviation across seeds.
- `metrics/best_by_experiment_source.csv`: lowest-mean-MAE model per experiment
  and source.
- `metrics/stage1_reconstruction_metrics.csv`: BOD and NH3N reconstruction
  metrics for the two-stage workflow.
- `metrics/error_by_wqi_band.csv`: WQI-band error summaries.
- `stats/bootstrap_ci.csv`: row-level bootstrap confidence intervals.
- `stats/paired_error_tests.csv`: paired Wilcoxon absolute-error comparisons
  with Holm correction.
- `stress_tests/stress_summary.csv`: scenario-based stress-test response
  summaries.

## Interpretation Boundary

Complete-input WQI5 computation remains the reference method when all five
indicators are available. These experiments evaluate whether surrogate models
can provide useful auxiliary estimates under missing indicators, delayed
measurements, indicator reconstruction, or controlled stress-test scenarios.
