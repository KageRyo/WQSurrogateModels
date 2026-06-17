# Original Benchmark Protocol

This document describes the archived benchmark style reflected by the original
training scripts under `archive/legacy_training/`.

## Archived Subset Usage

The archived experiments used predefined fixed-size subsets such as:

- `dataV1_1000.csv`
- `dataV1_10000.csv`
- `dataV1_50000.csv`

These subsets were used for sample-size sensitivity analysis.

## Train-Test Split

For each subset, the archived scripts applied:

- an `80/20` train-test split
- a fixed random seed (`random_state=0` in the archived scripts)

## Hyperparameter Selection

Hyperparameter selection was performed inside the training subset using `5-fold GridSearchCV`.

Examples from the archived scripts include:

- polynomial degree candidates `1..6` for the MPR-style pipeline
- library defaults for some estimators where no explicit search grid was provided

## Scope and Limitations

- The archived benchmark did not enforce temporal independence.
- The archived benchmark did not enforce spatial independence.
- The released dataset does not contain timestamp, station-sequence, or spatial-location identifiers needed to claim temporal or spatial forecasting.

Therefore, the archived results should be interpreted as `cross-sectional WQI5 surrogate-regression` results, not temporal or spatial forecasting results.

## Relationship to the Revised Reproducibility Workflow

The archived benchmark protocol is not identical to the revised reproducibility workflow documented in [experiment_protocol.md](experiment_protocol.md).

The revised workflow adds:

- stratified repeated random splitting by WQI5 category
- explicit seed lists
- consolidated result export files for reproducibility
