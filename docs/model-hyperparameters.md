# Model Hyperparameters

This document distinguishes between the revised reproducibility workflow and the archived exploratory scripts.

## Revised Reproducibility Workflow

The current reproducibility workflow uses the following main settings:

| Model | Library | Preprocessing | Key Hyperparameters |
| --- | --- | --- | --- |
| `direct_wqi5` | formula baseline | none | direct WQI5 equation |
| `lr` | scikit-learn | mean imputation + standard scaling | default `LinearRegression()` |
| `mpr` | scikit-learn | mean imputation + polynomial features + standard scaling | `degree=2`, `include_bias=False` |
| `svm` | scikit-learn | mean imputation + standard scaling | `kernel=rbf`, `C=10.0`, `epsilon=0.1` |
| `rf` | scikit-learn | mean imputation | `n_estimators=300`, `random_state=0`, `n_jobs=-1` |
| `xgboost` | xgboost | mean imputation | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `random_state=0` |
| `lightgbm` | lightgbm | mean imputation | `n_estimators=300`, `learning_rate=0.05`, `random_state=0` |

## Archived Exploratory Scripts

The archived training scripts under `src/training/` reflect an earlier exploratory workflow.

Common characteristics include:

- `80/20` train-test splits with `random_state=0`
- `5-fold GridSearchCV`
- polynomial degree candidates from `1` to `6` for MPR-style pipelines
- some model-specific parameters left at library defaults

The archived exploratory scripts should not be assumed to be identical to the revised reproducibility workflow used for future regeneration and robustness checks.
