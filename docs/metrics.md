# Metrics

This repository models `Score` as a regression target. Therefore, the term `accuracy` must not be interpreted as classification accuracy unless explicitly labeled as category accuracy.

## Regression Metrics

- `R²`: coefficient of determination
- `MAE`: mean absolute error
- `RMSE`: root mean squared error
- `NMAE`: normalized mean absolute error
- `Residual Mean`: mean of `y_i - ŷ_i`
- `Residual Std.`: standard deviation of `y_i - ŷ_i`

Residuals are defined as:

```text
residual_i = y_i - ŷ_i
```

where `y_i` is the reference WQI5 score and `ŷ_i` is the model-estimated WQI5 score.

## Relative Accuracy

For archived manuscript experiments, relative accuracy should be defined as:

```text
Relative Accuracy_i (%) = (1 - |y_i - ŷ_i| / y_i) * 100
```

Therefore:

- `Mean Relative Accuracy (%)` is the average sample-wise relative accuracy.
- `Std. of Relative Accuracy (%)` is the standard deviation of sample-wise relative accuracy.

When `y_i > 0`, this metric is equivalent to:

```text
Mean Relative Accuracy (%) = 100% - MAPE(%)
```

This metric should be reported together with `R²`, `MAE`, and `RMSE`, not as a replacement for them.

## Category Metrics

If predicted continuous scores are converted into WQI5 categories, the following classification-style metrics may also be reported:

- `Category Accuracy`
- `Macro-F1`

These metrics evaluate agreement after mapping continuous scores into category bands. They do not convert the main task into a native classification problem.

## Reporting Guidance

- Use `regression metrics` when discussing continuous WQI5 score estimation.
- Use `category metrics` only when explicitly evaluating post-hoc category agreement.
- Avoid using the unlabeled word `accuracy` by itself in revised manuscripts. Prefer `mean relative accuracy` or `category accuracy`.
