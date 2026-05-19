# Statistical Analysis

This document separates descriptive feature-score analysis from predictive model reliability analysis.

## Feature-Score Correlation

Pearson and Spearman correlation coefficients can be computed between each raw water-quality indicator and the calculated WQI5 `Score`:

- `DO` vs `Score`
- `BOD` vs `Score`
- `NH3N` vs `Score`
- `EC` vs `Score`
- `SS` vs `Score`

This analysis is descriptive because `Score` is constructed from the same five indicators.

It should therefore be interpreted as:

- a summary of how the raw indicators relate to the constructed WQI5 index
- not causal evidence
- not evidence of temporal prediction ability

## Model Reliability Analysis

Model reliability should be evaluated on held-out or otherwise unseen prediction outputs using metrics such as:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Residual Mean`
- `Residual Std.`
- `Mean Relative Accuracy`
- `Std. of Relative Accuracy`
- `Category Accuracy`
- `Macro-F1`

Residuals are defined as:

```text
residual_i = y_i - ŷ_i
```

where `y_i` is the reference WQI5 score and `ŷ_i` is the model-estimated WQI5 score.

## Reviewer-Safe Interpretation

- Feature-score correlation answers: "How do the raw indicators relate to the constructed WQI5 score?"
- Reliability analysis answers: "How closely do surrogate models reproduce held-out WQI5 scores?"

These two analyses should not be conflated.
