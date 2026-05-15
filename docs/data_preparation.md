# Data Preparation

`WQSurrogateModels` now frames the task as `WQI5-based assessment`, not time-series forecasting.

## Scope

- The processed dataset shipped in this repository is [`data/dataV1.csv`](/mnt/8tb_hdd/ryo/WQSurrogateModels/data/dataV1.csv) with `60,714` rows and `6` columns: `DO`, `BOD`, `NH3N`, `EC`, `SS`, `Score`.
- The manuscript discussion refers to an upstream pool of `87,005` records before project-level cleaning and filtering. That raw upstream source is not currently versioned in this repository, so this document records the intended audit trail and the remaining gap explicitly.
- The current repository dataset does not contain timestamps. Validation must therefore be described as `cross-sectional`, not temporal forecasting.

## Current Repository State

- Current processed rows: `60,714`
- Missing-value ratio in the committed processed dataset:
  - `DO`: `0.0%`
  - `BOD`: `0.0%`
  - `NH3N`: `0.0%`
  - `EC`: `0.0%`
  - `SS`: `0.0%`
  - `Score`: `0.0%`
- Available subsets already committed:
  - `100`, `1,000`, `5,000`, `10,000`, `20,000`, `50,000`

## Required Upstream Provenance

The following items must be preserved in the paper revision and any future raw-data release:

1. How the original `87,005` records were collected.
2. Which records came from Qilintan and which came from MOENV.
3. Whether those sources were merged, aligned, filtered, or synthetically generated.
4. Which exclusion rules reduced the pool to `60,714`.
5. Which records were discarded for invalid values, duplicates, unmatched fields, or QA filtering.

This repository currently documents the processed endpoint only. If raw provenance is added later, extend this file with row counts after each cleaning stage.

## Imputation and Trimming

- Existing training scripts use `mean` imputation. That is acceptable only as a simple baseline because the committed processed dataset is already complete.
- If mean imputation was applied upstream before creating `dataV1.csv`, the rationale should be documented as:
  - It preserves dataset size for surrogate-regression training.
  - It avoids introducing model-specific bias from more aggressive imputation schemes.
- `1%` outlier trimming should be described carefully:
  - It may stabilize regression training.
  - It may also remove true pollution events.
  - It should therefore be treated as a sensitivity-analysis choice, not an unquestioned denoising step.

## Split Protocol

- The revised experiment protocol uses stratified random splitting by WQI5 category.
- No temporal independence is claimed.
- Recommended seeds: `0, 1, 2, 3, 4`
- Recommended report format: `mean ± std`

## Subset Sampling

For the paper revision, the subset-generation rule should be stated explicitly:

1. Start from the processed `60,714`-row dataset.
2. Derive WQI5 categories from `Score`.
3. Sample each subset with fixed random seeds.
4. Preserve category proportions as closely as possible.

If new subset files are regenerated, store the generation script alongside this document.
