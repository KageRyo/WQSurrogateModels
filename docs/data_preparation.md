# Data Preparation

`WQSurrogateModels` now frames the task as `WQI5-based assessment`, not time-series forecasting.

## Scope

- The processed dataset shipped in this repository is [`data/dataV1.csv`](/mnt/8tb_hdd/ryo/WQSurrogateModels/data/dataV1.csv) with `60,714` rows and `6` columns: `DO`, `BOD`, `NH3N`, `EC`, `SS`, `Score`.
- The manuscript discussion refers to an upstream pool of `87,005` records before project-level cleaning and filtering. That raw upstream source is not currently versioned in this repository, so this document records the intended audit trail and the remaining gap explicitly.
- The current repository dataset does not contain timestamps. Validation must therefore be described as `cross-sectional`, not temporal forecasting.

## Current Repository State

- Current processed rows: `60,714`
- Current committed features: `DO`, `BOD`, `NH3N`, `EC`, `SS`
- Current committed target: `Score` interpreted as WQI5 score
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

## Paper-Safe Provenance Wording

Until the upstream audit trail is recovered and versioned, the manuscript should use wording such as:

> The released repository provides the processed 60,714-row modeling dataset. The study also references a larger upstream source pool of 87,005 observations, but exact intermediate filtering counts are not yet versioned in the public repository and are therefore documented as a current reproducibility limitation.

Use `merged`, `aligned`, `filtered`, or `processed` unless actual synthetic data generation can be demonstrated. At the current repository state, there is no versioned evidence that the committed modeling dataset was synthetically generated.

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
- With the current default protocol on `60,714` rows and `test_size=0.2`, each repeated split yields:
  - train: `48,571`
  - test: `12,143`

## Remaining Gap

The following details are still not recoverable from the committed repository alone and should remain explicitly acknowledged until the upstream sources are restored:

1. Qilintan row count within the original `87,005` records.
2. MOENV row count within the original `87,005` records.
3. Exact stage-by-stage row counts for field alignment, invalid-value removal, duplicate removal, and any 1% trimming step.
4. Whether any upstream imputation or trimming occurred before the public `dataV1.csv` snapshot was exported.

## Subset Sampling

For the paper revision, the subset-generation rule should be stated explicitly:

1. Start from the processed `60,714`-row dataset.
2. Derive WQI5 categories from `Score`.
3. Sample each subset with fixed random seeds.
4. Preserve category proportions as closely as possible.

If new subset files are regenerated, store the generation script alongside this document.
