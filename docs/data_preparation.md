# Data Preparation

`WQSurrogateModels` now frames the task as `WQI5-based assessment`, not time-series forecasting.

## Scope

- The processed dataset shipped in this repository is [`data/dataV1.csv`](/mnt/8tb_hdd/ryo/WQSurrogateModels/data/dataV1.csv) with `60,714` rows and `6` columns: `DO`, `BOD`, `NH3N`, `EC`, `SS`, `Score`.
- The manuscript discussion refers to an upstream pool of `87,005` records before project-level cleaning and filtering. That raw upstream source is not currently versioned in this repository, so this document records the intended audit trail and the remaining gap explicitly.
- The current repository dataset does not contain timestamps. Validation must therefore be described as `cross-sectional`, not temporal forecasting.

## Source-Level Provenance

At the source level, the current modeling dataset is understood to come from:

1. prior experimental / field-study data associated with earlier water-quality publications by the project team, including:
   - *Using intelligent water purification and ecological conservation system to improve environmental conservation benefits evaluation* (Environmental Technology & Innovation, 2025, DOI: `10.1016/j.eti.2025.104046`)
   - the later study referenced by the project team at `https://www.sciencedirect.com/science/article/abs/pii/S1642359326000200`
2. Ministry of Environment open data, especially river water quality monitoring resources such as:
   - MOENV open data platform: `https://data.moenv.gov.tw/en/dataset/detail/WQX_P_01`
   - Taiwan government open data mirror: `https://data.gov.tw/en/datasets/6078`

This means the repository can now document the source families used to assemble the processed modeling data. However, the repository still does not contain the exact versioned upstream extraction scripts or intermediate row-count logs needed to reconstruct every transition from `87,005` source observations to the committed `60,714`-row modeling table.

## Current Team Explanation for the 87,005 -> 60,714 Reduction

Based on current project-team clarification, the major reduction from the upstream `87,005` source observations to the committed `60,714` modeling rows is attributable to trimming extreme values at both tails during preprocessing, together with dataset alignment / filtering.

- The working explanation is that the upper and lower `1%` extremes (i.e., head and tail 1% trimming) were removed during preprocessing.
- Because trimming can be applied across multiple indicators, the final retained row count should not be interpreted as a simple single-step `80%` keep-rate on one variable.
- In practice, the retained set reflects the combined effect of:
  - source alignment between prior experimental data and government open data
  - validity / preprocessing filters
  - tail trimming of extreme observations

This explanation is sufficient for revision-era methodological transparency, but the repository still lacks a versioned preprocessing script that emits exact row counts after each individual operation.

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
- Head-and-tail `1%` outlier trimming should be described carefully:
  - It may stabilize regression training.
  - It may also remove true pollution events.
  - It should therefore be treated as a sensitivity-analysis choice, not an unquestioned denoising step.

Revision note:

- An earlier version of this document incorrectly stated `10%` tail trimming.
- The correct preprocessing step was **head and tail 1% trimming** (removal of the top 1% and bottom 1% extremes).
- Because the trimming was applied across multiple water quality indicators and combined with alignment/filtering steps, the overall reduction from `87,005` to `60,714` rows is consistent with this 1% per-tail rule.
- Until the original preprocessing script is restored and audited, it is safest to describe the step in manuscripts as `extreme-value tail trimming (head and tail 1%) during preprocessing`.

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
3. Exact stage-by-stage row counts for field alignment, invalid-value removal, duplicate removal, and the head-and-tail 1% trimming step.
4. Whether any upstream imputation or trimming occurred before the public `dataV1.csv` snapshot was exported.

## Subset Sampling

For the paper revision, the subset-generation rule should be stated explicitly:

1. Start from the processed `60,714`-row dataset.
2. Derive WQI5 categories from `Score`.
3. Sample each subset with fixed random seeds.
4. Preserve category proportions as closely as possible.

If new subset files are regenerated, store the generation script alongside this document.
