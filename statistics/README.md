# Statistics Workspace

This directory mixes three different kinds of assets:

## Track In Git

- `statistical_analysis_from_xlsx.py`
  - main statistical analysis script
  - reads `整理.xlsx` and `data/*.csv`
  - writes derived tables into `statistics/outputs/`
- `../scripts/reproduce_holdout_10714.py`
  - reconstructs the `10714` hold-out split as `dataV1.csv - dataV1_50000.csv`
  - verifies that the hold-out `Score` matches the Excel `10714筆測試` sheet
  - reproduces hold-out predictions for the saved `50000` artifacts that exist in `models/`
- `README.md`
  - this inventory and tracking policy

## Do Not Track In Git

- `整理.xlsx`
  - local experiment-result workbook
  - treated as a result record rather than source code or repository documentation
- `outputs/`
  - all files in `outputs/` are generated artifacts
  - examples:
    - `statistical_analysis_outputs.xlsx`
    - `*_metrics.csv`
    - `holdout_reproduction/*.csv`

These files are useful for local inspection and result packaging, but they should be regenerated rather than versioned in normal source control.

## Reproduction Flow

Generate the statistical analysis package:

```bash
python statistics/statistical_analysis_from_xlsx.py
```

If a narrative markdown report is needed, prepare it separately from the generated CSV/XLSX outputs rather than generating it directly from the public analysis script.

Reconstruct and validate the `10714` hold-out set:

```bash
python scripts/reproduce_holdout_10714.py
```

## Notes

- The `10714` hold-out rows are the tail rows of `data/dataV1.csv` after the first `50000` rows contained in `data/dataV1_50000.csv`.
- The hold-out source rows are reproducible from committed CSV files.
- Hold-out reconstruction and prediction-check workflows are provided by the committed scripts in this repository.
