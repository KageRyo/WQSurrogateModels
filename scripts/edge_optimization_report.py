"""Export immutable observation tables; never recalculates or overwrites old experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

from edge_optimization import all_jobs, job_id, sha
from edge_optimization_paths import artifact_path

MISSING_MODES = ("inference_dropout", "reduced_retraining", "indicator_reconstruction")
SOURCES = ("internal_test", "external_10714")
ROW_KEYS = ["seed", "trees", "missing", "mode", "source"]


def recorded(directory, name, field):
    """Return a consumed file only if it matches its completion-time SHA-256."""
    record = json.loads((directory / "completed.json").read_text())
    digest = record.get(field, {}).get(name)
    path = directory / name
    if digest is None:
        raise ValueError(f"Completion record does not cover {path}")
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"Completed output changed: {path}")
    return path


def frozen_config(root, config_path):
    """Load the experiment matrix only from the config frozen in identity.json."""
    identity = json.loads((root / "identity.json").read_text())
    if sha(config_path) not in identity["source_sha256"].values():
        raise ValueError("Config differs from the experiment's frozen identity")
    return yaml.safe_load(Path(config_path).read_text())


def destination_path(root, destination):
    """Confine a new export; never write into data/models or source evidence."""
    destination = artifact_path(destination)
    root = Path(root).resolve()
    if (destination == root or root in destination.parents) and (
        root / "reports" not in destination.parents
    ):
        raise ValueError("Exports inside the result tree must be under reports/")
    if destination.exists():
        raise FileExistsError(destination)
    return destination


def job_metrics(root, config, partial):
    """Read only hash-verified metrics, each matching its job and the matrix."""
    frames = []
    # Name order keeps re-exports byte-identical to the archived glob order.
    for job in sorted(all_jobs(config), key=lambda job: job_id(**job)):
        directory = root / "jobs" / job_id(**job)
        if not (directory / "completed.json").is_file():
            if partial:
                continue
            raise ValueError(f"Incomplete experiment matrix: {directory.name}")
        frame = pd.read_csv(recorded(directory, "metrics.csv", "sha256"))
        modes = ("full_reference",) if job["missing"] == "complete" else MISSING_MODES
        expected = {
            (job["seed"], job["trees"], job["missing"], mode, source)
            for mode in modes
            for source in SOURCES
        }
        observed = list(frame[ROW_KEYS].itertuples(index=False, name=None))
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f"Metrics do not match the job matrix: {directory.name}")
        frames.append(frame)
    if not frames:
        raise ValueError("No completed metrics")
    return pd.concat(frames, ignore_index=True)


def tables(root, config, partial=False):
    raw = job_metrics(root, config, partial)
    keys = ["seed", "missing", "mode", "source"]
    baseline = raw[raw.trees == 300][keys + ["mae", "accuracy"]].rename(
        columns={"mae": "baseline_mae", "accuracy": "baseline_accuracy"}
    )
    paired = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
    paired["relative_mae_change"] = paired.mae / paired.baseline_mae - 1
    paired["accuracy_change_percentage_points"] = (
        paired.accuracy - paired.baseline_accuracy
    ) * 100
    native = []
    for p in sorted((root / "nano-benchmarks").glob("*/result/result.json")):
        if not (p.parents[1] / "completed.json").is_file():
            continue
        d = json.loads(
            recorded(p.parents[1], "result/result.json", "output_sha256").read_text()
        )
        for t in d["timing"]:
            native.append(
                dict(
                    job=p.parents[1].name,
                    kind=d["kind"],
                    mode=d["mode"],
                    rows=t["rows"],
                    p50_ms=t["p50_ms"],
                    p95_ms=t["p95_ms"],
                    load_s=d["process_cold_load_s"],
                    peak_rss_kib=d["process_peak_rss_kib"],
                    model_bytes=d["model_bytes"],
                    max_abs_difference=d["max_abs_difference"],
                    above_1e_6=d["above_1e_6"],
                )
            )
    onnx = []
    for p in sorted((root / "onnx").glob("*/result.json")):
        d = json.loads(p.read_text())
        for r in d["parity"]:
            for name, v in r["parity"].items():
                onnx.append(
                    dict(
                        job=p.parent.name,
                        source=r["source"],
                        comparison=name,
                        native_bytes=d["native_bytes"],
                        onnx_bytes=d["onnx_bytes"],
                        **v,
                    )
                )
    return {
        "metrics_by_seed": raw,
        "paired_comparison": paired,
        "nano_measurements": pd.DataFrame(native),
        "onnx_parity": pd.DataFrame(onnx),
    }


def export(root, destination, config_path, partial=False):
    root = Path(root).resolve()
    destination = destination_path(root, destination)
    result = tables(root, frozen_config(root, config_path), partial)
    destination.mkdir(parents=True)
    for name, frame in result.items():
        frame.to_csv(destination / (name + ".csv"), index=False)
    notes = [
        (
            "Status",
            "Partial observation snapshot"
            if partial
            else "RF matrix completed; inspect Nano blockers and parity findings separately",
        ),
        (
            "Sources",
            "../jobs/*/metrics.csv; ../nano-benchmarks/*/result/result.json; ../onnx/*/result.json",
        ),
        (
            "Nature",
            "Read-only scientific observation export, not an editable calculation workbook. CSVs are the authoritative machine-readable tables.",
        ),
        (
            "Comparison",
            "Newly trained matched 300-tree baseline, same split and condition. No production model promotion.",
        ),
        (
            "Data",
            "50,000 row source, stratified 80/20, seeds 0–4; final 10,714 rows external. Previously examined data, not a pristine holdout.",
        ),
        (
            "Missing",
            "BOD, NH3N, both; dropout, reduced retraining, in-sample indicator reconstruction as in the original protocol.",
        ),
        (
            "Precision",
            "Native float64 and ONNX float32 compared separately; absolute threshold 1e-6 unchanged. Exceedances retained.",
        ),
        (
            "Timing",
            "x86 timings collected during shared-host jobs. Nano uses a one-CPU, 1100 MiB RAM / 3500 MiB RAM+swap container, alongside the unchanged API.",
        ),
        (
            "Memory",
            "Whole-job workstation high-water includes training; Nano high-water is per worker. Conversion and inference memory are separate.",
        ),
        (
            "Limitations",
            "WQI5 formula and dataset Score disagree. Synthetic 107-window stress is not real events or time forecasting. Category equality is not numerical equivalence.",
        ),
        (
            "Uncertainty",
            "Five matched splits, not five independent datasets. Paired descriptive changes do not establish statistical significance.",
        ),
        (
            "Backup",
            "A local E-drive copy does not imply remote Nextcloud verification or environment restoration.",
        ),
    ]
    workbook = destination / "edge-optimization-summary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(notes, columns=["Item", "Explanation"]).to_excel(
            writer, sheet_name="Readme", index=False
        )
        for name, frame in result.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for row in sheet:
                for cell in row:
                    cell.font = Font(name="Arial", size=10, bold=cell.row == 1)
                    if cell.row == 1:
                        cell.fill = PatternFill("solid", fgColor="DCE6F1")
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(
                    80, max(14, max(len(str(c.value or "")) for c in col[:30]) + 2)
                )
    checked = load_workbook(workbook, read_only=True, data_only=False)
    for sheet in checked:
        for row in sheet:
            if any(c.data_type == "e" for c in row):
                raise ValueError("Workbook contains error cells")
    checked.close()
    count = len(result["metrics_by_seed"])
    (destination / "edge-optimization-results.md").write_text(
        "# Edge optimization — "
        + ("進度快照" if partial else "結果彙整")
        + "\n\n"
        + f"已收錄 {count} 筆 dataset/condition 指標。完整規模為 80 個 RF 工作、200 個條件、400 筆指標；Nano 排程 42 個 seed-0 推論工作。\n\n"
        + "ONNX parity 超過 1e-6、轉換 OOM、Nano 阻擋與健康檢查失敗皆須保留，不可用完成數掩蓋。完整證據在上層 jobs、onnx、nano-benchmarks。\n\n"
        + "|項目|說明|\n|---|---|\n"
        + "\n".join("|" + k + "|" + v.replace("|", "/") + "|" for k, v in notes)
        + "\n"
    )
    return {k: len(v) for k, v in result.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    print(json.dumps(export(args.root, args.destination, args.config, args.partial)))


if __name__ == "__main__":
    main()
