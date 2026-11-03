"""Export observed extension results to new CSV/Markdown/XLSX snapshots."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from edge_optimization import categories, sha, write_json, utc
from edge_optimization_paths import artifact_path


def markdown_table(frame):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(map(cell, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(map(cell, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = artifact_path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    expected_missing = {
        f"s0-rf50-{condition}-{mode}"
        for condition in ["missing_bod", "missing_nh3n", "missing_bod_nh3n"]
        for mode in [
            "inference_dropout",
            "reduced_retraining",
            "indicator_reconstruction",
        ]
    }
    expected_families = {"lr", "mpr", "svm", "lightgbm", "xgboost"}
    for name, expected in [
        ("x86", expected_missing),
        ("nano", expected_missing),
        ("families", expected_families),
        ("nano-families", expected_families),
    ]:
        actual = {p.name for p in (args.run / name).iterdir() if p.is_dir()}
        if actual != expected:
            raise ValueError("Incomplete or unexpected final matrix: " + name)
    parity = []
    metrics = []
    nano = []
    health = []
    coverage = []
    for base, isfamily in [("x86", False), ("families", True)]:
        for directory in sorted((args.run / base).iterdir()):
            if not directory.is_dir():
                continue
            p = directory / "result.json"
            if not p.exists():
                coverage.append(
                    dict(
                        scope=base,
                        job=directory.name,
                        outcome="no_completed_result",
                        source=str(directory),
                    )
                )
                continue
            result = json.loads(p.read_text())
            coverage.append(
                dict(
                    scope=base,
                    job=directory.name,
                    outcome=result["phase"],
                    source=str(p),
                    source_sha256=sha(p),
                )
            )
            rows = (
                [
                    dict(
                        source="external_10714",
                        parity=result["parity"],
                        onnx_metrics=result["onnx_metrics"],
                    )
                ]
                if isfamily
                else result["parity"]
            )
            for row in rows:
                for comparison, value in row["parity"].items():
                    parity.append(
                        dict(
                            job=directory.name,
                            source=row["source"],
                            comparison=comparison,
                            **value,
                        )
                    )
                metrics.append(
                    dict(
                        job=directory.name, source=row["source"], **row["onnx_metrics"]
                    )
                )
    for base, isfamily in [("nano", False), ("nano-families", True)]:
        for directory in sorted((args.run / base).iterdir()):
            if not directory.is_dir():
                continue
            terminal = directory / "terminal.json"
            if not terminal.exists():
                raise ValueError(
                    "Cannot export final report with unfinished " + str(directory)
                )
            outcome = json.loads(terminal.read_text())["phase"]
            coverage.append(
                dict(
                    scope=base,
                    job=directory.name,
                    outcome=outcome,
                    source=str(terminal),
                    source_sha256=sha(terminal),
                )
            )
            polls = (
                json.loads((directory / "health-poll.json").read_text())
                if (directory / "health-poll.json").exists()
                else []
            )
            if isfamily:
                network_failed = sum(not p["network_ok"] for p in polls)
                loopback_failed = sum(not p["loopback_ok"] for p in polls)
                transport_unknown = sum("loopback_error" in p for p in polls)
            else:
                network_failed = sum(not p["network"]["ok"] for p in polls)
                loopback_failed = sum(not p["loopback"]["ok"] for p in polls)
                transport_unknown = sum("error" in p["loopback"] for p in polls)
            health.append(
                dict(
                    job=directory.name,
                    polls=len(polls),
                    network_failed=network_failed,
                    loopback_or_transport_failed=loopback_failed,
                    loopback_transport_unknown=transport_unknown,
                    scope="Inference only;5-second paired probes, transport errors do not prove loopback HTTP failure",
                )
            )
            if outcome != "completed":
                continue
            result = json.loads((directory / "output/result/result.json").read_text())
            reference = (
                args.run
                / ("families" if isfamily else "x86")
                / directory.name
                / "external_10714.npz"
            )
            saved = np.load(reference, allow_pickle=False)
            predicted = np.load(
                directory / "output/result/predictions.npz", allow_pickle=False
            )["predicted"]
            delta = np.abs(predicted - saved["native64"])
            onnxdelta = np.abs(predicted - saved["onnx32"])
            for timing in result["timing"]:
                nano.append(
                    dict(
                        job=directory.name,
                        batch_rows=timing["rows"],
                        p50_ms=timing["p50_ms"],
                        p95_ms=timing["p95_ms"],
                        load_seconds=result.get("process_cold_load_s"),
                        peak_rss_mib=result.get(
                            "peak_process_rss_kib", result.get("process_peak_rss_kib")
                        )
                        / 1024,
                        rows=len(predicted),
                        native_above_1e_6=int((delta > 1e-6).sum()),
                        native_max_difference=float(delta.max()),
                        native_category_changes=int(
                            (
                                categories(saved["native64"]) != categories(predicted)
                            ).sum()
                        ),
                        x86_onnx_above_1e_6=int((onnxdelta > 1e-6).sum()),
                        x86_onnx_max_difference=float(onnxdelta.max()),
                    )
                )
    stage = json.loads((args.run / "stage-analysis/summary.json").read_text())
    stage_rows = []
    for row in stage:
        stage_rows.append(
            dict(
                condition=row["condition"],
                rows=row["rows"],
                large_difference_rows=row["large_difference_rows"],
                branch_evidence_rows=row["large_difference_rows_with_branch_evidence"],
                reconstruction_score_max_difference=row[
                    "native_reconstruction_to_onnx_reconstruction"
                ]["max_abs_score_difference"],
                fixed_reconstruction_final_stage_max_difference=row[
                    "same_onnx_reconstruction_native_to_onnx_final_stage"
                ]["max_abs_difference"],
            )
        )
    frames = {
        "Coverage": pd.DataFrame(coverage),
        "X86Parity": pd.DataFrame(parity),
        "X86Metrics": pd.DataFrame(metrics),
        "Nano": pd.DataFrame(nano),
        "Health": pd.DataFrame(health),
        "StageIsolation": pd.DataFrame(stage_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(out / (name + ".csv"), index=False)
    notes = [
        [
            "Scope",
            "RF50 seed0,3 missing settings x3 modes; full-input LR/MPR/SVM/LightGBM/XGBoost. No retraining or production promotion.",
        ],
        [
            "Protocol",
            "Tolerance1e-6;5-column input; unavailable values ignored within composed graph.",
        ],
        [
            "Nano",
            "1CPU,1100MiB RAM,3500MiB RAM+swap,1200s deadline; checksum-warmed caches; shared API; CPU only.",
        ],
        [
            "Timing",
            "300 consecutive warm samples per batch1/128. Not independent cold starts.",
        ],
        [
            "Observed values",
            "Immutable measurement export; CSV source tables authoritative. No editable assumptions or recomputation model.",
        ],
        [
            "Runtime",
            "ONNX1.23.2,ORT1.31.1,skl2onnx1.20.0,onnxmltools1.16.0 for non-RF families.",
        ],
        [
            "Archive boundary",
            "Previous610 and20261101 results unchanged; new results in separate dated root.",
        ],
        [
            "Provenance",
            "Individual source.json/result.json and terminal manifest hashes retained; source_snapshot contains executed pre-format source.",
        ],
        [
            "Limits",
            "Seed0 screen,existing external set reused. Other missing-family combinations and larger missing RF ONNX graphs not evaluated here.",
        ],
        [
            "Backup",
            "Separate local E-drive copy; remote Nextcloud readback/restore is not established by that copy.",
        ],
    ]
    book = Workbook()
    book.remove(book.active)
    ws = book.create_sheet("ReadMe")
    ws.append(["Item", "Description"])
    for row in notes:
        ws.append(row)
    for name, frame in frames.items():
        ws = book.create_sheet(name)
        ws.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            ws.append(list(row))
    for ws in book:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for row in ws:
            for cell in row:
                cell.font = Font(name="Arial", size=10)
        for cell in ws[1]:
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24536C")
        for column in ws.columns:
            letter = column[0].column_letter
            ws.column_dimensions[letter].width = min(
                75, max(14, max(len(str(c.value or "")) for c in column) + 2)
            )
    path = out / "ONNX_Extension_20261103.xlsx"
    book.save(path)
    reopened = load_workbook(path, data_only=True)
    assert all(c.data_type != "e" for ws in reopened for row in ws for c in row)
    report = [
        "# ONNX extension actual evaluation —2026-11-03",
        "",
        f"Generated UTC:{utc()}",
        "",
        "Frozen observations; read the ReadMe sheet for protocol and limitations.",
        "",
    ]
    for name, frame in frames.items():
        report.extend(["## " + name, "", markdown_table(frame), ""])
    (out / "report.md").write_text("\n".join(report))
    write_json(
        out / "manifest.json",
        dict(
            utc=utc(),
            sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
            tables={name: len(frame) for name, frame in frames.items()},
            workbook_sheets=reopened.sheetnames,
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(out),
                tables={name: len(frame) for name, frame in frames.items()},
            )
        )
    )


if __name__ == "__main__":
    main()
