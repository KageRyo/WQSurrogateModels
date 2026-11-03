"""Export observed extension results to new CSV/Markdown/XLSX snapshots."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from edge_optimization import categories, sha, write_json, utc
from edge_optimization_report import destination_path


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


def verify_file(directory, name, recorded):
    path = directory / name
    digest = recorded.get(name)
    if digest is None or not path.is_file() or sha(path) != digest:
        raise ValueError("Completed output missing or changed: " + str(path))


def completed_x86(root, scope, directory):
    controller = root / scope / (directory.name + "-control.json")
    result_path = directory / "result.json"
    if not controller.is_file() or not result_path.is_file():
        raise ValueError("Incomplete x86 completion: " + str(directory))
    control = json.loads(controller.read_text())
    if control.get("phase") != "completed" or control.get("returncode") != 0:
        raise ValueError("Unsuccessful x86 controller: " + str(directory))
    result = json.loads(result_path.read_text())
    allowed = (
        {"completed"}
        if scope == "families"
        else {"completed_parity_pass", "completed_with_findings"}
    )
    if result.get("phase") not in allowed:
        raise ValueError("Unsuccessful x86 result: " + str(directory))
    names = (
        ["external_10714.npz"]
        if scope == "families"
        else ["internal_test.npz", "external_10714.npz"]
    )
    for name in names:
        verify_file(directory, name, result.get("sha256", {}))
    return result


def validate_keys(name, rows, columns, expected):
    actual = [tuple(row.get(column) for column in columns) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(name + " rows do not match the complete unique job matrix")


def export(run, output):
    root = Path(run).resolve()
    out = destination_path(root, output)
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
        actual = {p.name for p in (root / name).iterdir() if p.is_dir()}
        if actual != expected:
            raise ValueError("Incomplete or unexpected final matrix: " + name)
    parity = []
    metrics = []
    nano = []
    health = []
    coverage = []
    for base, isfamily in [("x86", False), ("families", True)]:
        for directory in sorted((root / base).iterdir()):
            if not directory.is_dir():
                continue
            p = directory / "result.json"
            result = completed_x86(root, base, directory)
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
        for directory in sorted((root / base).iterdir()):
            if not directory.is_dir():
                continue
            terminal = directory / "terminal.json"
            if not terminal.exists():
                raise ValueError(
                    "Cannot export final report with unfinished " + str(directory)
                )
            completion = json.loads(terminal.read_text())
            outcome = completion.get("phase")
            if outcome != "completed":
                raise ValueError("Unsuccessful Nano completion: " + str(directory))
            for name in [
                "output/result/result.json",
                "output/result/predictions.npz",
                "health-poll.json",
                "container.json",
            ]:
                verify_file(directory, name, completion.get("files", {}))
            container = json.loads((directory / "container.json").read_text())
            state = container.get("state", {})
            if (
                container.get("phase") != "completed"
                or state.get("ExitCode") != 0
                or state.get("OOMKilled") is not False
            ):
                raise ValueError("Unsuccessful Nano container: " + str(directory))
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
            result = json.loads((directory / "output/result/result.json").read_text())
            reference = (
                root
                / ("families" if isfamily else "x86")
                / directory.name
                / "external_10714.npz"
            )
            saved = np.load(reference, allow_pickle=False)
            predicted = np.load(
                directory / "output/result/predictions.npz", allow_pickle=False
            )["predicted"]
            for values in [predicted, saved["native64"], saved["onnx32"]]:
                if values.shape != (10714,) or not np.isfinite(values).all():
                    raise ValueError(
                        "Invalid external prediction rows: " + str(directory)
                    )
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
    stage = json.loads((root / "stage-analysis/summary.json").read_text())
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
    jobs = expected_missing | expected_families
    sources = {"internal_test", "external_10714"}
    comparisons = {"native64_to_native32", "native32_to_onnx32", "native64_to_onnx32"}
    source_keys = {(job, source) for job in expected_missing for source in sources} | {
        (job, "external_10714") for job in expected_families
    }
    validate_keys(
        "Coverage",
        coverage,
        ("scope", "job"),
        {
            (scope, job)
            for scope, expected in [
                ("x86", expected_missing),
                ("nano", expected_missing),
                ("families", expected_families),
                ("nano-families", expected_families),
            ]
            for job in expected
        },
    )
    validate_keys(
        "X86Parity",
        parity,
        ("job", "source", "comparison"),
        {(*key, comparison) for key in source_keys for comparison in comparisons},
    )
    for row in parity:
        expected_rows = 10000 if row["source"] == "internal_test" else 10714
        if row.get("rows") != expected_rows:
            raise ValueError("Invalid X86Parity prediction row count")
    validate_keys("X86Metrics", metrics, ("job", "source"), source_keys)
    validate_keys(
        "Nano",
        nano,
        ("job", "batch_rows"),
        {(job, batch) for job in jobs for batch in [1, 128]},
    )
    validate_keys("Health", health, ("job",), {(job,) for job in jobs})
    validate_keys(
        "StageIsolation",
        stage_rows,
        ("condition",),
        {
            (condition,)
            for condition in ["missing_bod", "missing_nh3n", "missing_bod_nh3n"]
        },
    )
    frames = {
        "Coverage": pd.DataFrame(coverage),
        "X86Parity": pd.DataFrame(parity),
        "X86Metrics": pd.DataFrame(metrics),
        "Nano": pd.DataFrame(nano),
        "Health": pd.DataFrame(health),
        "StageIsolation": pd.DataFrame(stage_rows),
    }
    out.mkdir(parents=True)
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
    if any(c.data_type == "e" for ws in reopened for row in ws for c in row):
        raise ValueError("Workbook contains error cells")
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
    return {name: len(frame) for name, frame in frames.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = export(args.run, args.output)
    print(json.dumps(dict(output=str(args.output), tables=counts)))


if __name__ == "__main__":
    main()
