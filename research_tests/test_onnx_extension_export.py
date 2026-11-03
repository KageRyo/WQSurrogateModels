"""Final exports must reject incomplete, altered or mislabeled evidence before writing."""

from pathlib import Path
import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_optimization import sha
from export_onnx_extension_report import export

CONDITIONS = ("missing_bod", "missing_nh3n", "missing_bod_nh3n")
MODES = ("inference_dropout", "reduced_retraining", "indicator_reconstruction")
FAMILIES = ("lr", "mpr", "svm", "lightgbm", "xgboost")
COMPARISONS = ("native64_to_native32", "native32_to_onnx32", "native64_to_onnx32")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reseal_nano(directory):
    write_json(
        directory / "terminal.json",
        dict(
            phase="completed",
            files={
                name: sha(directory / name)
                for name in [
                    "container.json",
                    "health-poll.json",
                    "output/result/result.json",
                    "output/result/predictions.npz",
                ]
            },
        ),
    )


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "results" / "experiment"
    metrics = dict(mae=0.0, rmse=0.0, r2=1.0, accuracy=1.0, macro_f1=1.0)
    values = np.full(10714, 49.0)
    for scope, jobs in [
        ("x86", [f"s0-rf50-{c}-{m}" for c in CONDITIONS for m in MODES]),
        ("families", FAMILIES),
    ]:
        for job in jobs:
            directory = root / scope / job
            directory.mkdir(parents=True)
            payload = directory / "external_10714.npz"
            np.savez_compressed(payload, native64=values, onnx32=values)
            if scope == "x86":
                np.savez_compressed(
                    directory / "internal_test.npz",
                    native64=np.full(10000, 49.0),
                    onnx32=np.full(10000, 49.0),
                )
            rows = []
            for source in (
                ["external_10714"]
                if scope == "families"
                else ["internal_test", "external_10714"]
            ):
                n = 10714 if source == "external_10714" else 10000
                parity = {
                    c: dict(
                        rows=n,
                        above_tolerance=0,
                        tolerance=1e-6,
                        max_abs_difference=0.0,
                        mean_abs_difference=0.0,
                        category_changes=0,
                    )
                    for c in COMPARISONS
                }
                rows.append(dict(source=source, parity=parity, onnx_metrics=metrics))
            result = dict(
                phase="completed" if scope == "families" else "completed_parity_pass",
                parity=rows[0]["parity"] if scope == "families" else rows,
                onnx_metrics=metrics,
                sha256={"external_10714.npz": sha(payload)},
            )
            if scope == "x86":
                result["sha256"]["internal_test.npz"] = sha(
                    directory / "internal_test.npz"
                )
            write_json(directory / "result.json", result)
            write_json(
                directory.parent / (job + "-control.json"),
                dict(phase="completed", returncode=0),
            )
            nano = root / ("nano-families" if scope == "families" else "nano") / job
            target = nano / "output/result"
            target.mkdir(parents=True)
            np.savez_compressed(target / "predictions.npz", predicted=values)
            write_json(
                target / "result.json",
                dict(
                    process_cold_load_s=1.0,
                    peak_process_rss_kib=1024,
                    timing=[dict(rows=n, p50_ms=0.1, p95_ms=0.2) for n in [1, 128]],
                ),
            )
            write_json(nano / "health-poll.json", [])
            write_json(
                nano / "container.json",
                dict(phase="completed", state=dict(ExitCode=0, OOMKilled=False)),
            )
            reseal_nano(nano)
    write_json(
        root / "stage-analysis/summary.json",
        [
            dict(
                condition=c,
                rows=10714,
                large_difference_rows=0,
                large_difference_rows_with_branch_evidence=0,
                native_reconstruction_to_onnx_reconstruction=dict(
                    max_abs_score_difference=0.0
                ),
                same_onnx_reconstruction_native_to_onnx_final_stage=dict(
                    max_abs_difference=0.0
                ),
            )
            for c in CONDITIONS
        ],
    )
    return root


def test_complete_verified_evidence_exports_all_tables(run):
    out = run / "reports" / "accepted"
    assert export(run, out) == dict(
        Coverage=28, X86Parity=69, X86Metrics=23, Nano=28, Health=14, StageIsolation=3
    )
    nano = pd.read_csv(out / "Nano.csv")
    assert not nano.duplicated(["job", "batch_rows"]).any()
    assert set(nano.native_max_difference) == {0.0}


@pytest.mark.parametrize(
    "fault",
    [
        "missing_internal_prediction",
        "missing_x86_result",
        "missing_x86_control",
        "failed_x86_control",
        "failed_x86_result",
        "changed_x86_payload",
        "missing_nano_terminal",
        "failed_nano_terminal",
        "missing_nano_result",
        "changed_nano_prediction",
        "missing_completion_hash",
        "failed_nano_container",
    ],
)
def test_incomplete_or_changed_completion_cannot_create_final_report(run, fault):
    x86 = run / "families/lr"
    nano = run / "nano-families/lr"
    if fault == "missing_internal_prediction":
        (
            run / "x86/s0-rf50-missing_bod-indicator_reconstruction/internal_test.npz"
        ).unlink()
    elif fault == "missing_x86_result":
        (x86 / "result.json").unlink()
    elif fault == "missing_x86_control":
        (x86.parent / "lr-control.json").unlink()
    elif fault == "failed_x86_control":
        write_json(x86.parent / "lr-control.json", dict(phase="failed", returncode=1))
    elif fault == "failed_x86_result":
        value = json.loads((x86 / "result.json").read_text())
        value["phase"] = "failed"
        write_json(x86 / "result.json", value)
    elif fault == "changed_x86_payload":
        (x86 / "external_10714.npz").write_bytes(b"changed")
    elif fault == "missing_nano_terminal":
        (nano / "terminal.json").unlink()
    elif fault == "failed_nano_terminal":
        write_json(nano / "terminal.json", dict(phase="oom"))
    elif fault == "missing_nano_result":
        (nano / "output/result/result.json").unlink()
    elif fault == "changed_nano_prediction":
        (nano / "output/result/predictions.npz").write_bytes(b"changed")
    elif fault == "missing_completion_hash":
        write_json(nano / "terminal.json", dict(phase="completed", files={}))
    elif fault == "failed_nano_container":
        write_json(
            nano / "container.json",
            dict(phase="completed", state=dict(ExitCode=137, OOMKilled=True)),
        )
        reseal_nano(nano)
    out = run / "reports" / "must-not-exist"
    with pytest.raises(ValueError):
        export(run, out)
    assert not out.parent.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_source",
        "missing_comparison",
        "wrong_row_count",
        "duplicate_timing",
        "missing_timing",
        "duplicate_stage",
        "truncated_predictions",
    ],
)
def test_resealed_but_incomplete_or_duplicate_rows_are_rejected(run, fault):
    x86 = run / "x86/s0-rf50-missing_bod-indicator_reconstruction"
    nano = run / "nano-families/lr"
    if fault in ["duplicate_source", "missing_comparison", "wrong_row_count"]:
        result = json.loads((x86 / "result.json").read_text())
        if fault == "duplicate_source":
            result["parity"][1] = copy.deepcopy(result["parity"][0])
        elif fault == "missing_comparison":
            del result["parity"][0]["parity"]["native64_to_native32"]
        else:
            result["parity"][0]["parity"]["native64_to_native32"]["rows"] = 9999
        write_json(x86 / "result.json", result)
    elif fault in ["duplicate_timing", "missing_timing"]:
        result = json.loads((nano / "output/result/result.json").read_text())
        if fault == "duplicate_timing":
            result["timing"][1] = copy.deepcopy(result["timing"][0])
        else:
            result["timing"].pop()
        write_json(nano / "output/result/result.json", result)
        reseal_nano(nano)
    elif fault == "duplicate_stage":
        stage = json.loads((run / "stage-analysis/summary.json").read_text())
        stage[1] = copy.deepcopy(stage[0])
        write_json(run / "stage-analysis/summary.json", stage)
    else:
        np.savez_compressed(
            nano / "output/result/predictions.npz", predicted=np.array([49.0])
        )
        reseal_nano(nano)
    out = run / "reports" / "must-not-exist"
    with pytest.raises(ValueError):
        export(run, out)
    assert not out.parent.exists()


@pytest.mark.parametrize(
    "relative",
    [
        "x86/export",
        "nano/export",
        "families/export",
        "nano-families/export",
        "stage-analysis/export",
        ".",
    ],
)
def test_destination_cannot_write_into_evidence(run, relative):
    out = run / relative
    before = {str(p.relative_to(run)) for p in run.rglob("*")}
    with pytest.raises(ValueError, match="under reports/"):
        export(run, out)
    assert {str(p.relative_to(run)) for p in run.rglob("*")} == before


def test_symlinked_reports_cannot_redirect_export(run, tmp_path):
    target = tmp_path / "protected"
    target.mkdir()
    (run / "reports").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        export(run, run / "reports/final")
    assert not list(target.iterdir())


def test_completed_parity_and_health_findings_are_preserved(run):
    # Numerical/health findings are valid observations, not failed execution.
    # The strict final gate must retain these instead of requiring parity pass.
    x86 = run / "x86/s0-rf50-missing_bod-indicator_reconstruction"
    result = json.loads((x86 / "result.json").read_text())
    result["phase"] = "completed_with_findings"
    write_json(x86 / "result.json", result)
    nano = run / "nano-families/lr"
    np.savez_compressed(
        nano / "output/result/predictions.npz", predicted=np.full(10714, 49.001)
    )
    write_json(
        nano / "health-poll.json",
        [dict(network_ok=False, loopback_ok=False, loopback_error="TimeoutExpired")],
    )
    reseal_nano(nano)
    out = run / "reports" / "findings"
    export(run, out)
    row = pd.read_csv(out / "Nano.csv").query("job == 'lr' and batch_rows == 1").iloc[0]
    assert row.native_above_1e_6 == 10714
    health = pd.read_csv(out / "Health.csv").set_index("job").loc["lr"]
    assert health.network_failed == 1
    assert health.loopback_transport_unknown == 1
