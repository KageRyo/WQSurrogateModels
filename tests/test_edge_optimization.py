"""Concrete integrity and whole-pipeline conversion regressions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from edge_optimization import (
    FEATURES,
    all_jobs,
    forest,
    metrics,
    sha,
    verified,
    write_json,
)


def test_resume_rejects_modified_completed_payload(tmp_path):
    payload = tmp_path / "prediction.npz"
    payload.write_bytes(b"original prediction")
    write_json(
        tmp_path / "completed.json", {"sha256": {"prediction.npz": sha(payload)}}
    )
    assert verified(tmp_path)
    payload.write_bytes(b"modified prediction")
    with pytest.raises(ValueError, match="Completed output changed"):
        verified(tmp_path)


def test_matrix_finishes_dependencies_before_missing_jobs():
    config = {
        "seeds": [0, 1],
        "trees": [50, 300],
        "missing_sets": {"missing_bod": ["BOD"]},
    }
    seen = set()
    for item in all_jobs(config):
        if item["missing"] != "complete":
            assert (item["seed"], item["trees"]) in seen
        else:
            seen.add((item["seed"], item["trees"]))


def test_nonfinite_predictions_cannot_be_completed():
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics(np.array([10.0, 20.0]), np.array([np.nan, 20.0]))


def test_native_worker_reports_prediction_and_timing_progress(tmp_path):
    import json
    import subprocess
    import joblib

    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.uniform(1, 20, size=(128, len(FEATURES))), columns=FEATURES
    )
    model = forest(
        {"forest_random_state": 0, "training_jobs": 1}, 3, frame, frame.DO * 2
    )
    joblib.dump({"wqi_model": model}, tmp_path / "model.joblib")
    np.savez_compressed(
        tmp_path / "inputs.npz",
        X=frame.to_numpy(),
        columns=np.array(FEATURES),
        reference=model.predict(frame),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "edge_optimization_benchmark.py"),
            "--kind",
            "native",
            "--model",
            str(tmp_path / "model.joblib"),
            "--inputs",
            str(tmp_path / "inputs.npz"),
            "--output",
            str(tmp_path / "result"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(
        e["phase"] == "chunk_completed" and e["completed_rows"] == 128 for e in events
    )
    assert any(e["phase"] == "timing" and e["completed_repeats"] > 0 for e in events)
    assert (
        json.loads((tmp_path / "result/progress.json").read_text())["phase"]
        == "completed"
    )
    assert json.loads((tmp_path / "result/result.json").read_text())["above_1e_6"] == 0


@pytest.mark.parametrize(
    "request_args",
    [
        ["--mode", "inference_dropout", "--missing", "BOD"],
        ["--missing", "BOD", "NH3N"],
    ],
)
def test_onnx_worker_refuses_missing_value_requests(tmp_path, request_args):
    import subprocess

    # The ONNX session would run complete inputs while recording these labels.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "edge_optimization_benchmark.py"),
            "--kind",
            "onnx",
            "--model",
            str(tmp_path / "pipeline.onnx"),
            "--inputs",
            str(tmp_path / "inputs.npz"),
            "--output",
            str(tmp_path / "result"),
            *request_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "ONNX evaluation supports only" in result.stderr
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "mode, missing",
    [("full_reference", ["BOD"]), ("reduced_retraining", [])],
)
def test_recorded_mode_and_missing_labels_must_agree(mode, missing):
    from edge_optimization_benchmark import validate_request

    with pytest.raises(ValueError, match="missing"):
        validate_request("native", mode, missing)


def test_nano_wrapper_rejects_onnx_missing_request_before_remote_setup(monkeypatch):
    import edge_optimization_nano as nano

    def no_remote(*args, **kwargs):
        raise AssertionError("remote output must not be created")

    monkeypatch.setattr(nano, "BoundedRemote", no_remote)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edge_optimization_nano.py",
            "--host=nano",
            "--control=control",
            "--name=wm-opt-test",
            "--image=example/research:tag",
            "--packet=/srv/edge-optimization-test/packet",
            "--worker=/srv/edge-optimization-test/worker",
            "--remote-output=/srv/edge-optimization-test/output",
            "--evidence=/tmp/edge-optimization-test-evidence.json",
            "--timeout-seconds=60",
            "--kind=onnx",
            "--model-filename=pipeline.onnx",
            "--mode=indicator_reconstruction",
            "--missing",
            "BOD",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        nano.main()
    assert stopped.value.code == 2
