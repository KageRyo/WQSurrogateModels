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
