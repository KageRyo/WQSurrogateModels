from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from edge_replay.finalize import finalize


@pytest.mark.parametrize("mode", ["success", "failed_aggregate", "wrong_findings", "incomplete"])
def test_finalization_never_publishes_failed_or_incomplete_report(tmp_path, monkeypatch, mode):
    staging, results, output = tmp_path / "staging", tmp_path / "results", tmp_path / "reports/final"
    staging.mkdir()
    results.mkdir()
    (staging / "preparation-complete.json").write_text(json.dumps({"jobs": 2}))
    (results / "runner-complete.json").write_text(json.dumps({
        "phase": "all_jobs_complete", "jobs": 1 if mode == "incomplete" else 2,
        "all_parity_pass": False, "parity_difference_jobs": ["unstable-lr"],
    }))
    calls = []

    def aggregate(command, **kwargs):
        calls.append(command)
        report = Path(command[command.index("--output") + 1])
        report.mkdir()
        (report / "metrics.csv").write_text("retained evidence")
        if mode == "failed_aggregate":
            raise subprocess.CalledProcessError(1, command)
        (report / "manifest.json").write_text(json.dumps({
            "verified_jobs": 2, "all_parity_pass": False,
            "parity_difference_jobs": [] if mode == "wrong_findings" else ["unstable-lr"],
        }))

    monkeypatch.setattr(subprocess, "run", aggregate)
    if mode == "success":
        finalize(staging, results, output, record_parity_failures=True)
        assert (output / "metrics.csv").read_text() == "retained evidence"
        with pytest.raises(FileExistsError):
            finalize(staging, results, output, record_parity_failures=True)
        assert len(calls) == 1
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            finalize(staging, results, output, record_parity_failures=True)
        assert not output.exists()
        if mode == "incomplete":
            assert calls == []
        else:
            assert list(output.parent.glob("aggregation-attempt-*/report/metrics.csv"))
