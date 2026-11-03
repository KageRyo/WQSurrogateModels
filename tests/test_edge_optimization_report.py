"""The professor-facing export must only publish verified, complete, confined data."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_optimization import all_jobs, job_id, sha
from edge_optimization_report import MISSING_MODES, SOURCES, export

CONFIG = {
    "seeds": [0],
    "trees": [50, 300],
    "missing_sets": {"missing_bod": ["BOD"]},
}


def write_metrics(directory, frame):
    frame.to_csv(directory / "metrics.csv", index=False)
    (directory / "completed.json").write_text(
        json.dumps({"sha256": {"metrics.csv": sha(directory / "metrics.csv")}})
    )


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "results" / "edge_optimization_test"
    # The repository config differs, so the archived provenance copy must match.
    config = root / "provenance" / "source" / "configs" / "edge_optimization.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(CONFIG))
    (root / "identity.json").write_text(
        json.dumps({"source_sha256": {str(config): sha(config)}})
    )
    for job in all_jobs(CONFIG):
        directory = root / "jobs" / job_id(**job)
        directory.mkdir(parents=True)
        modes = ["full_reference"] if job["missing"] == "complete" else MISSING_MODES
        write_metrics(
            directory,
            pd.DataFrame(
                [
                    dict(**job, mode=mode, source=source, n=10, mae=1.0, accuracy=0.9)
                    for mode in modes
                    for source in SOURCES
                ]
            ),
        )
    return root, config


def test_modified_completed_metrics_are_not_exported(experiment):
    root, _ = experiment
    assert export(root, root / "reports" / "first")["metrics_by_seed"] == 16
    edited = root / "jobs" / "s0-rf50-complete" / "metrics.csv"
    edited.write_text(edited.read_text().replace("1.0", "0.5"))
    with pytest.raises(ValueError, match="Completed output changed"):
        export(root, root / "reports" / "second")
    assert not (root / "reports" / "second").exists()


def drop_completion(root, config):
    (root / "jobs" / "s0-rf300-missing_bod" / "completed.json").unlink()


def duplicate_row(root, config):
    # Re-recorded hashes make the CSV verify; its keys must still fit the job.
    directory = root / "jobs" / "s0-rf300-missing_bod"
    frame = pd.read_csv(directory / "metrics.csv")
    write_metrics(directory, pd.concat([frame, frame.iloc[[0]]]))


def relabel_job(root, config):
    directory = root / "jobs" / "s0-rf50-complete"
    write_metrics(directory, pd.read_csv(directory / "metrics.csv").assign(trees=300))


def shrink_matrix(root, config):
    config.write_text(yaml.safe_dump({**CONFIG, "trees": [50]}))


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (drop_completion, "Incomplete experiment matrix"),
        (duplicate_row, "do not match the job matrix"),
        (relabel_job, "do not match the job matrix"),
        (shrink_matrix, "matches the frozen identity"),
    ],
)
def test_export_requires_the_frozen_complete_matrix(experiment, corrupt, message):
    root, config = experiment
    corrupt(root, config)
    with pytest.raises(ValueError, match=message):
        export(root, root / "reports" / "final")
    assert not (root / "reports" / "final").exists()


def test_partial_snapshot_may_omit_unfinished_jobs(experiment):
    root, config = experiment
    drop_completion(root, config)
    counts = export(root, root / "reports" / "progress", partial=True)
    assert counts["metrics_by_seed"] == 10


def symlinked_parent(root, tmp_path):
    protected = tmp_path / "unrelated"
    protected.mkdir()
    (root / "reports").symlink_to(protected, target_is_directory=True)
    return root / "reports" / "final", protected


@pytest.mark.parametrize(
    "target, message",
    [
        (lambda root, tmp: (tmp / "models" / "export", None), "input/model"),
        (lambda root, tmp: (root / "jobs" / "export", None), "under reports/"),
        (lambda root, tmp: (root, None), "under reports/"),
        (symlinked_parent, "symbolic"),
    ],
)
def test_export_destination_is_confined(experiment, tmp_path, target, message):
    root, _ = experiment
    destination, protected = target(root, tmp_path)
    with pytest.raises(ValueError, match=message):
        export(root, destination)
    assert not (tmp_path / "models").exists()
    assert not (root / "jobs" / "export").exists()
    if protected is not None:
        assert list(protected.iterdir()) == []
