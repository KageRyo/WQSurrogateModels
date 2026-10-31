"""Private backup completeness and integrity, using only tiny fixture files."""
import json
from pathlib import Path
import sys

import pytest


def test_backup_preserves_external_report_originals_and_untracked_units(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay import backup

    root = tmp_path / "repo"
    def write(relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    original = write("results/old/metrics.csv", "score\n1\n")
    write("experiment/provenance/original_x86_results_sha256.json", json.dumps([
        {"path": "results/old/metrics.csv", "bytes": original.stat().st_size,
         "sha256": backup.sha256(original)}]))
    write("experiment/replay-results/runner-complete.json", json.dumps({"jobs": 1, "all_parity_pass": True}))
    write("experiment/environment/hardware.json", '{"cpu":"fixture"}')
    write("experiment/packets/job/model.joblib", "private model")
    write("report/manifest.json", json.dumps({"verified_jobs": 1, "all_parity_pass": True}))
    write("report/summary.csv", "verified\ntrue\n")
    for name in ("dataV1.csv", "dataV1_1000.csv", "dataV1_5000.csv", "dataV1_10000.csv", "dataV1_50000.csv"):
        write("data/" + name, "fixture\n")
    write("deploy/systemd/watermirror-replay.service", "[Service]\n")
    write("deploy/systemd/watermirror-replay.timer", "[Timer]\n")
    write("evidence/health.json", '{}')
    write("evidence/nano-password.txt", 'sensitive')
    destination = tmp_path / "snapshot"
    monkeypatch.setattr(backup, "ROOT", root)
    def git_output(command, **kwargs):
        return "fixture" if kwargs.get("text") else b""
    monkeypatch.setattr(backup.subprocess, "check_output", git_output)
    monkeypatch.setattr(sys, "argv", ["backup", "--experiment", str(root / "experiment"),
        "--report", str(root / "report"), "--evidence", str(root / "evidence"),
        "--destination", str(destination)])
    backup.main()
    manifest = json.loads((destination / "SHA256.json").read_text())
    paths = {item["path"] for item in manifest}
    assert {"report/summary.csv", "original-x86/results/old/metrics.csv",
            "source/deploy/systemd/watermirror-replay.service",
            "source/deploy/systemd/watermirror-replay.timer",
            "experiments/environment/hardware.json", "experiments/packets/job/model.joblib",
            "deployment-evidence/health.json"} <= paths
    assert not (destination / "deployment-evidence/nano-password.txt").exists()
    for item in manifest:
        assert backup.sha256(destination / item["path"]) == item["sha256"]
    assert original.read_text() == "score\n1\n"
    with pytest.raises(FileExistsError):
        backup.main()


def test_copy_verified_refuses_corrupted_copy(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay import backup
    source = tmp_path / "input"
    source.write_text("original")
    monkeypatch.setattr(backup.shutil, "copy2", lambda source, target: Path(target).write_text("damaged!"))
    with pytest.raises(ValueError, match="differs"):
        backup.copy_verified(source, tmp_path / "output")


@pytest.mark.parametrize("source_name", ["experiment", "report", "evidence"])
@pytest.mark.parametrize("relationship", ["same", "inside", "ancestor", "partial_inside", "symlink"])
def test_backup_rejects_overlapping_sources_before_creating_files(tmp_path, monkeypatch, source_name, relationship):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay import backup

    sources = {name: tmp_path / name for name in ("experiment", "report", "evidence")}
    for source in sources.values():
        source.mkdir()
    source = sources[source_name]
    if relationship == "same":
        destination = source
    elif relationship == "inside":
        destination = source / "nested" / "snapshot"
    elif relationship == "ancestor":
        destination = tmp_path
    elif relationship == "partial_inside":
        sources[source_name] = source / "snapshot.partial" / "input"
        sources[source_name].mkdir(parents=True)
        destination = source / "snapshot"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        destination = alias / "snapshot"
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}
    monkeypatch.setattr(sys, "argv", ["backup", "--experiment", str(sources["experiment"]),
        "--report", str(sources["report"]), "--evidence", str(sources["evidence"]),
        "--destination", str(destination)])
    with pytest.raises(ValueError, match="overlaps a source"):
        backup.main()
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")} == before
