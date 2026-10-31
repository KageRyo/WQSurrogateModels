"""Create a new verified experiment backup without merging into older backups."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from edge_replay.worker import exclusive_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def copy_verified(source, target):
    source = Path(source)
    target = Path(target)
    digest = sha256(source)
    shutil.copy2(source, target)
    if source.stat().st_size != target.stat().st_size or sha256(target) != digest:
        raise ValueError("Backup copy differs from its source: " + str(source))
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--record-parity-failures", action="store_true", help="Back up a complete experiment including explicitly recorded failed parity findings")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    report_directory = args.report.resolve()
    destination = args.destination.resolve()
    temporary = destination.with_name(destination.name + ".partial").resolve()
    sources = [experiment, report_directory]
    if args.evidence:
        sources.append(args.evidence.resolve())
    for source in sources:
        for output in (destination, temporary):
            if output == source or source in output.parents or output in source.parents:
                raise ValueError("Backup destination overlaps a source directory: " + str(source))
    # copytree dereferences nested links by default. A link back to an output
    # ancestor would recursively copy the new partial backup into itself. Reject
    # links before creating any output; never traverse directory links here.
    for source in sources:
        for directory, directories, files in os.walk(source, followlinks=False):
            for name in directories + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("Backup source contains a symbolic link: " + str(Path(directory) / name))
    report = json.loads((report_directory / "manifest.json").read_text())
    runner = json.loads((experiment / "replay-results/runner-complete.json").read_text())
    findings_allowed = (args.record_parity_failures and report.get("parity_findings_recorded") is True
                        and runner.get("parity_findings_recorded") is True
                        and set(report.get("parity_difference_jobs", [])) == set(runner.get("parity_difference_jobs", [])))
    quality_agrees = (runner.get("all_parity_pass") is report["all_parity_pass"]
                      and set(report.get("parity_difference_jobs", [])) == set(runner.get("parity_difference_jobs", [])))
    if (report["all_parity_pass"] is not True and not findings_allowed) or report["verified_jobs"] != runner["jobs"] or not quality_agrees:
        raise ValueError("Final experiment verification is incomplete")
    original = json.loads((experiment / "provenance/original_x86_results_sha256.json").read_text())
    for record in original:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archived result path escapes the repository")
        path = ROOT / relative
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("An archived x86 result changed: " + record["path"])
    if destination.exists() or temporary.exists():
        raise FileExistsError("Choose a new backup name; existing backups are never overwritten")
    temporary.mkdir(parents=True)
    # The private packet/input/model outputs belong in the backup, never Git.
    shutil.copytree(experiment, temporary / "experiments", copy_function=copy_verified)
    # Reports may be generated outside the experiment tree.
    try:
        report_directory.relative_to(experiment)
    except ValueError:
        shutil.copytree(report_directory, temporary / "report", copy_function=copy_verified)
    # Preserve the archived x86 tables/splits used to establish the comparison.
    for record in original:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archived result path escapes the repository")
        target = temporary / "original-x86" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(ROOT / relative, target)
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    owned_new = ["Dockerfile.jetson", "Dockerfile.edge-research", "requirements-edge-research.in",
                 "requirements-edge-research.txt", "scripts/benchmark_edge.py", "scripts/replay_edge_experiments.py",
                 "docs/jetson-nano.md", "docs/jetson-experiment-replay.md", "tests/test_edge_replay.py"]
    owned_new += [str(p.relative_to(ROOT)) for p in (ROOT / "scripts/edge_replay").glob("*.py")]
    owned_new += [str(p.relative_to(ROOT)) for p in (ROOT / "deploy/systemd").glob("watermirror-*.*")]
    owned_new += [str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_edge_*.py")]
    for relative in sorted(set(tracked + owned_new)):
        if not relative or relative.startswith(("data/", "results/")):
            continue
        source = ROOT / relative
        if not source.is_file() or (source.name.startswith(".env") and source.name != ".env.example"):
            continue
        target = temporary / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(source, target)
    for relative in ["dataV1.csv", "dataV1_1000.csv", "dataV1_5000.csv", "dataV1_10000.csv", "dataV1_50000.csv"]:
        target = temporary / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(ROOT / "data" / relative, target)
    if args.evidence:
        selected = temporary / "deployment-evidence"
        selected.mkdir()
        for source in args.evidence.iterdir():
            if any(token in source.name.lower() for token in ("password", "credential", "secret", "token", ".env")):
                continue
            if source.is_file() and source.suffix.lower() in {".json", ".xml", ".png", ".txt", ".log", ".sh", ".md"}:
                copy_verified(source, selected / source.name)
    git = {
        "branch":subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"],text=True).strip(),
        "head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        "status":subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True),
    }
    exclusive_json(temporary / "backup-metadata.json", {
        "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "git":git,
        "verified_jobs":report["verified_jobs"], "original_x86_results_unchanged":True,
        "all_prediction_parity_pass":report["all_parity_pass"],
        "parity_difference_jobs":report.get("parity_difference_jobs", []),
        "model_scope":"Lossless replay packet components are included; archived original model bundles are referenced by source hash, not duplicated.",
        "scope":"Private dated experiment snapshot. Dataset and model copies are deliberately excluded from Git publication.",
        "sync_note":"This manifest verifies the requested filesystem backup. It does not independently prove completion of an external Nextcloud/NAS synchronization.",
    })
    manifest = []
    for path in sorted(temporary.rglob("*")):
        if path.is_file():
            manifest.append({"path":str(path.relative_to(temporary)), "bytes":path.stat().st_size, "sha256":sha256(path)})
    exclusive_json(temporary / "SHA256.json", manifest)
    # Re-read the destination files after copy, including large private packets.
    for record in manifest:
        path = temporary / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Backup verification failed: " + record["path"])
    temporary.rename(destination)
    print(json.dumps({"backup":str(destination),"files":len(manifest),"verified":True}),flush=True)
