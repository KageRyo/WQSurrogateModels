"""Publish a complete aggregation atomically, retaining failed attempts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from edge_replay.runner import acquire_runner_lock


def finalize(staging, results, output, *, record_parity_failures=False):
    staging, results, output = map(Path, (staging, results, output))
    runner = json.loads((results / "runner-complete.json").read_text())
    preparation = json.loads((staging / "preparation-complete.json").read_text())
    if runner["phase"] != "all_jobs_complete" or runner["jobs"] != preparation["jobs"]:
        raise ValueError("The full replay is not complete")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire_runner_lock(output.parent)
    if output.exists():
        raise FileExistsError("Final report already exists; it will not be replaced")
    attempt = Path(tempfile.mkdtemp(prefix="aggregation-attempt-", dir=output.parent))
    report = attempt / "report"
    command = [sys.executable, str(Path(__file__).parents[1] / "replay_edge_experiments.py"),
               "aggregate", "--staging", str(staging.resolve()), "--results", str(results.resolve()),
               "--output", str(report.resolve())]
    if record_parity_failures:
        command.append("--record-parity-failures")
    subprocess.run(command, check=True)
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["verified_jobs"] != runner["jobs"]:
        raise ValueError("Report job count differs from the completed replay")
    if (manifest["all_parity_pass"] is not runner["all_parity_pass"]
            or set(manifest["parity_difference_jobs"]) != set(runner["parity_difference_jobs"])):
        raise ValueError("Report findings differ from the completed replay")
    report.rename(output)
    attempt.rmdir()
    print(json.dumps({"phase": "complete_report_published", "output": str(output),
                      "verified_jobs": runner["jobs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-parity-failures", action="store_true")
    args = parser.parse_args()
    finalize(args.staging, args.results, args.output, record_parity_failures=args.record_parity_failures)
