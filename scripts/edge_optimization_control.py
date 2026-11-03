"""Bounded research workers. Timeout is a terminal finding, never success."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Deadline must be finite and positive")
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_bounded(command, *, timeout_seconds, log, evidence, grace_seconds=15):
    """Terminate the entire dedicated process group, including stuck native code."""
    timeout_seconds = positive_seconds(timeout_seconds)
    grace_seconds = positive_seconds(grace_seconds)
    started = time.monotonic()
    proc = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    result = dict(
        phase="running",
        pid=proc.pid,
        deadline_seconds=timeout_seconds,
        started_epoch=time.time(),
    )
    save(evidence, result)
    try:
        try:
            proc.wait(timeout=timeout_seconds)
            result.update(
                phase="completed" if proc.returncode == 0 else "failed",
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            result.update(
                phase="timed_out", returncode=124, worker_returncode=proc.returncode
            )
    except BaseException:
        # A controller interruption must not leave an unbounded research child.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        result.update(phase="controller_interrupted", returncode=proc.returncode)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save(evidence, result)
    return result


def supervise_container(
    remote,
    *,
    name,
    create_args,
    command,
    timeout_seconds,
    evidence,
    grace_seconds=15,
    poll_seconds=2,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Create and supervise ONLY a new named research container; retain it on exit.

    The remote image must contain coreutils timeout. Its independent deadline
    survives a dropped SSH/controller; the supervisor also records stop evidence.
    Remote transport operations must themselves have a bounded timeout.
    """
    import uuid
    import re

    if not re.fullmatch(r"wm-opt-[A-Za-z0-9_.-]+", name):
        raise ValueError("Only dedicated wm-opt-* research container names are allowed")
    seconds = positive_seconds(timeout_seconds)
    grace = positive_seconds(grace_seconds)
    if poll_seconds <= 0:
        raise ValueError("Polling interval must be positive")
    token = uuid.uuid4().hex
    started = clock()
    record = dict(
        phase="creating", name=name, owner_token=token, deadline_seconds=seconds
    )
    save(evidence, record)
    # Docker create fails on name collisions. Never reuse/replace a container.
    remote.docker(
        [
            "create",
            "--name",
            name,
            "--label",
            "watermirror.optimization=" + token,
            *create_args,
            "timeout",
            "--signal=TERM",
            "--kill-after=" + str(grace) + "s",
            str(seconds) + "s",
            *command,
        ]
    )

    def inspect_owned():
        item = json.loads(remote.docker(["inspect", name]).stdout)[0]
        if item["Config"].get("Labels", {}).get("watermirror.optimization") != token:
            raise RuntimeError(
                "Research container ownership changed; refuse to stop it"
            )
        return item

    def stop_owned():
        if inspect_owned()["State"]["Running"]:
            remote.docker(["stop", "--time", str(math.ceil(grace)), name])

    try:
        remote.docker(["start", name])
        while True:
            item = inspect_owned()
            if not item["State"]["Running"]:
                state = item["State"]
                code = state["ExitCode"]
                record.update(
                    phase="timed_out"
                    if code == 124
                    else "oom"
                    if state.get("OOMKilled")
                    else "completed"
                    if code == 0
                    else "failed",
                    state=state,
                )
                break
            elapsed = clock() - started
            if elapsed >= seconds:
                stop_owned()
                record.update(phase="timed_out", state=inspect_owned()["State"])
                break
            record.update(phase="running", elapsed_seconds=elapsed)
            save(evidence, record)
            sleep(min(poll_seconds, max(0.01, seconds - elapsed)))
    except BaseException as exc:
        record.update(phase="controller_error", error=type(exc).__name__)
        try:
            stop_owned()
        except Exception as cleanup:
            record["stop_error"] = str(cleanup)
        raise
    finally:
        # Save state before fetching logs so transport failures cannot erase it.
        record["elapsed_seconds"] = clock() - started
        save(evidence, record)
        try:
            logs = remote.docker(["logs", name])
            Path(evidence).with_suffix(".container.log").write_text(
                logs.stdout + "\n" + logs.stderr
            )
        except Exception as exc:
            record["log_error"] = str(exc)
            save(evidence, record)
    return record


def main():
    """A bounded entry point for standalone conversion/report subprocesses."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a worker command after --")
    positive_seconds(args.timeout_seconds)
    if args.evidence.exists() or args.log.exists():
        raise FileExistsError("Use new execution evidence and log paths")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x") as log:
        result = run_bounded(
            command,
            timeout_seconds=args.timeout_seconds,
            log=log,
            evidence=args.evidence,
        )
    print(json.dumps(result))
    raise SystemExit(
        0
        if result["phase"] == "completed"
        else 124
        if result["phase"] == "timed_out"
        else 1
    )


if __name__ == "__main__":
    main()
