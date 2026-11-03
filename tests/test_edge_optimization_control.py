"""Deadlines terminate workers and retain failures without touching API containers."""

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import edge_optimization_control as control
from edge_optimization_control import run_bounded, supervise_container


def test_deadline_kills_sigterm_ignoring_worker_and_keeps_log(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('loaded',flush=True); time.sleep(60)"
    path = tmp_path / "execution.json"
    with (tmp_path / "worker.log").open("w") as log:
        result = run_bounded(
            [sys.executable, "-c", code],
            timeout_seconds=0.5,
            grace_seconds=0.1,
            log=log,
            evidence=path,
        )
    assert result["phase"] == "timed_out"
    assert result["returncode"] == 124
    assert result["worker_returncode"] == -signal.SIGKILL
    assert result["termination"] == "confirmed"
    assert (tmp_path / "worker.log").read_text() == "loaded\n"
    assert json.loads(path.read_text())["phase"] == "timed_out"
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(result["pid"], 0)


def test_descendant_ignoring_term_is_killed_after_leader_exits(tmp_path):
    import os

    ready = tmp_path / "descendant.pid"
    descendant = (
        "import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(60)"
    )
    # The leader keeps default TERM handling, so only a group SIGKILL can stop
    # the TERM-ignoring descendant left in the dedicated session.
    leader = "\n".join(
        [
            "import os, subprocess, sys, time",
            f"subprocess.Popen([sys.executable, '-c', {descendant!r}, {str(ready)!r}])",
            f"while not os.path.exists({str(ready)!r}):",
            "    time.sleep(0.01)",
            "time.sleep(60)",
        ]
    )
    with (tmp_path / "worker.log").open("w") as log:
        result = run_bounded(
            [sys.executable, "-c", leader],
            timeout_seconds=3,
            grace_seconds=1,
            log=log,
            evidence=tmp_path / "execution.json",
        )
    assert result["phase"] == "timed_out"
    assert result["worker_returncode"] == -signal.SIGTERM
    assert result["termination"] == "confirmed"
    pid = int(ready.read_text())  # Written only after TERM was being ignored.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_failed_initial_evidence_write_still_kills_worker(tmp_path, monkeypatch):
    import os

    writes = []

    def failing_first_save(path, value):
        writes.append(dict(value))
        if len(writes) == 1:
            raise OSError("No space left on device")
        control.atomic_json(path, value)

    monkeypatch.setattr(control, "save", failing_first_save)
    with (tmp_path / "worker.log").open("w") as log:
        with pytest.raises(OSError, match="No space"):
            run_bounded(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout_seconds=60,
                grace_seconds=1,
                log=log,
                evidence=tmp_path / "execution.json",
            )
    final = json.loads((tmp_path / "execution.json").read_text())
    assert final["phase"] == "controller_interrupted"
    assert final["termination"] == "confirmed"
    with pytest.raises(ProcessLookupError):
        os.kill(writes[0]["pid"], 0)


def test_cleanup_waits_are_bounded_for_unkillable_worker(tmp_path, monkeypatch):
    import subprocess

    launched = []
    reap = subprocess.Popen.wait

    class NeverExits(subprocess.Popen):
        """Simulates a worker stuck in an uninterruptible kernel wait."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            launched.append(self)

        def wait(self, timeout=None):
            assert timeout is not None, "controller must never wait without bound"
            raise subprocess.TimeoutExpired(self.args, timeout)

    monkeypatch.setattr(control.subprocess, "Popen", NeverExits)
    try:
        with (tmp_path / "worker.log").open("w") as log:
            result = run_bounded(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout_seconds=0.2,
                grace_seconds=0.2,
                log=log,
                evidence=tmp_path / "execution.json",
            )
    finally:
        for proc in launched:
            reap(proc, timeout=10)  # Reap the real, already SIGKILLed child.
    assert result["phase"] == "timed_out" and result["returncode"] == 124
    assert result["termination"] == "unconfirmed"
    assert json.loads((tmp_path / "execution.json").read_text())["termination"] == (
        "unconfirmed"
    )


def test_normal_exit_is_completed_not_timeout(tmp_path):
    with (tmp_path / "worker.log").open("w") as log:
        result = run_bounded(
            [sys.executable, "-c", "print('done')"],
            timeout_seconds=5,
            log=log,
            evidence=tmp_path / "execution.json",
        )
    assert result["phase"] == "completed" and result["returncode"] == 0


class FakeRemote:
    def __init__(self):
        self.calls = []
        self.running = True
        self.token = None
        self.now = 0

    def sleep(self, seconds):
        self.now += seconds

    def docker(self, args):
        self.calls.append(args)
        if args[0] == "create":
            self.token = args[args.index("--label") + 1].split("=", 1)[1]
        if args[0] == "stop":
            self.running = False
        if args[0] == "inspect":
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        {
                            "Config": {
                                "Labels": {"watermirror.optimization": self.token}
                            },
                            "State": {
                                "Running": self.running,
                                "ExitCode": 137,
                                "OOMKilled": False,
                            },
                        }
                    ]
                ),
                stderr="",
            )
        return SimpleNamespace(stdout="retained logs", stderr="")


def test_remote_deadline_stops_only_owned_research_and_retains_evidence(tmp_path):
    remote = FakeRemote()
    result = supervise_container(
        remote,
        name="wm-opt-test",
        create_args=["eval-image"],
        command=["python", "worker.py"],
        timeout_seconds=1,
        evidence=tmp_path / "run.json",
        clock=lambda: remote.now,
        sleep=remote.sleep,
    )
    assert result["phase"] == "timed_out"
    assert result["state"]["OOMKilled"] is False  # 137 alone is not OOM.
    assert ["stop", "--time", "15", "wm-opt-test"] in remote.calls
    assert not any(args[0] == "rm" for args in remote.calls)
    assert "timeout" in remote.calls[0]  # Deadline survives controller/SSH failure.
    assert (tmp_path / "run.container.log").read_text() == "retained logs\n"


def test_production_container_name_is_rejected_before_docker_call(tmp_path):
    remote = FakeRemote()
    with pytest.raises(ValueError, match="research"):
        supervise_container(
            remote,
            name="watermirror-edge",
            create_args=[],
            command=[],
            timeout_seconds=1,
            evidence=tmp_path / "run.json",
        )
    assert not remote.calls


def test_container_ownership_change_never_stops_replacement(tmp_path):
    class ChangedOwner(FakeRemote):
        def docker(self, args):
            response = super().docker(args)
            if args[0] == "inspect":
                items = json.loads(response.stdout)
                items[0]["Config"]["Labels"]["watermirror.optimization"] = "another-run"
                response.stdout = json.dumps(items)
            return response

    remote = ChangedOwner()
    with pytest.raises(RuntimeError, match="ownership changed"):
        supervise_container(
            remote,
            name="wm-opt-test",
            create_args=["eval-image"],
            command=["python"],
            timeout_seconds=1,
            evidence=tmp_path / "run.json",
        )
    assert not any(args[0] == "stop" for args in remote.calls)
    assert (
        json.loads((tmp_path / "run.json").read_text())["phase"] == "controller_error"
    )
