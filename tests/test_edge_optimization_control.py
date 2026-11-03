"""Deadlines terminate workers and retain failures without touching API containers."""

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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
    assert (tmp_path / "worker.log").read_text() == "loaded\n"
    assert json.loads(path.read_text())["phase"] == "timed_out"
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(result["pid"], 0)


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
