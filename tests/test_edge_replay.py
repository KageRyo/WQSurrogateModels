from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def test_ssh_recovery_replaces_socket_left_by_killed_controller(tmp_path, monkeypatch):
    import socket
    import subprocess
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.runner import Remote

    control = tmp_path / "ssh-control"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(control))
    sock.close()
    remote = Remote("user@nano", str(control), "sudo-secret", ssh_password="ssh-secret")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 255, "", "Connection refused")
        assert not control.exists(), "A stale socket prevents SSH from creating the replacement master"
        assert "ssh-secret" not in " ".join(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", execute)
    remote.renew_control()
    assert len(calls) == 2


def test_resume_archives_only_incomplete_reference_and_excludes_second_controller(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.runner import acquire_runner_lock, archive_incomplete_reference

    lock = acquire_runner_lock(tmp_path)
    with pytest.raises(RuntimeError, match="Another replay controller"):
        acquire_runner_lock(tmp_path)
    lock.close()
    acquire_runner_lock(tmp_path).close()
    reference = tmp_path / "x86/job"
    reference.mkdir(parents=True)
    (reference / "partial.npz").write_bytes(b"preserve evidence")
    archived = archive_incomplete_reference(reference, tmp_path)
    assert (archived / "partial.npz").read_bytes() == b"preserve evidence"
    assert not reference.exists()
    reference.mkdir()
    (reference / "completed.json").write_text("{}")
    assert archive_incomplete_reference(reference, tmp_path) is None
    assert (reference / "completed.json").exists()


@pytest.mark.parametrize("mode", ["full_reference", "inference_dropout", "reduced_retraining", "indicator_reconstruction"])
def test_isolated_replay_preserves_original_bundle_predictions(tmp_path, monkeypatch, mode):
    import importlib.metadata
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from edge_replay.schema import Component, Condition, InputFrame, ReplayJob
    from edge_replay.worker import replay, sha256
    from reproduce_results import build_model
    from run_missing_indicator_robustness_experiments import predict_bundle

    # A base install supports LR without optional boosting libraries. Completion
    # metadata must not introduce a dependency the actual inference does not use.
    installed_version = importlib.metadata.version

    def base_install_version(name):
        if name in {"xgboost", "lightgbm"}:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr(importlib.metadata, "version", base_install_version)

    features = ["DO", "BOD", "NH3N", "EC", "SS"]
    available = ["DO", "EC", "SS"]
    frame = pd.DataFrame(np.random.default_rng(7).uniform(1, 20, (30, 5)), columns=features)
    y = frame.DO.to_numpy() * 2 + frame.BOD.to_numpy() + 5
    full = build_model("lr").fit(frame, y)
    reduced = build_model("lr").fit(frame[available], y)
    reconstructors = {name: build_model("lr").fit(frame[available], frame[name]) for name in ["BOD", "NH3N"]}
    reconstructed_train = frame.copy()
    for name, model in reconstructors.items():
        reconstructed_train[name] = model.predict(frame[available])
    stage2 = build_model("lr").fit(reconstructed_train, y)
    selected = reduced if mode == "reduced_retraining" else stage2 if mode == "indicator_reconstruction" else full
    bundle = {"wqi_model": selected, "missing_indicators": ["BOD", "NH3N"],
              "available_features": available, "indicator_models": reconstructors}
    expected = predict_bundle(mode, bundle, frame)
    packet = tmp_path / "packet"
    packet.mkdir()
    np.savez_compressed(packet / "inputs.npz", **{"test/x": frame.to_numpy(), "test/y": y, "test/row_id": np.arange(len(y))})
    models = {"wqi": selected}
    if mode == "indicator_reconstruction":
        models.update({name.lower(): model for name, model in reconstructors.items()})
    components = []
    for key, model in models.items():
        path = packet / (key + ".joblib")
        joblib.dump(model, path, compress=3)
        columns = available if key != "wqi" or mode == "reduced_retraining" else features
        components.append(Component(key=key, path=path.name, sha256=sha256(path), features=columns))
    condition = Condition(name=mode, mode=mode,
                          missing_indicators=[] if mode == "full_reference" else ["BOD", "NH3N"],
                          reconstruction={name:name.lower() for name in reconstructors} if mode == "indicator_reconstruction" else {})
    job = ReplayJob(job_id="fixture", protocol="complete_input", seed=0, model_type="lr",
                    source_artifact="fixture", source_sha256="0"*64,
                    inputs_sha256=sha256(packet / "inputs.npz"),
                    frames=[InputFrame(name="test", kind="test", n_rows=len(y))],
                    components=components, conditions=[condition], repeats=1, warmup=0)
    (packet / "job.json").write_text(job.model_dump_json())
    output = tmp_path / "output"
    replay(packet, output, None)
    with np.load(output / "wqi.npz") as predictions:
        np.testing.assert_allclose(predictions[mode + "/test"], expected, rtol=0, atol=1e-12)
    before = (output / "completed.json").read_bytes()
    assert json.loads(before)["versions"]["xgboost"] is None
    assert json.loads(before)["versions"]["lightgbm"] is None
    with pytest.raises(FileExistsError):
        replay(packet, output, None)
    assert (output / "completed.json").read_bytes() == before
    if mode == "full_reference":
        reference = tmp_path / "reference"
        reference.mkdir()
        (reference / "completed.json").write_bytes(before)
        np.savez_compressed(reference / "wqi.npz", **{mode + "/test":expected + 1})
        recorded = tmp_path / "recorded-finding"
        replay(packet, recorded, reference, record_parity_failures=True)
        finding = json.loads((recorded / "completed.json").read_text())
        assert finding["all_reference_parity_pass"] is False
        assert finding["parity_findings_recorded"] is True
        with np.load(recorded / "wqi.npz") as measured:
            np.testing.assert_allclose(measured[mode + "/test"], expected, rtol=0, atol=1e-12)


def test_component_checksum_is_checked_before_deserialization(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.schema import Component, ReplayJob
    from edge_replay.worker import component_job

    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "model.joblib").write_bytes(b"changed model")
    job = ReplayJob(job_id="fixture", protocol="complete_input", seed=0, model_type="lr",
                    source_artifact="fixture", source_sha256="0"*64, inputs_sha256="0"*64,
                    frames=[], conditions=[], components=[Component(key="wqi", path="model.joblib", sha256="0"*64, features=["DO"])])
    (packet / "job.json").write_text(job.model_dump_json())
    monkeypatch.setattr(joblib, "load", lambda *args, **kw: pytest.fail("Unchecked pickle was loaded"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        component_job(packet, tmp_path / "output", "wqi")


@pytest.mark.parametrize("damage", ["missing_frame", "wrong_prediction"])
def test_completion_marker_cannot_hide_missing_or_divergent_predictions(tmp_path, monkeypatch, damage):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.schema import Component, Condition, InputFrame, ReplayJob
    from edge_replay.worker import sha256
    from edge_replay.runner import verify_complete

    packet = tmp_path / "packet"
    packet.mkdir()
    job = ReplayJob(job_id="fixture", protocol="complete_input", seed=0, model_type="lr",
                    source_artifact="fixture", source_sha256="0"*64, inputs_sha256="0"*64,
                    frames=[InputFrame(name="test", kind="test", n_rows=2)],
                    conditions=[Condition(name="full_reference", mode="full_reference")],
                    components=[Component(key="wqi", path="wqi.joblib", sha256="0"*64, features=["DO"])])
    (packet / "job.json").write_text(job.model_dump_json())
    for platform in ["x86", "nano"]:
        result = tmp_path / platform / "fixture"
        result.mkdir(parents=True)
        (result / "completed.json").write_text(json.dumps({"job_sha256":sha256(packet / "job.json"),
                                                          "machine":"aarch64" if platform == "nano" else "x86_64",
                                                          "all_reference_parity_pass":True}))
        values = np.array([40.0, 80.0])
        if platform == "nano" and damage == "wrong_prediction":
            values[0] += 1.0
        key = "wrong/test" if platform == "nano" and damage == "missing_frame" else "full_reference/test"
        np.savez_compressed(result / "wqi.npz", **{key:values})
        (result / "wqi-timing.csv").write_text("phase,seconds\naccuracy,1\n")
        (result / "wqi-resources.json").write_text("{}")
    with pytest.raises(ValueError, match="prediction keys differ|prediction parity failed"):
        verify_complete(packet, tmp_path / "nano/fixture", parity=True)
    if damage == "wrong_prediction":
        with pytest.raises(ValueError, match="parity flag disagrees"):
            verify_complete(packet, tmp_path / "nano/fixture", parity=True, record_parity_failures=True)
        path = tmp_path / "nano/fixture/completed.json"
        marker = json.loads(path.read_text())
        marker["all_reference_parity_pass"] = False
        path.write_text(json.dumps(marker))
        assert verify_complete(packet, tmp_path / "nano/fixture", parity=True, record_parity_failures=True) is False
        assert json.loads(path.read_text())["all_reference_parity_pass"] is False
    else:
        with pytest.raises(ValueError, match="prediction keys differ"):
            verify_complete(packet, tmp_path / "nano/fixture", parity=True, record_parity_failures=True)


@pytest.mark.parametrize("scenario", ["complete", "incomplete", "interrupted_archive"])
def test_resume_recovers_finished_remote_job_without_repeating_inference(tmp_path, monkeypatch, scenario):
    incomplete = scenario != "complete"
    import shutil
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.schema import Component, Condition, InputFrame, ReplayJob
    from edge_replay.worker import sha256
    from edge_replay.runner import recover_remote_result

    packet = tmp_path / "staging/packets/fixture"
    packet.mkdir(parents=True)
    job = ReplayJob(job_id="fixture", protocol="complete_input", seed=0, model_type="lr",
                    source_artifact="fixture", source_sha256="0"*64, inputs_sha256="0"*64,
                    frames=[InputFrame(name="test", kind="test", n_rows=2)],
                    conditions=[Condition(name="full_reference", mode="full_reference")],
                    components=[Component(key="wqi", path="wqi.joblib", sha256="0"*64, features=["DO"])])
    serialized = job.model_dump_json()
    (packet / "job.json").write_text(serialized)
    output = tmp_path / "results"
    remote_result = tmp_path / "remote/fixture"
    for result, machine in [(output / "x86/fixture", "x86_64"), (remote_result, "aarch64")]:
        result.mkdir(parents=True)
        (result / "completed.json").write_text(json.dumps({"job_sha256":sha256(packet / "job.json"),
                                                          "machine":machine, "all_reference_parity_pass":True}))
        np.savez_compressed(result / "wqi.npz", **{"full_reference/test":np.array([40.,80.])})
        (result / "wqi-timing.csv").write_text("phase,seconds\naccuracy,1\n")
        (result / "wqi-resources.json").write_text("{}")
    partial = output / "nano/fixture"
    partial.mkdir(parents=True)
    (partial / "incomplete-copy.txt").write_text("retain this evidence")
    inspections = []
    deleted = []
    moved = set()
    interrupted = False

    class Remote:
        def shell(self, command, **kwargs):
            if command.startswith("test -e ") and command.removeprefix("test -e ") in moved:
                return SimpleNamespace(returncode=1, stdout="")
            if incomplete and command == "test -f /home/test/edge-research/outputs/fixture/completed.json":
                return SimpleNamespace(returncode=1, stdout="")
            return SimpleNamespace(returncode=0, stdout=serialized if command.startswith("cat ") else "")

        def docker(self, args, **kwargs):
            assert args[0] == "inspect", "Recovery must never run another model container"
            inspections.append(args)
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Running":len(inspections)==1}), stderr="")

        def download(self, source, destination):
            assert len(inspections) == 2, "Do not collect or delete an active worker"
            shutil.copytree(remote_result, destination / "fixture")

        def sudo(self, args):
            nonlocal interrupted
            if incomplete:
                assert len(inspections) >= 2
                assert args[0] in {"mkdir", "mv"}, "Incomplete results must be preserved"
                if args[0] == "mv":
                    if scenario == "interrupted_archive" and moved and not interrupted:
                        interrupted = True
                        assert "/home/test/edge-research/current-packet" not in moved
                        raise ConnectionError("archive transport interrupted")
                    moved.add(args[2])
            else:
                assert (output / "nano/fixture/completed.json").exists()
            deleted.append(args)

    monkeypatch.setattr("edge_replay.runner.time.sleep", lambda _: None)
    if scenario == "interrupted_archive":
        with pytest.raises(ConnectionError, match="archive transport interrupted"):
            recover_remote_result(Remote(), "/home/test/edge-research", tmp_path / "staging", output, resume_incomplete=True)
    recovered = recover_remote_result(Remote(), "/home/test/edge-research", tmp_path / "staging", output, resume_incomplete=incomplete)
    if incomplete:
        assert recovered is None
        assert len(deleted) == (5 if scenario == "interrupted_archive" else 4)
        assert moved == {"/home/test/edge-research/current-packet", "/home/test/edge-research/current-reference", "/home/test/edge-research/outputs/fixture"}
        assert not partial.exists()
        assert list((output / "interruptions").glob("*/incomplete-copy.txt"))
        return
    assert recovered == "fixture"
    assert len(deleted) == 1
    assert list((output / "previous-downloads").glob("*/incomplete-copy.txt"))


@pytest.mark.parametrize("failure", ["Permission denied (publickey,password).", "Connection reset by peer"])
def test_ssh_renewal_does_not_repeat_a_possibly_running_inference(tmp_path, monkeypatch, failure):
    import subprocess
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.runner import Remote

    remote = Remote("user@nano", "/tmp/test-control", "sudo-secret", ssh_password="ssh-secret")
    calls, renewed = [], []

    def execute(command, **kwargs):
        assert "ssh-secret" not in " ".join(command)
        calls.append(command)
        return subprocess.CompletedProcess(command, 255 if len(calls)==1 else 0, "", failure if len(calls)==1 else "")

    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setattr(remote, "renew_control", lambda: renewed.append(True))
    if failure.startswith("Permission denied"):
        assert remote.docker(["run", "model-image"]).returncode == 0
        assert len(calls) == 2 and renewed == [True]
    else:
        with pytest.raises(subprocess.CalledProcessError):
            remote.docker(["run", "model-image"])
        assert len(calls) == 1 and renewed == []

@pytest.mark.parametrize("manifest", [None, '{"job_id":'])
@pytest.mark.parametrize("running", [False, True])
def test_unidentified_transfer_is_preserved_and_never_moved_with_live_worker(tmp_path, monkeypatch, manifest, running):
    import shlex
    import shutil
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from edge_replay.runner import recover_remote_result

    root = tmp_path / "remote"
    packet = root / "current-packet"
    packet.mkdir(parents=True)
    (packet / "partial-model").write_bytes(b"model evidence")
    if manifest is not None:
        (packet / "job.json").write_text(manifest)
    reference = root / "current-reference"
    reference.mkdir()
    (reference / "wqi.npz").write_bytes(b"reference evidence")
    result = root / "outputs/job"
    result.mkdir(parents=True)
    (result / "partial-result").write_bytes(b"result evidence")
    output = tmp_path / "local"
    output.mkdir()

    class Remote:
        def shell(self, command, **kwargs):
            args = shlex.split(command)
            if args[0] == "cat":
                return SimpleNamespace(stdout=Path(args[1]).read_text(), returncode=0)
            path = Path(args[2])
            present = path.is_file() if args[1] == "-f" else path.exists()
            return SimpleNamespace(returncode=0 if present else 1)

        def docker(self, args, **kwargs):
            assert args[0] == "ps"
            return SimpleNamespace(stdout="watermirror-replay-job\n" if running else "")

        def sudo(self, args):
            assert not running, "Live worker staging must not move"
            if args[0] == "mkdir":
                Path(args[-1]).mkdir(parents=True)
            else:
                assert args[:2] == ["mv", "--"]
                shutil.move(args[2], args[3])

    if running:
        with pytest.raises(RuntimeError, match="running replay worker"):
            recover_remote_result(Remote(), str(root), tmp_path, output, resume_incomplete=True)
        assert (packet / "partial-model").read_bytes() == b"model evidence"
        assert (result / "partial-result").exists()
    else:
        recover_remote_result(Remote(), str(root), tmp_path, output, resume_incomplete=True)
        archive, = (root / "interruptions").iterdir()
        assert (archive / "current-packet/partial-model").read_bytes() == b"model evidence"
        assert (archive / "current-reference/wqi.npz").read_bytes() == b"reference evidence"
        assert (archive / "outputs/job/partial-result").read_bytes() == b"result evidence"
        assert not packet.exists() and not reference.exists()
        # Another invocation must accept the now-empty staging area.
        assert recover_remote_result(Remote(), str(root), tmp_path, output, resume_incomplete=True) is None

@pytest.mark.parametrize("target", ["-oProxyCommand=touch /tmp/injected", "user@host;id", "host\n-oProxyCommand=id"])
def test_remote_rejects_option_or_shell_target_before_starting_transport(target, monkeypatch):
    from edge_replay.runner import Remote
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Invalid target reached a subprocess"))
    with pytest.raises(ValueError, match="SSH target"):
        Remote(target, None, "secret")


@pytest.mark.parametrize("remote_path", ["-oProxyCommand=id", "/tmp/../etc", "/tmp/$(id)", "/tmp/a;id", "/"])
def test_remote_transfer_rejects_unsafe_paths_before_scp(tmp_path, remote_path, monkeypatch):
    from edge_replay.runner import Remote
    remote = Remote("user@100.67.19.24", None, "secret")
    monkeypatch.setattr(remote, "transport", lambda *a, **kw: pytest.fail("Invalid path reached SCP"))
    for method in [remote.upload, remote.download]:
        with pytest.raises(ValueError, match="Remote staging"):
            if method == remote.upload:
                method(tmp_path / "packet", remote_path)
            else:
                method(remote_path, tmp_path / "output")


def test_scp_keeps_absolute_paths_and_separates_options(tmp_path, monkeypatch):
    from edge_replay.runner import Remote
    remote = Remote("user@nano", None, "secret")
    calls = []
    monkeypatch.setattr(remote, "transport", lambda command, **kw: calls.append(command))
    remote.upload(tmp_path / "-packet", "/home/user/edge-research/current-packet")
    command, = calls
    assert command[command.index("--") + 1:] == [str(tmp_path / "-packet"), "user@nano:/home/user/edge-research/current-packet"]


@pytest.mark.parametrize("field,value", [("key", "../../original"), ("path", "../../model.joblib"),
                                         ("path", "/tmp/model.joblib")])
def test_component_schema_rejects_packet_path_escape(field, value):
    from edge_replay.schema import Component
    data = {"key": "wqi", "path": "wqi.joblib", "sha256": "0" * 64, "features": ["DO"]}
    data[field] = value
    with pytest.raises(ValueError):
        Component(**data)


def test_replay_path_boundary_preserves_reference_and_rejects_symlink_escape(tmp_path):
    from edge_replay.worker import packet_file, replay_paths
    packet = tmp_path / "packet"
    packet.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    for output in [packet / "result", reference / "result", tmp_path, tmp_path / "safe/../escaped"]:
        with pytest.raises(ValueError):
            replay_paths(packet, output, reference)
        assert not output.exists() or output == tmp_path
    external = tmp_path / "original-model.joblib"
    external.write_bytes(b"preserve original")
    (packet / "wqi.joblib").symlink_to(external)
    with pytest.raises(ValueError, match="including symlinks"):
        packet_file(packet.resolve(), "wqi.joblib")
    assert external.read_bytes() == b"preserve original"
    assert replay_paths(packet, tmp_path / "new-absolute-output", reference)[1] == tmp_path / "new-absolute-output"


def test_reconstruction_dependency_cannot_escape_output_directory():
    from edge_replay.schema import Condition
    with pytest.raises(ValueError, match="Replay identifiers"):
        Condition(name="reconstruction", mode="indicator_reconstruction", reconstruction={"BOD": "../original"})


@pytest.mark.parametrize("value", ["--all", "-a", "edge container", "edge;id", "registry/edge", "edge:latest", ""])
def test_container_names_reject_docker_options_and_non_names(value):
    from edge_replay.runner import safe_container_name
    with pytest.raises(ValueError, match="Docker container"):
        safe_container_name(value)


@pytest.mark.parametrize("value", ["--privileged", "-v/tmp:/data", "image name", "image;id", ""])
def test_image_references_reject_option_or_shell_injection(value):
    from edge_replay.runner import safe_image_reference
    with pytest.raises(ValueError, match="Docker image"):
        safe_image_reference(value)


@pytest.mark.parametrize("value", ["watermirror-edge-research:20261026", "ghcr.io/team/edge:release-1",
                                   "localhost:5000/team/edge:v2", "edge@sha256:" + "a" * 64])
def test_image_references_preserve_registry_port_tag_and_digest(value):
    from edge_replay.runner import safe_image_reference
    assert safe_image_reference(value) == value


def test_runner_rejects_api_container_option_before_creating_output(tmp_path, monkeypatch):
    from edge_replay.runner import main
    import subprocess
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run", "--staging", str(tmp_path / "staging"),
                                     "--output", str(output), "--remote", "user@nano", "--api-container=--all"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Invalid name reached Docker"))
    with pytest.raises(ValueError, match="Docker container"):
        main()
    assert not output.exists()


def test_runner_keeps_empty_wait_container_as_disabled(tmp_path, monkeypatch):
    import edge_replay.runner as runner
    class StopBeforeCredentials(Exception):
        pass
    def stop_after_validation(*args):
        raise StopBeforeCredentials
    monkeypatch.setattr(sys, "argv", ["run", "--staging", str(tmp_path / "staging"),
                                     "--output", str(tmp_path / "output"), "--remote", "user@nano", "--wait-container="])
    monkeypatch.setattr(runner, "acquire_runner_lock", stop_after_validation)
    with pytest.raises(StopBeforeCredentials):
        runner.main()
    assert runner.safe_container_name("watermirror-edge") == "watermirror-edge"
    assert runner.safe_container_name("watermirror-soak-rf") == "watermirror-soak-rf"
