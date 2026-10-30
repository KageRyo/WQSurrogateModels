"""Transfer and replay one completed packet at a time, with durable outputs."""
from __future__ import annotations

import argparse
import getpass
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np

from edge_replay.schema import ReplayJob, safe_local_path
from edge_replay.worker import sha256

ROOT = Path(__file__).resolve().parents[2]


def safe_remote_path(value):
    if (not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
            or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) in {"/", "//"}):
        raise ValueError("Remote staging paths must be absolute, without traversal or shell characters")
    return str(PurePosixPath(value))


def safe_ssh_target(value):
    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?(?:[A-Za-z0-9][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:]+\])", value):
        raise ValueError("SSH target must be a hostname or IP, optionally prefixed with user@")
    return value


def safe_container_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Docker container names must not contain options, paths or whitespace")
    return value


def safe_image_reference(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]*", value):
        raise ValueError("Docker image references must not contain leading options, whitespace or shell characters")
    return value


class Remote:
    def __init__(self, target, control, password, ssh_password=None):
        self.target = safe_ssh_target(target)
        control = str(safe_local_path(control)) if control else None
        self.options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
        if control:
            self.options += ["-o", "ControlPath=" + control]
        self.password = password
        self.control = control
        self.ssh_password = ssh_password

    def renew_control(self):
        if not self.control or self.ssh_password is None:
            raise RuntimeError("SSH password renewal requires a control socket and an in-memory password")
        checked = subprocess.run(["ssh", "-S", self.control, "-O", "check", "--", self.target],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if checked.returncode == 0:
            return
        control_path = Path(self.control)
        if "Connection refused" in checked.stderr and control_path.exists():
            if not stat.S_ISSOCK(control_path.lstat().st_mode):
                raise RuntimeError("Refusing to replace a non-socket SSH control path")
            control_path.unlink()
        # The helper contains code only. Credentials travel through the child
        # environment into SSH's askpass pipe, never a file or process argument.
        with tempfile.TemporaryDirectory(prefix="watermirror-ssh-") as directory:
            helper = Path(directory) / "askpass.py"
            helper.write_text("#!/usr/bin/env python3\nimport os\nprint(os.environ['WATERMIRROR_SSH_PASSWORD'])\n")
            helper.chmod(0o700)
            environment = {**os.environ, "SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
                           "DISPLAY": "watermirror", "WATERMIRROR_SSH_PASSWORD": self.ssh_password}
            result = subprocess.run(["ssh", "-MNf", "-S", self.control, "-o", "ControlPersist=yes",
                                     "-o", "BatchMode=no", "-o", "StrictHostKeyChecking=yes",
                                     "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
                                     "-o", "ConnectTimeout=20", "--", self.target],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=environment, start_new_session=True, timeout=45)
            if result.returncode:
                raise RuntimeError("Could not renew the authenticated SSH control connection")

    def transport(self, command, *, check=True, **kwargs):
        if not isinstance(command, list) or not command or command[0] not in {"ssh", "scp"}:
            raise ValueError("Replay transport only invokes SSH or SCP argument lists")
        result = subprocess.run(command, check=False, shell=False, **kwargs)
        # This error proves authentication failed before the remote command ran.
        # A dropped connection during a running Docker command is NOT retried:
        # the existing worker must be inspected/recovered instead of repeated.
        if result.returncode == 255 and "Permission denied (publickey,password)" in (result.stderr or "") and self.ssh_password is not None:
            print(json.dumps({"phase":"renewing_ssh_control_connection"}), flush=True)
            self.renew_control()
            result = subprocess.run(command, check=False, shell=False, **kwargs)
        if check:
            result.check_returncode()
        return result

    def shell(self, command, *, check=True, input=None):
        return self.transport(["ssh", *self.options, "--", self.target, command], input=input,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def docker(self, args, *, check=True):
        return self.sudo(["docker", *args], check=check)

    def sudo(self, args, *, check=True):
        # Password is passed only through stdin, never argv, files or logs.
        command = "sudo -S -p '' -- " + shlex.join(args)
        return self.shell(command, check=check, input=self.password + "\n")

    def upload(self, local, remote):
        remote = safe_remote_path(remote)
        local = safe_local_path(local)
        self.transport(["scp", *self.options, "-q", "-r", "--", str(local), self.target + ":" + remote], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def download(self, remote, local):
        remote = safe_remote_path(remote)
        local = safe_local_path(local)
        self.transport(["scp", *self.options, "-q", "-r", "--", self.target + ":" + remote, str(local)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def progress(path, **values):
    values["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(values), flush=True)


def verify_complete(packet, result, *, parity=False, reference=None, record_parity_failures=False):
    marker = json.loads((result / "completed.json").read_text())
    if marker["job_sha256"] != sha256(packet / "job.json"):
        raise ValueError("Completed output does not match the packet")
    job = ReplayJob.model_validate_json((packet / "job.json").read_text())
    expected = {condition.name + "/" + frame.name: frame.n_rows
                for condition in job.conditions for frame in job.frames}
    for component in job.components:
        for suffix in [".npz", "-timing.csv", "-resources.json"]:
            if not (result / (component.key + suffix)).is_file():
                raise ValueError("Completed output is missing a component result")
        with np.load(result / (component.key + ".npz"), allow_pickle=False) as predictions:
            if set(predictions.files) != set(expected):
                raise ValueError("Completed output prediction keys differ from the job")
            for key, rows in expected.items():
                values = predictions[key]
                if values.shape != (rows,) or not np.isfinite(values).all():
                    raise ValueError("Completed output has invalid prediction rows: " + key)
    if parity and marker["all_reference_parity_pass"] is not True and not record_parity_failures:
        raise ValueError("Completed ARM64 output failed reference parity")
    if parity and marker["machine"] != "aarch64":
        raise ValueError("The edge result was not produced by an ARM64 worker")
    if parity:
        matches = True
        reference = reference or result.parent.parent / "x86" / result.name
        reference_marker = json.loads((reference / "completed.json").read_text())
        if reference_marker["job_sha256"] != marker["job_sha256"]:
            raise ValueError("Reference and edge output use different jobs")
        with np.load(reference / "wqi.npz", allow_pickle=False) as original, np.load(result / "wqi.npz", allow_pickle=False) as actual:
            if set(original.files) != set(expected):
                raise ValueError("Reference prediction keys differ from the job")
            for key, rows in expected.items():
                values = original[key]
                if values.shape != (rows,) or not np.isfinite(values).all():
                    raise ValueError("Invalid reference prediction rows: " + key)
                if not np.all(np.abs(actual[key] - values) <= 1e-6):
                    matches = False
                if not matches and not record_parity_failures:
                    raise ValueError("Saved prediction parity failed: " + key)
        if marker["all_reference_parity_pass"] is not matches:
            raise ValueError("Saved prediction parity flag disagrees with the arrays")
        return matches


def archive_unidentified_staging(remote, remote_root, output):
    """Preserve interrupted transfers only after ruling out every live replay."""
    sources = []
    for name in ["current-reference", "current-packet"]:
        source = remote_root + "/" + name
        observed = remote.shell("test -e " + shlex.quote(source), check=False)
        if observed.returncode == 0:
            sources.append(source)
        elif observed.returncode != 1:
            raise RuntimeError("Cannot inspect unidentified replay staging")
    if not sources:
        return
    running = remote.docker(["ps", "--filter", "name=^/watermirror-replay-", "--format", "{{.Names}}"])
    if running.stdout.strip():
        raise RuntimeError("Unidentified staging has a running replay worker; retained for inspection")
    archive = remote_root + "/interruptions/unidentified-" + str(time.time_ns())
    remote.sudo(["mkdir", "-p", archive])
    # Preserve any associated output too. A missing/truncated manifest cannot
    # reliably identify its job. No replay worker may be alive during this move.
    outputs = remote_root + "/outputs"
    observed = remote.shell("test -e " + shlex.quote(outputs), check=False)
    if observed.returncode == 0:
        remote.sudo(["mv", "--", outputs, archive + "/"])
    elif observed.returncode != 1:
        raise RuntimeError("Cannot inspect unidentified replay outputs")
    for source in sources:
        remote.sudo(["mv", "--", source, archive + "/"])
    progress(output / "progress.json", phase="unidentified_remote_staging_archived", archive=archive)


def recover_remote_result(remote, remote_root, staging, output, *, record_parity_failures=False, resume_incomplete=False):
    """Recover a terminal job after a lost connection without executing it again."""
    remote_packet = remote_root + "/current-packet"
    observed = remote.shell("test -f " + shlex.quote(remote_packet + "/job.json"), check=False)
    if observed.returncode == 1:
        if resume_incomplete:
            archive_unidentified_staging(remote, remote_root, output)
        return None
    if observed.returncode:
        raise RuntimeError("Cannot inspect remote replay staging")
    serialized = remote.shell("cat " + shlex.quote(remote_packet + "/job.json")).stdout
    try:
        job = ReplayJob.model_validate_json(serialized)
    except ValueError:
        if not resume_incomplete:
            raise
        archive_unidentified_staging(remote, remote_root, output)
        return None
    if not job.job_id or any(not (char.isalnum() or char in "-_") for char in job.job_id):
        raise ValueError("Unsafe remote job identifier")
    packet = staging / "packets" / job.job_id
    if serialized != (packet / "job.json").read_text():
        raise ValueError("Remote packet does not match the local replay job")
    container = "watermirror-replay-" + job.job_id
    while True:
        inspected = remote.docker(["inspect", "-f", "{{json .State}}", container], check=False)
        if inspected.returncode:
            if "No such" not in inspected.stderr:
                raise RuntimeError("Cannot verify the interrupted worker state")
            break
        state = json.loads(inspected.stdout)
        if not state["Running"]:
            break
        progress(output / "progress.json", phase="waiting_for_existing_worker", job_id=job.job_id, container=container)
        time.sleep(30)
    remote_result = remote_root + "/outputs/" + job.job_id
    completed = remote.shell("test -f " + shlex.quote(remote_result + "/completed.json"), check=False)
    if completed.returncode:
        if completed.returncode != 1:
            raise RuntimeError("Cannot inspect interrupted result")
        if resume_incomplete:
            archive = remote_root + "/interruptions/" + job.job_id + "-" + str(time.time_ns())
            remote.sudo(["mkdir", "-p", archive])
            # No container is running. Preserve partial transfers/results before
            # retrying this packet; never delete an unverified result. Keep
            # the packet recovery pointer until every associated path is moved.
            for source in [remote_root + "/current-reference", remote_result, remote_packet]:
                exists = remote.shell("test -e " + shlex.quote(source), check=False)
                if exists.returncode == 0:
                    remote.sudo(["mv", "--", source, archive + "/"])
                elif exists.returncode != 1:
                    raise RuntimeError("Cannot inspect interrupted staging")
            partial_download = output / "nano" / job.job_id
            if partial_download.exists() and not (partial_download / "completed.json").exists():
                local_archive = output / "interruptions" / (job.job_id + "-nano-download-" + str(time.time_ns()))
                local_archive.parent.mkdir(parents=True, exist_ok=True)
                partial_download.rename(local_archive)
            progress(output / "progress.json", phase="incomplete_remote_job_archived", job_id=job.job_id, archive=archive)
            return None
        raise RuntimeError("Interrupted job has no complete remote result; partial data retained for inspection")
    reference = output / "x86" / job.job_id
    verify_complete(packet, reference)
    # A failed download may leave only a subset of files. Download into a new
    # directory and validate all arrays before replacing that partial copy.
    temporary = output / "recovered-downloads" / str(time.time_ns())
    temporary.mkdir(parents=True)
    remote.download(remote_result, temporary)
    recovered = temporary / job.job_id
    verify_complete(packet, recovered, parity=True, reference=reference, record_parity_failures=record_parity_failures)
    destination = output / "nano" / job.job_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = output / "previous-downloads" / (job.job_id + "-" + str(time.time_ns()))
        previous.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(previous)
    recovered.rename(destination)
    remote.sudo(["rm", "-rf", "--", remote_packet, remote_root + "/current-reference", remote_result])
    progress(output / "progress.json", phase="remote_result_recovered", job_id=job.job_id)
    return job.job_id


def install_remote_source(remote, remote_root):
    target = safe_remote_path(remote_root) + "/source"
    remote.shell("mkdir -p " + shlex.quote(target + "/scripts") + " " + shlex.quote(target + "/src"))
    for item in ["Dockerfile.edge-research", "requirements-edge-research.txt"]:
        remote.upload(ROOT / item, target + "/")
    # Only public runtime/research source is sent to the image build context.
    # Model packets and private inputs have a separate directory and mounts.
    archive = subprocess.Popen(["tar", "--exclude=__pycache__", "-C", str(ROOT), "-cf", "-", "src", "scripts"], stdout=subprocess.PIPE)
    subprocess.run(["ssh", *remote.options, "--", remote.target,
                    "tar -xf - -C " + shlex.quote(target)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("Source archive transfer failed")
    remote.upload(ROOT / ".dockerignore", target + "/")
    return target


def archive_incomplete_reference(reference, output):
    if reference.exists() and not (reference / "completed.json").exists():
        archive = output / "interruptions" / (reference.name + "-x86-" + str(time.time_ns()))
        archive.parent.mkdir(parents=True, exist_ok=True)
        reference.rename(archive)
        return archive


def acquire_runner_lock(output):
    handle = (output / "runner.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another replay controller owns this output directory") from None
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--ssh-control")
    parser.add_argument("--remote-root", help="Absolute remote staging directory; defaults below the SSH user's home")
    parser.add_argument("--image", default="watermirror-edge-research:20261026")
    parser.add_argument("--wait-container", default="watermirror-soak-rf")
    parser.add_argument("--api-container", default="watermirror-edge")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--record-parity-failures", action="store_true", help="Complete research measurements while retaining failed parity as findings; never mark them as passed")
    parser.add_argument("--ssh-password-auth", action="store_true", help="Keep SSH password in memory to renew an expired control connection automatically")
    parser.add_argument("--resume-incomplete", action="store_true", help="Archive terminal incomplete outputs before retrying; wait for live Nano workers")
    parser.add_argument("--systemd-credentials", action="store_true", help="Read nano-password from systemd's credential directory")
    args = parser.parse_args()
    args.staging = safe_local_path(args.staging)
    output = safe_local_path(args.output)
    if output.is_relative_to(args.staging) or args.staging.is_relative_to(output):
        raise ValueError("Runner output must not overlap replay staging")
    safe_ssh_target(args.remote)
    if args.remote_root is not None:
        args.remote_root = safe_remote_path(args.remote_root)
    args.image = safe_image_reference(args.image)
    args.api_container = safe_container_name(args.api_container)
    args.wait_container = safe_container_name(args.wait_container) if args.wait_container else ""
    output.mkdir(parents=True, exist_ok=True)
    runner_lock = acquire_runner_lock(output)
    if (output / "runner-complete.json").exists():
        raise FileExistsError("This replay already completed; use its existing results")
    if args.systemd_credentials:
        password = (Path(os.environ["CREDENTIALS_DIRECTORY"]) / "nano-password").read_text().rstrip("\n")
        ssh_password = password
    else:
        password = getpass.getpass("Nano sudo password (memory only): ")
        ssh_password = getpass.getpass("Nano SSH password (memory only): ") if args.ssh_password_auth else None
    remote = Remote(args.remote, args.ssh_control, password, ssh_password=ssh_password)
    uid = remote.shell("id -u").stdout.strip()
    gid = remote.shell("id -g").stdout.strip()
    if args.remote_root is None:
        remote_home = remote.shell('printf "%s" "$HOME"').stdout.strip()
        if not remote_home.startswith("/"):
            raise ValueError("Cannot determine an absolute remote home directory")
        args.remote_root = safe_remote_path(remote_home.rstrip("/") + "/water-quality/edge-research-20261026")
    state = output / "progress.json"
    # Do not overlap research installation, transfer or inference with the
    # sustained device benchmark. A missing container alone is not success.
    if args.wait_container:
        while True:
            observed = remote.docker(["inspect", "-f", "{{.State.Running}}", args.wait_container], check=False)
            if observed.returncode == 0 and observed.stdout.strip() == "true":
                progress(state, phase="waiting_for_live_soak_container", container=args.wait_container)
                time.sleep(30)
                continue
            if observed.returncode != 0 and "No such" not in observed.stderr:
                raise RuntimeError("Cannot verify soak container state: " + observed.stderr)
            status = remote.shell("cat /home/isilab/water-quality/evidence/soak-rf.exit", check=False)
            if status.returncode != 0 or status.stdout.strip() != "0":
                raise RuntimeError("Soak is no longer running but has no successful exit evidence")
            break
    recover_remote_result(remote, args.remote_root, args.staging, output, record_parity_failures=args.record_parity_failures, resume_incomplete=args.resume_incomplete)
    progress(state, phase="building_separate_research_image")
    source = install_remote_source(remote, args.remote_root)
    build = remote.docker(["build", "-f", source + "/Dockerfile.edge-research", "-t", args.image, source], check=False)
    with (output / "research-image-build.log").open("a") as log:
        log.write("\n" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n" + build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError("Research image build failed; see research-image-build.log")
    image_id = remote.docker(["image", "inspect", "-f", "{{.Id}}", args.image]).stdout.strip()
    environment = remote.docker(["run", "--rm", args.image, "python", "-c",
        "import json,platform,importlib.metadata as m; print(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'packages':{n:m.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','xgboost','lightgbm']}}))"])
    (output / "research-container-environment.json").write_text(environment.stdout)
    remote.shell("mkdir -p " + shlex.quote(args.remote_root + "/outputs"))
    # Release this task's API model cache while running the research replay.
    # Restart the same task-owned container when the runner finishes or fails.
    remote.docker(["stop", args.api_container])
    completed = 0
    newly_completed = 0
    try:
        while True:
            packets = list((args.staging / "packets").glob("*/job.json"))
            priority = {name:i for i,name in enumerate(["lightgbm", "lr", "mpr", "svm", "xgboost", "rf", "direct_wqi5"])}
            packets.sort(key=lambda p:(priority.get(json.loads(p.read_text())["model_type"],99), p.stat().st_mtime_ns))
            pending = []
            completed = 0
            for manifest in packets:
                packet = manifest.parent
                result = output / "nano" / packet.name
                if (result / "completed.json").exists():
                    verify_complete(packet, result, parity=True, record_parity_failures=args.record_parity_failures)
                    completed += 1
                else:
                    pending.append(packet)
            if not pending:
                if (args.staging / "preparation-complete.json").exists():
                    break
                progress(state, phase="waiting_for_preparation", completed_jobs=completed)
                time.sleep(10)
                continue
            for packet in pending:
                job_id = packet.name
                reference = output / "x86" / job_id
                if args.resume_incomplete:
                    archive_incomplete_reference(reference, output)
                if not (reference / "completed.json").exists():
                    progress(state, phase="x86_reference_inference", job_id=job_id, completed_jobs=completed)
                    command = [sys.executable, str(ROOT / "scripts/replay_edge_experiments.py"), "worker",
                               "--packet", str(packet.resolve()), "--output", str(reference)]
                    with (output / (job_id + "-x86.log")).open("a") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                verify_complete(packet, reference)
                progress(state, phase="transferring_packet", job_id=job_id, completed_jobs=completed)
                remote_packet = args.remote_root + "/current-packet"
                # A partial transfer is retained on failure. Never silently
                # reuse or remove it on a resumed run; inspect before recovery.
                exists = remote.shell("test -e " + shlex.quote(remote_packet), check=False)
                if exists.returncode == 0:
                    raise FileExistsError("Remote packet still exists; inspect prior worker/transfer state before recovery")
                remote.upload(packet, remote_packet)
                reference_remote = args.remote_root + "/current-reference"
                remote.shell("mkdir " + shlex.quote(reference_remote))
                remote.upload(reference / "wqi.npz", reference_remote + "/")
                remote.upload(reference / "completed.json", reference_remote + "/")
                container = "watermirror-replay-" + job_id
                progress(state, phase="nano_inference", job_id=job_id, container=container, completed_jobs=completed)
                result = remote.docker(["run", "--rm", "--network", "none", "--user", uid+":"+gid, "--name", container,
                    "--env", "REPLAY_IMAGE_ID="+image_id,
                    "--mount", "type=bind,src="+remote_packet+",dst=/packet,readonly",
                    "--mount", "type=bind,src="+reference_remote+",dst=/reference,readonly",
                    "--mount", "type=bind,src="+args.remote_root+"/outputs,dst=/outputs",
                    args.image, "python", "scripts/replay_edge_experiments.py", "worker",
                    "--packet", "/packet", "--output", "/outputs/"+job_id, "--reference", "/reference",
                    *(["--record-parity-failures"] if args.record_parity_failures else [])], check=False)
                (output / (job_id + "-nano.log")).write_text(result.stdout + result.stderr)
                destination = output / "nano"
                destination.mkdir(parents=True, exist_ok=True)
                remote.download(args.remote_root + "/outputs/" + job_id, destination)
                if result.returncode:
                    raise RuntimeError("Nano worker failed; partial results and remote packet retained: " + job_id)
                parity_pass = verify_complete(packet, destination / job_id, parity=True, record_parity_failures=args.record_parity_failures)
                # Delete only verified, task-owned remote staging copies.
                remote.sudo(["rm", "-rf", "--", remote_packet, reference_remote, args.remote_root + "/outputs/" + job_id])
                completed += 1
                newly_completed += 1
                progress(state, phase="job_verified" if parity_pass else "job_complete_with_parity_difference", job_id=job_id, completed_jobs=completed, parity_pass=parity_pass)
                if args.max_jobs and newly_completed >= args.max_jobs:
                    return
        differences = [p.parent.name for p in (output / "nano").glob("*/completed.json") if json.loads(p.read_text())["all_reference_parity_pass"] is False]
        progress(output / "runner-complete.json", phase="all_jobs_complete", jobs=len(packets), all_parity_pass=not differences,
                 parity_difference_jobs=differences, parity_findings_recorded=args.record_parity_failures)
    finally:
        remote.docker(["start", args.api_container], check=False)
