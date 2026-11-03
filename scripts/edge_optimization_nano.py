"""Run a pre-staged packet under a bounded, isolated Nano research container."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

from edge_replay.runner import Remote, safe_image_reference, safe_remote_path
from edge_optimization_benchmark import MODES, validate_request
from edge_optimization_control import supervise_container
from edge_optimization_paths import artifact_path


class BoundedRemote(Remote):
    def transport(self, command, *, check=True, **kwargs):
        kwargs.setdefault("timeout", 30)
        return super().transport(command, check=check, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--packet", required=True)
    p.add_argument("--worker", required=True)
    p.add_argument("--remote-output", required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--timeout-seconds", type=float, required=True)
    p.add_argument("--kind", choices=["native", "onnx"], required=True)
    p.add_argument("--model-filename", required=True)
    p.add_argument("--mode", choices=MODES, default="full_reference")
    p.add_argument("--missing", nargs="*", default=[])
    args = p.parse_args()
    # Reject before creating a remote output directory or research container.
    try:
        validate_request(args.kind, args.mode, args.missing)
    except ValueError as exc:
        p.error(str(exc))
    args.evidence = artifact_path(args.evidence)
    for value in [args.packet, args.worker, args.remote_output]:
        safe_remote_path(value)
        parts = PurePosixPath(value).parts
        if ".." in parts or not any(x.startswith("edge-optimization-") for x in parts):
            raise ValueError(
                "Use a dedicated edge-optimization-* research staging directory"
            )
    if Path(args.model_filename).name != args.model_filename or args.model_filename in (
        ".",
        "..",
    ):
        raise ValueError("Model must be a packet filename")
    if args.evidence.exists():
        raise FileExistsError(args.evidence)
    password = (
        (Path(os.environ["CREDENTIALS_DIRECTORY"]) / "nano-password")
        .read_text()
        .rstrip("\n")
    )
    remote = BoundedRemote(args.host, args.control, password, ssh_password=password)
    remote.renew_control()
    # Refuse existing outputs. Never clear a result or replace a running service.
    remote.sudo(["mkdir", args.remote_output])
    remote.sudo(["chown", "1000:1000", args.remote_output])
    create = [
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "1100m",
        "--memory-swap",
        "3500m",
        "--user",
        "1000:1000",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,size=64m",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-v",
        args.packet + ":/packet:ro",
        "-v",
        args.worker + ":/worker:ro",
        "-v",
        args.remote_output + ":/output:rw",
        safe_image_reference(args.image),
    ]
    command = [
        "python",
        "/worker/benchmark.py",
        "--kind",
        args.kind,
        "--model",
        "/packet/" + args.model_filename,
        "--inputs",
        "/packet/inputs.npz",
        "--output",
        "/output/result",
        "--mode",
        args.mode,
        "--missing",
        *args.missing,
    ]
    result = supervise_container(
        remote,
        name=args.name,
        create_args=create,
        command=command,
        timeout_seconds=args.timeout_seconds,
        evidence=args.evidence,
    )
    print(result)
    if result["phase"] != "completed":
        raise SystemExit(124 if result["phase"] == "timed_out" else 1)


if __name__ == "__main__":
    main()
