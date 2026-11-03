"""Run a pre-staged packet under a bounded, isolated Nano research container."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

from edge_replay.runner import Remote, safe_image_reference, safe_remote_path
from edge_optimization_benchmark import INDICATORS, MODES, validate_request
from edge_optimization_control import supervise_container
from edge_optimization_paths import artifact_path


class BoundedRemote(Remote):
    def transport(self, command, *, check=True, **kwargs):
        kwargs.setdefault("timeout", 30)
        return super().transport(command, check=check, **kwargs)


def prepare_remote_output(remote, path):
    """Create only a validated staging output; operands cannot become options."""
    path = safe_remote_path(path)
    if not any(
        part.startswith("edge-optimization-") for part in PurePosixPath(path).parts
    ):
        raise ValueError(
            "Use a dedicated edge-optimization-* research staging directory"
        )
    remote.sudo(["mkdir", "--", path])
    remote.sudo(["chown", "--", "1000:1000", path])


def build_worker_command(kind, mode, missing, model_filename):
    """Serialize the fixed packet schema, not caller-supplied command operands."""
    kinds = {"native": "native", "onnx": "onnx"}
    modes = {value: value for value in MODES}
    try:
        kind = kinds[kind]
        mode = modes[mode]
    except KeyError as exc:
        raise ValueError("Unsupported worker kind or mode") from exc
    missing = validate_request(kind, mode, missing)
    native_models = {
        "full_reference": "full_reference.joblib",
        "inference_dropout": "full_reference.joblib",
        "reduced_retraining": "reduced_retraining.joblib",
        "indicator_reconstruction": "indicator_reconstruction.joblib",
    }
    expected_model = "pipeline.onnx" if kind == "onnx" else native_models[mode]
    if model_filename != expected_model:
        raise ValueError("Model filename does not match the worker packet schema")
    return [
        "python",
        "/worker/benchmark.py",
        "--kind",
        kind,
        "--model",
        "/packet/" + expected_model,
        "--inputs",
        "/packet/inputs.npz",
        "--output",
        "/output/result",
        "--mode",
        mode,
        "--missing",
        *missing,
    ]


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
    p.add_argument("--missing", nargs="*", default=[], choices=INDICATORS)
    args = p.parse_args()
    # Reject before creating a remote output directory or research container.
    try:
        command = build_worker_command(
            args.kind, args.mode, args.missing, args.model_filename
        )
    except ValueError as exc:
        p.error(str(exc))
    args.evidence = artifact_path(args.evidence)
    args.packet = safe_remote_path(args.packet)
    args.worker = safe_remote_path(args.worker)
    args.remote_output = safe_remote_path(args.remote_output)
    for value in [args.packet, args.worker, args.remote_output]:
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
    prepare_remote_output(remote, args.remote_output)
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
