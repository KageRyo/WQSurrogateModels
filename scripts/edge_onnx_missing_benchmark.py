"""Inference-only worker for metadata-bound missing-indicator ONNX graphs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import onnxruntime as ort
from edge_optimization_paths import artifact_path, atomic_json
import hashlib


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_contract(contract, mode, missing, columns):
    if contract.get("mode") != mode or contract.get("missing") != missing:
        raise ValueError("ONNX graph missing-condition metadata does not match request")
    if contract.get("input_features") != columns or columns != [
        "DO",
        "BOD",
        "NH3N",
        "EC",
        "SS",
    ]:
        raise ValueError("ONNX graph input feature order does not match packet")
    if contract.get("available_features") != [c for c in columns if c not in missing]:
        raise ValueError("ONNX graph available feature order does not match request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["inference_dropout", "reduced_retraining", "indicator_reconstruction"],
        required=True,
    )
    parser.add_argument("--missing", nargs="+", choices=["BOD", "NH3N"], required=True)
    args = parser.parse_args()
    output = artifact_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    began = time.monotonic()

    def progress(phase, **values):
        record = dict(phase=phase, elapsed_seconds=time.monotonic() - began, **values)
        atomic_json(output / "progress.json", record)
        print(json.dumps(record), flush=True)

    progress("verifying_inputs")
    model_sha = sha(args.model)
    input_sha = sha(args.inputs)
    data = np.load(args.inputs, allow_pickle=False)
    X = data["X"].astype(np.float32)
    columns = data["columns"].tolist()
    progress("loading_onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    model = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_s = time.perf_counter() - started
    contract = json.loads(model.get_modelmeta().custom_metadata_map["wq_contract"])
    validate_contract(contract, args.mode, args.missing, columns)
    progress("model_loaded", load_s=load_s)
    chunks = []
    for i in range(0, len(X), 128):
        progress("predicting_chunk", completed_rows=i, total_rows=len(X))
        chunks.append(model.run(["score"], {"features": X[i : i + 128]})[0].reshape(-1))
    predictions = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite prediction")
    np.savez_compressed(output / "predictions.npz", predicted=predictions)
    # The same packet tests x86 ONNX portability and native numerical parity.
    parity = {}
    for name in ["x86_onnx", "native64"]:
        delta = np.abs(predictions - data[name])
        parity[name] = dict(
            rows=len(delta),
            above_1e_6=int((delta > 1e-6).sum()),
            max_abs_difference=float(delta.max()),
            mean_abs_difference=float(delta.mean()),
        )
    poisoned = X[:128].copy()
    for indicator in args.missing:
        poisoned[:, columns.index(indicator)] = np.nan
    poison_pred = model.run(["score"], {"features": poisoned})[0].reshape(-1)
    if not np.array_equal(poison_pred, predictions[: len(poisoned)]):
        raise ValueError("Unavailable features leaked into prediction")
    timing = []
    for size in [1, 128]:
        for _ in range(3):
            model.run(["score"], {"features": X[:size]})
        samples = []
        for repeat in range(300):
            if repeat % 20 == 0:
                progress(
                    "timing",
                    batch_rows=size,
                    completed_repeats=repeat,
                    total_repeats=300,
                )
            started = time.perf_counter_ns()
            model.run(["score"], {"features": X[:size]})
            samples.append((time.perf_counter_ns() - started) / 1e6)
        timing.append(
            dict(
                rows=size,
                repeats=300,
                p50_ms=float(np.percentile(samples, 50)),
                p95_ms=float(np.percentile(samples, 95)),
                samples_ms=samples,
            )
        )
    atomic_json(
        output / "result.json",
        dict(
            mode=args.mode,
            missing=args.missing,
            machine=platform.machine(),
            platform=platform.platform(),
            python=sys.version,
            onnxruntime=ort.__version__,
            model_sha256=model_sha,
            input_sha256=input_sha,
            model_bytes=args.model.stat().st_size,
            process_cold_load_s=load_s,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            parity=parity,
            timing=timing,
            missing_column_poison_check=True,
            caveat="Checksum-warmed file cache, shared Nano API, fixed seed0; no OS cold-start or isolated-device claim",
        ),
    )
    progress("completed")


if __name__ == "__main__":
    main()
