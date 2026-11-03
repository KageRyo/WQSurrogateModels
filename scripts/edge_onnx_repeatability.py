"""Bounded caller probes saved ONNX predictions across batch/optimizer settings."""

from pathlib import Path
import argparse
import json
import resource
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = artifact_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = np.load(args.inputs, allow_pickle=False)
    X = data["X"].astype(np.float32)
    runs = {}
    records = []
    for label, batch, disable in [
        ("batch128_a", 128, False),
        ("batch128_b", 128, False),
        ("batch1", 1, False),
        ("batch128_no_optimization", 128, True),
    ]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if disable:
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        model = ort.InferenceSession(
            str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        chunks = []
        started = time.perf_counter()
        for i in range(0, len(X), batch):
            if i % 1024 == 0:
                record = dict(
                    phase="predicting",
                    variant=label,
                    completed_rows=i,
                    total_rows=len(X),
                )
                atomic_json(output / "progress.json", record)
                print(json.dumps(record), flush=True)
            chunks.append(
                model.run(None, {"features": X[i : i + batch]})[0].reshape(-1)
            )
        pred = np.concatenate(chunks).astype(np.float64)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        runs[label] = pred
        delta = np.abs(pred - data["reference"])
        records.append(
            dict(
                variant=label,
                batch=batch,
                disabled_optimization=disable,
                elapsed_s=time.perf_counter() - started,
                rows=len(pred),
                reference_above_1e_6=int((delta > 1e-6).sum()),
                reference_max_abs_difference=float(delta.max()),
            )
        )
        del model
    np.savez_compressed(output / "predictions.npz", **runs)
    atomic_json(
        output / "result.json",
        dict(
            runtime=ort.__version__,
            model_sha256=sha(args.model),
            inputs_sha256=sha(args.inputs),
            runs=records,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope="Post-hoc numerical repeatability; not a performance benchmark",
        ),
    )


if __name__ == "__main__":
    main()
