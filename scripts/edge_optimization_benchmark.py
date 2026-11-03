"""Portable single-process CPU worker for isolated native/ONNX comparisons.

Inputs and model are read-only mounts. No API, training or production mutation.
Process-cold loading follows checksum verification; OS caches are not controlled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from edge_optimization_paths import artifact_path, atomic_json

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


MODES = (
    "full_reference",
    "inference_dropout",
    "reduced_retraining",
    "indicator_reconstruction",
)
INDICATORS = ("DO", "BOD", "NH3N", "EC", "SS")
RETRAINED_MODES = ("reduced_retraining", "indicator_reconstruction")


def validate_request(kind, mode, missing):
    """Refuse requests whose recorded mode/missing labels would not be computed.

    Only complete-input pipelines have an ONNX conversion; an ONNX session would
    otherwise run full inputs while the result claims a missing-value condition.
    """
    if kind == "onnx" and (mode != "full_reference" or missing):
        raise ValueError(
            "ONNX evaluation supports only --mode full_reference without --missing"
        )
    if mode == "full_reference" and missing:
        raise ValueError("--mode full_reference does not accept --missing")
    if mode != "full_reference" and not missing:
        raise ValueError("--mode " + mode + " requires the --missing indicators")
    if len(set(missing)) != len(missing):
        raise ValueError("--missing must not repeat an indicator")
    if any(indicator not in INDICATORS for indicator in missing):
        raise ValueError("--missing contains an unknown water-quality indicator")
    # Emit the fixed domain constants in feature order, never raw variadic CLI
    # strings. This also makes equivalent label sets deterministic in wrappers.
    return [indicator for indicator in INDICATORS if indicator in missing]


def validate_bundle(bundle, mode, missing):
    """Refuse an artifact whose trained missing-value condition differs.

    Reduced and reconstruction bundles compute from their own stored features,
    not from --missing; dropout and full reference require a complete-input
    model, to which only --missing NaNs are applied at inference.
    """
    trained = set(bundle.get("missing_indicators", []))
    expected = set(missing) if mode in RETRAINED_MODES else set()
    if trained != expected:
        raise ValueError(
            "Requested missing indicators do not match model artifact: "
            f"{mode} needs a model trained without {sorted(expected)}, "
            f"but the artifact was trained without {sorted(trained)}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["native", "onnx"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, default="full_reference")
    p.add_argument("--missing", nargs="*", default=[], choices=INDICATORS)
    args = p.parse_args()
    try:
        validate_request(args.kind, args.mode, args.missing)
    except ValueError as exc:
        p.error(str(exc))
    args.output = artifact_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, mode=0o700)
    began = time.monotonic()

    def progress(phase, **values):
        record = dict(phase=phase, elapsed_seconds=time.monotonic() - began, **values)
        atomic_json(args.output / "progress.json", record)
        print(json.dumps(record), flush=True)

    progress("verifying_inputs")
    model_sha = sha(args.model)
    inputs_sha = sha(args.inputs)
    data = np.load(args.inputs, allow_pickle=False)
    X = data["X"]
    columns = list(data["columns"])
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=1):
        started = time.perf_counter()
        progress("loading_model", kind=args.kind)
        if args.kind == "native":
            import joblib

            bundle = joblib.load(args.model, mmap_mode="r")
            validate_bundle(bundle, args.mode, args.missing)
            model = bundle["wqi_model"]
            model.set_params(model__n_jobs=1)
            for indicator_model in bundle.get("indicator_models", {}).values():
                indicator_model.set_params(model__n_jobs=1)

            def predict(values):
                frame = pd.DataFrame(values, columns=columns)
                if args.mode == "inference_dropout":
                    frame[args.missing] = np.nan
                elif args.mode == "reduced_retraining":
                    frame = frame[bundle["available_features"]]
                elif args.mode == "indicator_reconstruction":
                    for name, indicator_model in bundle["indicator_models"].items():
                        frame[name] = indicator_model.predict(
                            frame[bundle["available_features"]]
                        )
                return model.predict(frame)

            version = __import__("sklearn").__version__
        else:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model = ort.InferenceSession(
                str(args.model),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            def predict(values):
                return model.run(None, {"features": values.astype(np.float32)})[
                    0
                ].reshape(-1)

            version = ort.__version__
        load_s = time.perf_counter() - started
        load_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        progress("model_loaded", load_s=load_s)
        # Chunked whole-set parity prevents a large ONNX batch from changing
        # the memory claim relative to the batch-1/128 latency measurements.
        chunks = []
        for i in range(0, len(X), 128):
            progress("predicting_chunk", completed_rows=i, total_rows=len(X))
            chunks.append(predict(X[i : i + 128]))
            progress(
                "chunk_completed",
                completed_rows=min(i + 128, len(X)),
                total_rows=len(X),
            )
        outputs = np.concatenate(chunks)
        if not np.isfinite(outputs).all():
            raise ValueError("Nonfinite runtime prediction")
        np.savez_compressed(args.output / "predictions.npz", predicted=outputs)
        delta = np.abs(outputs - data["reference"])
        timings = []
        for size in [1, 128]:
            progress("timing_warmup", batch_rows=size)
            probe = X[:size]
            for _ in range(3):
                predict(probe)
            samples = []
            for repeat in range(300):
                if repeat % 10 == 0:
                    progress(
                        "timing",
                        batch_rows=size,
                        completed_repeats=repeat,
                        total_repeats=300,
                    )
                started = time.perf_counter_ns()
                predict(probe)
                samples.append((time.perf_counter_ns() - started) / 1e6)
            timings.append(
                dict(
                    rows=size,
                    repeats=300,
                    p50_ms=float(np.percentile(samples, 50)),
                    p95_ms=float(np.percentile(samples, 95)),
                    samples_ms=samples,
                )
            )
        result = dict(
            kind=args.kind,
            mode=args.mode,
            missing=args.missing,
            platform=platform.platform(),
            machine=platform.machine(),
            python=sys.version,
            runtime_version=version,
            model_sha256=model_sha,
            input_sha256=inputs_sha,
            model_bytes=args.model.stat().st_size,
            process_cold_load_s=load_s,
            load_high_water_rss_kib=load_rss,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            parity_rows=len(delta),
            max_abs_difference=float(delta.max()),
            above_1e_6=int((delta > 1e-6).sum()),
            timing=timings,
            caveat="Process-cold only; checksum read warms file cache. Shared host API remains running. Not isolated-device or power-meter results.",
        )
        atomic_json(args.output / "result.json", result)
        progress("completed")


if __name__ == "__main__":
    main()
