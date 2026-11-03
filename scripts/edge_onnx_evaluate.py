"""Convert an isolated, fitted complete-input RF pipeline and retain parity findings."""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from threadpoolctl import threadpool_limits

from edge_optimization import (
    FEATURES,
    categories,
    metrics,
    sha,
    utc,
    verified,
    write_json,
)
from edge_optimization_paths import artifact_path


def convert_pipeline(pipeline):
    return convert_sklearn(
        pipeline,
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=17,
    )


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def compare(native64, native32, converted, tolerance=1e-6):
    result = {}
    for name, left, right in [
        ("native64_to_native32", native64, native32),
        ("native32_to_onnx32", native32, converted),
        ("native64_to_onnx32", native64, converted),
    ]:
        delta = np.abs(np.asarray(left) - np.asarray(right))
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite parity result")
        result[name] = dict(
            max_abs_difference=float(delta.max()),
            mean_abs_difference=float(delta.mean()),
            rows=int(len(delta)),
            above_tolerance=int((delta > tolerance).sum()),
            tolerance=tolerance,
            category_changes=int((categories(left) != categories(right)).sum()),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = artifact_path(args.output)
    if not verified(args.job):
        raise ValueError("Unverified source job")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = args.job / "full_reference.joblib"
    write_json(
        args.output / "source.json",
        dict(
            path=str(source.resolve()),
            sha256=sha(source),
            utc=utc(),
            onnx=onnx.__version__,
            onnxruntime=ort.__version__,
        ),
    )
    try:
        with threadpool_limits(limits=1):
            pipeline = joblib.load(source, mmap_mode="r")["wqi_model"]
            started = time.perf_counter()
            graph = convert_pipeline(pipeline)
            onnx.checker.check_model(graph)
            path = args.output / "pipeline.onnx"
            path.write_bytes(graph.SerializeToString())
            conversion_s = time.perf_counter() - started
            del graph
            started = time.perf_counter()
            runtime = session(path)
            load_s = time.perf_counter() - started
            frame = pd.read_csv(args.data_root / "data/dataV1.csv")
            split = np.load(args.job / "splits.npz", allow_pickle=False)
            rows = []
            for label, idx in [
                ("internal_test", split["internal_test"]),
                ("external_10714", split["external"]),
            ]:
                probe = frame.iloc[idx][FEATURES]
                X32 = probe.astype(np.float32)
                native64 = pipeline.predict(probe)
                native32 = pipeline.predict(X32)
                converted = (
                    runtime.run(None, {"features": X32.to_numpy()})[0]
                    .reshape(-1)
                    .astype(np.float64)
                )
                np.savez_compressed(
                    args.output / (label + ".npz"),
                    row_id=idx,
                    actual=frame.iloc[idx].Score,
                    native64=native64,
                    native32=native32,
                    onnx32=converted,
                )
                rows.append(
                    dict(
                        source=label,
                        parity=compare(native64, native32, converted),
                        onnx_metrics=metrics(frame.iloc[idx].Score, converted),
                    )
                )
            # Exercise the fitted imputer in its actual ONNX forward path.
            missing = frame.iloc[split["external"][:128]][FEATURES].copy()
            missing.loc[missing.index[::2], "BOD"] = np.nan
            missing.loc[missing.index[::3], "NH3N"] = np.nan
            X32 = missing.astype(np.float32)
            converted = (
                runtime.run(None, {"features": X32.to_numpy()})[0]
                .reshape(-1)
                .astype(np.float64)
            )
            rows.append(
                dict(
                    source="missing_input_probes",
                    parity=compare(
                        pipeline.predict(missing), pipeline.predict(X32), converted
                    ),
                )
            )
            timings = []
            for n in [1, 128]:
                X = frame.iloc[split["external"][:n]][FEATURES].to_numpy(
                    dtype=np.float32
                )
                for _ in range(3):
                    runtime.run(None, {"features": X})
                samples = []
                for _ in range(30):
                    started = time.perf_counter_ns()
                    runtime.run(None, {"features": X})
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                timings.append(
                    dict(
                        rows=n,
                        p50_ms=float(np.percentile(samples, 50)),
                        p95_ms=float(np.percentile(samples, 95)),
                        samples_ms=samples,
                    )
                )
            result = dict(
                phase="completed_with_findings"
                if any(
                    v["above_tolerance"] for row in rows for v in row["parity"].values()
                )
                else "completed_parity_pass",
                utc=utc(),
                parity=rows,
                timing=timings,
                conversion_s=conversion_s,
                session_load_s=load_s,
                onnx_bytes=path.stat().st_size,
                native_bytes=source.stat().st_size,
                peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                caveat="Process high-water RSS includes conversion and both runtimes; not inference-only RSS",
                sha256={"pipeline.onnx": sha(path)},
            )
            write_json(args.output / "result.json", result)
    except BaseException as exc:
        write_json(
            args.output / "failure.json",
            dict(
                phase="blocked",
                utc=utc(),
                exception=type(exc).__name__,
                message=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    main()
