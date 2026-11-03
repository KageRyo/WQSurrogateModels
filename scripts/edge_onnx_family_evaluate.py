"""Bounded caller evaluates existing full-input production models without fitting."""

from pathlib import Path
import argparse
import json
import resource
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import onnx
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from threadpoolctl import threadpool_limits
from edge_optimization import FEATURES, sha, utc, write_json, metrics
from edge_onnx_evaluate import compare, session
from edge_optimization_paths import artifact_path


def register_external():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
        convert_lightgbm,
    )
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

    update_registered_converter(
        LGBMRegressor,
        "LightGbmLGBMRegressor",
        calculate_linear_regressor_output_shapes,
        convert_lightgbm,
    )
    update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--model", choices=["lr", "mpr", "svm", "lightgbm", "xgboost"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = artifact_path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest = args.data_root / "models/production_model_manifest.json"
    entry = next(
        e
        for e in json.loads(manifest.read_text())["artifacts"]
        if e["model_type"] == args.model
    )
    source = args.data_root / entry["production_artifact"]
    assert sha(source) == entry["sha256"]
    write_json(
        out / "source.json",
        dict(
            artifact=entry,
            manifest_sha256=sha(manifest),
            packages={
                k: importlib.metadata.version(k)
                for k in [
                    "onnx",
                    "onnxruntime",
                    "skl2onnx",
                    "onnxmltools",
                    "numpy",
                    "scikit-learn",
                    "xgboost",
                    "lightgbm",
                ]
            },
            source_script_sha256=sha(Path(__file__)),
            utc=utc(),
            tolerance=1e-6,
            scope="Complete-input production artifact, external rows only; no fitting, missing-bundle or stress claim",
        ),
    )
    try:
        with threadpool_limits(limits=1):
            register_external()
            pipeline = joblib.load(source, mmap_mode="r")
            # Inference is CPU only; original artifact itself is never resaved.
            if args.model == "lightgbm":
                pipeline.set_params(model__device_type="cpu", model__n_jobs=1)
            if args.model == "xgboost":
                pipeline.set_params(model__device="cpu", model__n_jobs=1)
            started = time.perf_counter()
            graph = convert_sklearn(
                pipeline,
                initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
                target_opset={"": 17, "ai.onnx.ml": 3},
            )
            onnx.checker.check_model(graph)
            model = out / "pipeline.onnx"
            model.write_bytes(graph.SerializeToString())
            conversion_s = time.perf_counter() - started
            runtime = session(model)
            frame = pd.read_csv(args.data_root / "data/dataV1.csv").iloc[50000:]
            X = frame[FEATURES]
            native64 = pipeline.predict(X)
            native32 = pipeline.predict(X.astype(np.float32))
            chunks = []
            for i in range(0, len(X), 128):
                write_json(
                    out / "progress.json",
                    dict(
                        phase="onnx_prediction",
                        completed_rows=i,
                        total_rows=len(X),
                        utc=utc(),
                    ),
                )
                chunks.append(
                    runtime.run(
                        None,
                        {"features": X.iloc[i : i + 128].to_numpy(dtype=np.float32)},
                    )[0].reshape(-1)
                )
            converted = np.concatenate(chunks).astype(np.float64)
            np.savez_compressed(
                out / "external_10714.npz",
                row_id=np.arange(50000, 50000 + len(X)),
                native64=native64,
                native32=native32,
                onnx32=converted,
                actual=frame.Score.to_numpy(),
            )
            write_json(
                out / "result.json",
                dict(
                    phase="completed",
                    model=args.model,
                    parity=compare(native64, native32, converted),
                    native_metrics=metrics(frame.Score, native64),
                    onnx_metrics=metrics(frame.Score, converted),
                    conversion_s=conversion_s,
                    onnx_bytes=model.stat().st_size,
                    native_bytes=source.stat().st_size,
                    conversion_process_peak_rss_kib=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    sha256={
                        model.name: sha(model),
                        "external_10714.npz": sha(out / "external_10714.npz"),
                    },
                ),
            )
    except BaseException as exc:
        write_json(
            out / "failure.json",
            dict(
                phase="failed",
                exception=type(exc).__name__,
                message=str(exc),
                utc=utc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
