"""Convert fitted missing-indicator bundles into one self-contained ONNX graph.

No training or promotion. Full five-column input is reduced/masked inside the
 graph, so values in unavailable columns cannot leak into predictions.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import onnx
from onnx import compose, helper, numpy_helper, TensorProto
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from threadpoolctl import threadpool_limits

from edge_optimization import FEATURES, predict, sha, utc, verified, write_json, metrics
from edge_onnx_evaluate import compare, session
from edge_optimization_paths import artifact_path

MODES = ("inference_dropout", "reduced_retraining", "indicator_reconstruction")


def convert_bundle(bundle, mode, source_sha):
    if mode not in MODES:
        raise ValueError("Unsupported missing-input mode")
    missing = list(bundle["missing_indicators"])
    available = [c for c in FEATURES if c not in missing]
    if (
        not missing
        or len(set(missing)) != len(missing)
        or any(c not in FEATURES for c in missing)
    ):
        raise ValueError("Invalid missing indicators")
    if bundle["available_features"] != available:
        raise ValueError("Available feature order does not match training contract")
    if mode == "indicator_reconstruction" and set(bundle["indicator_models"]) != set(
        missing
    ):
        raise ValueError("Reconstruction stages do not match missing indicators")
    nodes, initializers = [], []
    opsets = {"": 17}
    ir_versions = []

    def gather(input_name, indices, output):
        name = output + "_indices"
        initializers.append(
            numpy_helper.from_array(np.asarray(indices, dtype=np.int64), name)
        )
        nodes.append(helper.make_node("Gather", [input_name, name], [output], axis=1))

    gather("features", [FEATURES.index(c) for c in available], "available")

    def stage(model, width, name, input_name):
        graph = convert_sklearn(
            model,
            initial_types=[("features", FloatTensorType([None, width]))],
            target_opset=17,
        )
        if graph.functions:
            raise ValueError(
                "Local ONNX functions require explicit composition support"
            )
        ir_versions.append(graph.ir_version)
        compose.add_prefix(graph, name + "/", inplace=True)
        nodes.append(
            helper.make_node("Identity", [input_name], [graph.graph.input[0].name])
        )
        nodes.extend(graph.graph.node)
        initializers.extend(graph.graph.initializer)
        for op in graph.opset_import:
            if op.domain in opsets and opsets[op.domain] != op.version:
                raise ValueError("Incompatible operator sets between stages")
            opsets[op.domain] = op.version
        return graph.graph.output[0].name

    columns = []
    if mode == "reduced_retraining":
        model_input = "available"
        width = len(available)
    else:
        for index, feature in enumerate(FEATURES):
            name = "column_" + feature
            if feature not in missing:
                gather("features", [index], name)
            elif mode == "indicator_reconstruction":
                reconstructed = stage(
                    bundle["indicator_models"][feature],
                    len(available),
                    "reconstruct_" + feature,
                    "available",
                )
                nodes.append(helper.make_node("Identity", [reconstructed], [name]))
            else:
                gather("features", [index], name + "_shape_source")
                nodes.append(
                    helper.make_node(
                        "Shape", [name + "_shape_source"], [name + "_shape"]
                    )
                )
                nodes.append(
                    helper.make_node(
                        "ConstantOfShape",
                        [name + "_shape"],
                        [name],
                        value=helper.make_tensor(
                            "nan", TensorProto.FLOAT, [1], [float("nan")]
                        ),
                    )
                )
            columns.append(name)
        model_input = "model_features"
        width = len(FEATURES)
        nodes.append(helper.make_node("Concat", columns, [model_input], axis=1))
    score = stage(bundle["wqi_model"], width, "wqi", model_input)
    nodes.append(helper.make_node("Identity", [score], ["score"]))
    outputs = [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])]
    if mode == "indicator_reconstruction":
        outputs.append(
            helper.make_tensor_value_info(
                "model_features", TensorProto.FLOAT, [None, len(FEATURES)]
            )
        )
    graph = helper.make_graph(
        nodes,
        "missing_indicator_bundle",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, [None, len(FEATURES)]
            )
        ],
        outputs,
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="WQSurrogateModels.research",
        opset_imports=[helper.make_opsetid(k, v) for k, v in opsets.items()],
    )
    model.ir_version = max(ir_versions)
    helper.set_model_props(
        model,
        {
            "wq_contract": json.dumps(
                dict(
                    mode=mode,
                    missing=missing,
                    available_features=available,
                    input_features=list(FEATURES),
                    source_sha256=source_sha,
                ),
                sort_keys=True,
            )
        },
    )
    onnx.checker.check_model(model)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--complete-job", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = artifact_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    began = time.monotonic()

    def progress(phase, **values):
        record = dict(
            phase=phase, utc=utc(), elapsed_seconds=time.monotonic() - began, **values
        )
        write_json(output / "progress.json", record)
        print(json.dumps(record), flush=True)

    try:
        progress("verifying_source")
        if not verified(args.job) or not verified(args.complete_job):
            raise ValueError("Unverified source job")
        source = (
            args.complete_job / "full_reference.joblib"
            if args.mode == "inference_dropout"
            else args.job / (args.mode + ".joblib")
        )
        source_sha = sha(source)
        write_json(
            output / "source.json",
            dict(
                source=str(source.resolve()),
                source_sha256=source_sha,
                job=str(args.job.resolve()),
                job_marker_sha256=sha(args.job / "completed.json"),
                source_commit=__import__("subprocess")
                .check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                )
                .strip(),
                packages={
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "onnx",
                        "onnxruntime",
                        "skl2onnx",
                    ]
                },
                mode=args.mode,
                tolerance=1e-6,
            ),
        )
        bundle = joblib.load(source, mmap_mode="r")
        if args.mode == "inference_dropout":
            condition = json.loads((args.job / "job.json").read_text())["missing"]
            missing_sets = {
                "missing_bod": ["BOD"],
                "missing_nh3n": ["NH3N"],
                "missing_bod_nh3n": ["BOD", "NH3N"],
            }
            missing = missing_sets[condition]
            bundle = {
                **bundle,
                "missing_indicators": missing,
                "available_features": [c for c in FEATURES if c not in missing],
            }
        with threadpool_limits(limits=1):
            progress("converting", source_bytes=source.stat().st_size)
            started = time.perf_counter()
            graph = convert_bundle(bundle, args.mode, source_sha)
            path = output / "pipeline.onnx"
            path.write_bytes(graph.SerializeToString())
            conversion_s = time.perf_counter() - started
            del graph
            gc.collect()
            progress("loading_onnx", onnx_bytes=path.stat().st_size)
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
                progress("native64_prediction", source=label)
                native64 = predict(bundle, probe, args.mode)
                progress("native32_prediction", source=label)
                native32 = predict(bundle, probe.astype(np.float32), args.mode)
                converted = []
                reconstructed = []
                for i in range(0, len(probe), 128):
                    progress(
                        "onnx_chunk",
                        source=label,
                        completed_rows=i,
                        total_rows=len(probe),
                    )
                    outputs = runtime.run(
                        None,
                        {
                            "features": probe.iloc[i : i + 128].to_numpy(
                                dtype=np.float32
                            )
                        },
                    )
                    converted.append(outputs[0].reshape(-1))
                    if len(outputs) > 1:
                        reconstructed.append(outputs[1])
                converted = np.concatenate(converted).astype(np.float64)
                # Poison only unavailable columns: the graph must ignore their values.
                poison = probe.iloc[:128].copy()
                poison[bundle["missing_indicators"]] = 999999.0
                poison_pred = runtime.run(
                    ["score"], {"features": poison.to_numpy(dtype=np.float32)}
                )[0].reshape(-1)
                if not np.array_equal(poison_pred, converted[: len(poison)]):
                    raise ValueError("Unavailable-column values leak into ONNX output")
                archived = np.load(
                    args.job / (args.mode + "-" + label + ".npz"), allow_pickle=False
                )["predicted"]
                parity = compare(native64, native32, converted)
                archived_delta = np.abs(native64 - archived)
                values = dict(
                    row_id=idx,
                    actual=frame.iloc[idx].Score.to_numpy(),
                    native64=native64,
                    native32=native32,
                    onnx32=converted,
                    archived_native64=archived,
                )
                reconstruction_parity = {}
                if reconstructed:
                    restored = np.concatenate(reconstructed)
                    values["onnx_reconstructed_features"] = restored
                    for indicator in bundle["missing_indicators"]:
                        native_indicator = bundle["indicator_models"][
                            indicator
                        ].predict(probe[bundle["available_features"]])
                        values["native_reconstructed_" + indicator] = native_indicator
                        delta = np.abs(
                            native_indicator - restored[:, FEATURES.index(indicator)]
                        )
                        reconstruction_parity[indicator] = dict(
                            max_abs_difference=float(delta.max()),
                            above_1e_6=int((delta > 1e-6).sum()),
                        )
                np.savez_compressed(output / (label + ".npz"), **values)
                rows.append(
                    dict(
                        source=label,
                        parity=parity,
                        archived_native_max_abs_difference=float(archived_delta.max()),
                        missing_column_poison_check=True,
                        reconstruction_parity=reconstruction_parity,
                        onnx_metrics=metrics(frame.iloc[idx].Score, converted),
                    )
                )
            write_json(
                output / "result.json",
                dict(
                    phase="completed_with_findings"
                    if any(
                        row["parity"]["native64_to_onnx32"]["above_tolerance"]
                        for row in rows
                    )
                    else "completed_parity_pass",
                    utc=utc(),
                    mode=args.mode,
                    missing=bundle["missing_indicators"],
                    parity=rows,
                    conversion_s=conversion_s,
                    session_load_s=load_s,
                    onnx_bytes=path.stat().st_size,
                    source_bytes=source.stat().st_size,
                    peak_process_rss_kib=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    caveat="Conversion process includes native and ONNX runtimes; not inference-only RSS",
                    sha256={
                        path.name: sha(path),
                        **{f.name: sha(f) for f in output.glob("*.npz")},
                    },
                ),
            )
            progress("completed")
    except BaseException as exc:
        write_json(
            output / "failure.json",
            dict(
                phase="failed",
                utc=utc(),
                exception=type(exc).__name__,
                message=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    main()
