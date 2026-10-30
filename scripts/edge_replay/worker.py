"""Run one replay job without keeping multiple large estimators resident.

Reconstruction estimators run in separate processes, followed by the WQI
estimator. This preserves predictions and avoids loading several gigabyte RF
estimators simultaneously on a 2GB device. Stage timings are summed separately
from orchestration/cold-start time; they are never called API latency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from edge_replay.schema import ReplayJob

FEATURES = ["DO", "BOD", "NH3N", "EC", "SS"]


def runtime_versions():
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}
    for name in ["xgboost", "lightgbm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cpu_estimator(model):
    # Original artifacts can record a GPU backend. Change only device selection;
    # preserve learned parameters, imputers, scalers and feature order.
    params = model.get_params(deep=True)
    updates = {}
    for key in params:
        if key.split("__")[-1] in {"device", "device_type"}:
            updates[key] = "cpu"
    if updates:
        model.set_params(**updates)
    return updates


def component_job(packet: Path, output: Path, key: str):
    import joblib

    job = ReplayJob.model_validate_json((packet / "job.json").read_text())
    component = next(item for item in job.components if item.key == key)
    artifact = packet / component.path
    if sha256(artifact) != component.sha256:
        raise ValueError("Component artifact checksum mismatch")
    if sha256(packet / "inputs.npz") != job.inputs_sha256:
        raise ValueError("Replay input checksum mismatch")
    load_start = time.perf_counter()
    model = joblib.load(artifact)
    load_seconds = time.perf_counter() - load_start
    backend_updates = cpu_estimator(model)
    data = np.load(packet / "inputs.npz", allow_pickle=False)
    predictions = {}
    timing = []
    process_cpu_start = time.process_time()
    start = time.perf_counter()
    for condition in job.conditions:
        for frame_spec in job.frames:
            frame = pd.DataFrame(data[frame_spec.name + "/x"], columns=FEATURES)
            if key == "wqi":
                for indicator in condition.missing_indicators:
                    if indicator not in condition.reconstruction:
                        frame[indicator] = np.nan
                for indicator, dependency in condition.reconstruction.items():
                    with np.load(output / f"{dependency}.npz", allow_pickle=False) as reconstructed:
                        frame[indicator] = reconstructed[condition.name + "/" + frame_spec.name]
            selected = frame[component.features]
            before = time.perf_counter()
            result = np.asarray(model.predict(selected), dtype=np.float64)
            elapsed = time.perf_counter() - before
            if len(result) != frame_spec.n_rows or not np.isfinite(result).all():
                raise ValueError("Non-finite prediction or incorrect row count")
            prediction_key = condition.name + "/" + frame_spec.name
            predictions[prediction_key] = result
            timing.append({"condition": condition.name, "frame": frame_spec.name,
                           "stage": key, "phase": "accuracy", "repeat": 0,
                           "n_rows": len(result), "seconds": elapsed})
            if frame_spec.kind == "external_10714" or job.protocol in {"complete_input", "reduced_indicator", "sample_size_experiments"}:
                for _ in range(job.warmup):
                    model.predict(selected)
                for repeat in range(job.repeats):
                    before = time.perf_counter()
                    model.predict(selected)
                    timing.append({"condition": condition.name, "frame": frame_spec.name,
                                   "stage": key, "phase": "timing", "repeat": repeat,
                                   "n_rows": len(selected), "seconds": time.perf_counter() - before})
    # Exclusivity is checked before NumPy opens its own output handle.
    with (output / f"{key}.npz").open("xb") as stream:
        np.savez_compressed(stream, **predictions)
    pd.DataFrame(timing).to_csv(output / f"{key}-timing.csv", index=False, mode="x")
    exclusive_json(output / f"{key}-resources.json", {
        "component": key, "load_seconds": load_seconds,
        "inference_wall_seconds": time.perf_counter() - start,
        "inference_cpu_seconds": time.process_time() - process_cpu_start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu_backend_updates": backend_updates,
        "thread_parameters": {k: str(v) for k,v in model.get_params().items() if k.split("__")[-1] in {"n_jobs", "nthread", "num_threads"}},
    })


def replay(packet: Path, output: Path, reference: Path | None, *, record_parity_failures=False):
    from run_missing_indicator_robustness_experiments import prediction_metrics, score_to_category
    from run_stress107_event_windows import append_window_rows, sequential_windows

    job = ReplayJob.model_validate_json((packet / "job.json").read_text())
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    components = [item for item in job.components if item.key != "wqi"]
    components += [item for item in job.components if item.key == "wqi"]
    for component in components:
        subprocess.run([sys.executable, str(Path(__file__).parents[1] / "replay_edge_experiments.py"),
                        "worker", "--packet", str(packet), "--output", str(output),
                        "--component", component.key], check=True)
    predictions = np.load(output / "wqi.npz", allow_pickle=False)
    inputs = np.load(packet / "inputs.npz", allow_pickle=False)
    metrics = []
    window_rows = []
    parity_rows = []
    global_stress_rows = []
    if reference:
        reference_manifest = json.loads((reference / "completed.json").read_text())
        if reference_manifest["job_sha256"] != sha256(packet / "job.json"):
            raise ValueError("Reference uses a different replay job")
    reference_data = np.load(reference / "wqi.npz", allow_pickle=False) if reference else None
    for condition in job.conditions:
        for spec in job.frames:
            key = condition.name + "/" + spec.name
            predicted = predictions[key]
            actual = inputs[spec.name + "/y"]
            if spec.kind in {"internal_test", "external_10714", "train", "test"}:
                metrics.append({"protocol": job.protocol, "source": spec.kind,
                                "experiment": condition.name, "missing_set": condition.missing_set,
                                "experiment_mode": condition.mode, "seed": job.seed,
                                "model_type": job.model_type, "n": len(actual),
                                **prediction_metrics(actual, predicted)})
            if spec.kind == "stress107":
                baseline = predictions[condition.name + "/external_10714"]
                append_window_rows(window_rows, missing_set=condition.missing_set,
                                   mode=condition.mode, seed=job.seed, model_type=job.model_type,
                                   scenario_name=spec.scenario, severity_name=spec.severity,
                                   severity_rank=spec.severity_rank, perturbation_pct=spec.perturbation_pct,
                                   multipliers=spec.multipliers,
                                   windows=sequential_windows(spec.n_rows, job.windows),
                                   y_true=actual, baseline_pred=baseline, stress_pred=predicted)
            if spec.kind == "stress":
                from run_missing_indicator_experiments import category_rank, score_to_category
                baseline = predictions[condition.name + "/external_10714"]
                before = score_to_category(baseline)
                after = score_to_category(predicted)
                global_stress_rows.append({
                    "experiment": condition.name, "seed": job.seed,
                    "model_type": job.model_type, "stress_scenario": spec.scenario,
                    "n": len(predicted), "baseline_mean_score": float(np.mean(baseline)),
                    "stress_mean_score": float(np.mean(predicted)),
                    "mean_delta_stress_minus_baseline": float(np.mean(predicted-baseline)),
                    "median_delta_stress_minus_baseline": float(np.median(predicted-baseline)),
                    "pct_score_decreased": float(np.mean(predicted < baseline)),
                    "pct_category_worse": float(np.mean([category_rank(a)<category_rank(b) for a,b in zip(after,before)])),
                })
            if reference_data is not None:
                expected = reference_data[key]
                if expected.shape != predicted.shape:
                    raise ValueError("Reference prediction shape differs")
                delta = np.abs(expected - predicted)
                parity_rows.append({"experiment": condition.name, "frame": spec.name,
                                    "n_rows": len(delta), "max_abs_difference": float(delta.max()),
                                    "mean_abs_difference": float(delta.mean()),
                                    "category_disagreements": int(np.count_nonzero(np.asarray(score_to_category(expected)) != np.asarray(score_to_category(predicted)))),
                                    "tolerance": 1e-6, "pass": bool(np.all(delta <= 1e-6))})
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False, mode="x")
    if window_rows:
        pd.DataFrame(window_rows).to_csv(output / "stress107-windows.csv", index=False, mode="x")
    if global_stress_rows:
        pd.DataFrame(global_stress_rows).to_csv(output / "global-stress.csv", index=False, mode="x")
    if parity_rows:
        pd.DataFrame(parity_rows).to_csv(output / "parity.csv", index=False, mode="x")
    exclusive_json(output / "completed.json", {
        "job_id": job.job_id, "protocol": job.protocol,
        "elapsed_seconds_including_subprocesses_and_load": time.perf_counter() - started,
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": runtime_versions(),
        "thread_environment": {name: os.getenv(name) for name in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "job_sha256": sha256(packet / "job.json"),
        "worker_sha256": sha256(Path(__file__)),
        "container_image_id": os.getenv("REPLAY_IMAGE_ID"),
        "driver_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "largest_child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "all_reference_parity_pass": all(row["pass"] for row in parity_rows) if parity_rows else None,
        "parity_findings_recorded": record_parity_failures,
    })
    if any(not row["pass"] for row in parity_rows) and not record_parity_failures:
        raise SystemExit("Prediction parity failed; results retained for investigation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--component")
    parser.add_argument("--record-parity-failures", action="store_true")
    args = parser.parse_args()
    if args.component:
        component_job(args.packet, args.output, args.component)
    else:
        replay(args.packet, args.output, args.reference, record_parity_failures=args.record_parity_failures)
