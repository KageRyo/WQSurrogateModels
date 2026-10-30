"""Measure one model per process; run HTTP measurements separately from raw inference.

Use the same script, inputs and locked dependencies on x86 and ARM64. Output is
created exclusively, so an existing experiment is never silently overwritten.
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
import statistics
import time
from urllib.request import Request, urlopen


# Synthetic complete-input probes spanning clean to polluted conditions. Errors
# on these probes describe deployment behaviour, not held-out model accuracy.
PROBES = [
    {"DO": 96.2, "BOD": 1.5, "NH3N": 0.22, "EC": 171.0, "SS": 2.6},
    {"DO": 85.0, "BOD": 2.0, "NH3N": 0.1, "EC": 400.0, "SS": 10.0},
    {"DO": 65.0, "BOD": 4.0, "NH3N": 1.0, "EC": 750.0, "SS": 40.0},
    {"DO": 40.0, "BOD": 10.0, "NH3N": 3.0, "EC": 1500.0, "SS": 100.0},
    {"DO": 20.0, "BOD": 20.0, "NH3N": 6.0, "EC": 2500.0, "SS": 300.0},
]


def summary(values):
    ordered = sorted(values)
    # Linear interpolation, equivalent to NumPy's default percentile method.
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["direct_wqi5", "rf", "lightgbm", "xgboost"])
    parser.add_argument("--api-url", help="Omit for raw inference; use loopback on Nano for device API timing")
    parser.add_argument("--request-timeout-seconds", type=float, default=300, help="Allow cold artifact loading on a memory-constrained device")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=0, help="Minimum measured duration, in addition to sample count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Raw x86 output for this same model")
    parser.add_argument("--parity-atol", type=float, default=1e-6)
    parser.add_argument("--power-mode", required=True, help="Record measured nvpmodel output; never inferred from CPU load")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.duration_seconds < 0 or args.parity_atol < 0 or args.request_timeout_seconds <= 0:
        parser.error("Invalid sample, warmup, duration or tolerance")
    if args.output.exists():
        parser.error("Output already exists; choose a new experiment filename")

    # Imports count towards process memory but are outside raw model-load timing.
    from wqsurrogatemodels.wqi import direct_wqi5_score

    baseline = [direct_wqi5_score(**{k.lower(): v for k, v in row.items()}) for row in PROBES]
    artifact = None
    model_settings = None
    load_start = time.perf_counter()
    if args.api_url:
        def predict(row):
            request = Request(
                args.api_url.rstrip("/") + "/api/v2/assessment",
                data=json.dumps({**row, "model_type": args.model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=args.request_timeout_seconds) as response:
                result = json.load(response)
            if result["model_type"] != args.model:
                raise RuntimeError("API returned a different model")
            return float(result["score"])
    elif args.model == "direct_wqi5":
        def predict(row):
            return direct_wqi5_score(**{k.lower(): v for k, v in row.items()})
    else:
        import pandas as pd
        from wqsurrogatemodels.enums import ModelTypeEnum
        from wqsurrogatemodels.services import WaterQualityService
        from wqsurrogatemodels.settings import FEATURE_COLUMNS

        service = WaterQualityService()
        kind = ModelTypeEnum(args.model)
        artifact_path = service._pick_artifact(kind)
        load_start = time.perf_counter()
        model = service._load_model(kind)
        load_ms = (time.perf_counter() - load_start) * 1000
        digest = hashlib.sha256()
        with artifact_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        artifact = {"bytes": artifact_path.stat().st_size, "sha256": digest.hexdigest()}
        model_settings = {k: str(v) for k, v in model.get_params().items() if k.split("__")[-1] in {"n_jobs", "nthread", "num_threads"}}
        # Prebuild one-row frames: raw inference excludes DataFrame construction.
        frames = {tuple(row.values()): pd.DataFrame([row], columns=FEATURE_COLUMNS) for row in PROBES}

        def predict(row):
            return float(model.predict(frames[tuple(row.values())])[0])

    if args.api_url or args.model == "direct_wqi5":
        load_ms = None
    first_start = time.perf_counter()
    first_prediction = predict(PROBES[0])
    first_ms = (time.perf_counter() - first_start) * 1000
    predictions = [predict(row) for row in PROBES]
    for i in range(args.warmup):
        predict(PROBES[i % len(PROBES)])

    cpu_start = time.process_time()
    start = time.perf_counter()
    latencies = []
    while len(latencies) < args.samples or time.perf_counter() - start < args.duration_seconds:
        before = time.perf_counter()
        predict(PROBES[len(latencies) % len(PROBES)])
        latencies.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    result = {
        "schema_version": 1,
        "model": args.model,
        "measurement": "http_end_to_end" if args.api_url else "raw_inference",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "power_mode": args.power_mode,
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "thread_environment": {name: os.getenv(name) for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "model_thread_settings": model_settings, "artifact": artifact,
        "probes": PROBES, "predictions": predictions, "direct_wqi5": baseline,
        "probe_mae_vs_formula": statistics.mean(abs(a-b) for a,b in zip(predictions, baseline)),
        "warmup": args.warmup, "model_load_ms": load_ms,
        "first_prediction_ms": first_ms, "first_prediction": first_prediction,
        "latency": summary(latencies), "elapsed_seconds": elapsed,
        "throughput_per_second": len(latencies) / elapsed,
        "client_or_raw_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
        "client_or_raw_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "api_url": args.api_url,
    }
    failed = False
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["model"] != args.model or reference["probes"] != PROBES or reference["measurement"] != "raw_inference":
            raise ValueError("Reference must use the same model/probes and raw inference")
        if not args.api_url and reference["artifact"] != artifact:
            raise ValueError("Reference artifact hash/size differs")
        if len(reference["predictions"]) != len(predictions):
            raise ValueError("Reference prediction count differs")
        delta = max(abs(a-b) for a,b in zip(predictions, reference["predictions"]))
        # HTTP scores are rounded to three decimals by the public API contract.
        tolerance = max(args.parity_atol, 0.000501) if args.api_url else args.parity_atol
        result["reference_max_abs_difference"] = delta
        result["reference_tolerance"] = tolerance
        result["reference_parity_pass"] = delta <= tolerance
        failed = delta > tolerance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "latency": result["latency"], "parity_failed": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
