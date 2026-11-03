"""Checkpointed, private RF optimization experiments; never promotes models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from edge_optimization_control import run_bounded

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wqsurrogatemodels.settings import FEATURE_COLUMNS as FEATURES
from wqsurrogatemodels.wqi import categorize_score


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def categories(values):
    return np.array([categorize_score(float(v))[0] for v in values])


def metrics(y, pred):
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite prediction")
    return dict(
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        r2=float(r2_score(y, pred)),
        accuracy=float(accuracy_score(categories(y), categories(pred))),
        macro_f1=float(f1_score(categories(y), categories(pred), average="macro")),
    )


def forest(config, trees, X, y):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=trees,
                    random_state=config["forest_random_state"],
                    n_jobs=config["training_jobs"],
                ),
            ),
        ]
    )
    model.fit(X, y)
    model.set_params(model__n_jobs=1)
    return model


def predict(bundle, frame, mode):
    available = bundle["available_features"]
    if mode == "reduced_retraining":
        return bundle["wqi_model"].predict(frame[available])
    X = frame[FEATURES].copy()
    if mode == "inference_dropout":
        X[bundle["missing_indicators"]] = np.nan
    elif mode == "indicator_reconstruction":
        for name, model in bundle["indicator_models"].items():
            X[name] = model.predict(frame[available])
    return bundle["wqi_model"].predict(X)


def perturb(frame, scenario, factor):
    X = frame[FEATURES].copy()
    if scenario in ("organic_pollution", "combined_pollution"):
        X["DO"] *= {1.3: 0.7, 2.0: 0.5, 4.0: 0.3}[factor]
    indicators = {
        "organic_pollution": ["BOD"],
        "ammonia_pollution": ["NH3N"],
        "suspended_solids_event": ["SS"],
        "conductivity_event": ["EC"],
        "combined_pollution": ["BOD", "NH3N", "SS", "EC"],
    }[scenario]
    X[indicators] *= factor
    return X


def job_id(seed, trees, missing):
    return f"s{seed}-rf{trees}-{missing}"


def all_jobs(config):
    # Finish the complete-input matrix first; missing jobs reuse those models.
    for missing in ["complete", *config["missing_sets"]]:
        for seed in config["seeds"]:
            for trees in config["trees"]:
                yield dict(seed=seed, trees=trees, missing=missing)


def verified(directory):
    marker = directory / "completed.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    for name, digest in record["sha256"].items():
        path = directory / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Completed output changed: {path}")
    return True


def sources(data_root, config):
    subset = pd.read_csv(data_root / "data/dataV1_50000.csv")
    full = pd.read_csv(data_root / "data/dataV1.csv")
    if len(subset) != config["expected_train_source_rows"]:
        raise ValueError("Unexpected training-source size")
    pd.testing.assert_frame_equal(
        subset.reset_index(drop=True), full.iloc[: len(subset)].reset_index(drop=True)
    )
    external = full.iloc[len(subset) :].reset_index(drop=True)
    if len(external) != config["expected_external_rows"]:
        raise ValueError("Unexpected external size")
    return subset, external


def run_job(args, config):
    key = job_id(args.seed, args.trees, args.missing)
    target = args.output / "jobs" / key
    if verified(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(key + ".partial-" + str(time.time_ns()))
    stage.mkdir()

    def progress(phase, **values):
        payload = dict(phase=phase, job=key, utc=utc(), **values)
        write_json(stage / "progress.json", payload)
        print(json.dumps(payload), flush=True)

    progress("loading_inputs")
    write_json(
        stage / "job.json",
        dict(seed=args.seed, trees=args.trees, missing=args.missing, utc=utc()),
    )
    subset, external = sources(args.data_root, config)
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=config["test_size"], random_state=args.seed
    )
    train_idx, test_idx = next(split.split(subset[FEATURES], categories(subset.Score)))
    np.savez_compressed(
        stage / "splits.npz",
        train=train_idx,
        internal_test=test_idx,
        external=np.arange(len(subset), len(subset) + len(external)),
    )
    train, test = subset.iloc[train_idx], subset.iloc[test_idx]
    started = time.perf_counter()
    if args.missing == "complete":
        progress("training_full_reference")
        model = forest(config, args.trees, train[FEATURES], train.Score)
        bundles = {
            "full_reference": dict(
                wqi_model=model,
                available_features=list(FEATURES),
                missing_indicators=[],
            )
        }
        joblib.dump(
            bundles["full_reference"], stage / "full_reference.joblib", compress=0
        )
    else:
        base = args.output / "jobs" / job_id(args.seed, args.trees, "complete")
        if not verified(base):
            raise ValueError("Complete-input dependency is not verified")
        base_bundle = joblib.load(base / "full_reference.joblib", mmap_mode="r")
        missing = config["missing_sets"][args.missing]
        available = [c for c in FEATURES if c not in missing]
        common = dict(missing_indicators=missing, available_features=available)
        progress("training_reduced_model")
        reduced = forest(config, args.trees, train[available], train.Score)
        reconstructed = train[FEATURES].copy()
        indicators = {}
        for indicator in missing:
            progress("training_indicator", indicator=indicator)
            indicators[indicator] = forest(
                config, args.trees, train[available], train[indicator]
            )
            reconstructed[indicator] = indicators[indicator].predict(train[available])
        progress("training_reconstruction_stage2")
        stage2 = forest(config, args.trees, reconstructed, train.Score)
        bundles = dict(
            inference_dropout={**base_bundle, **common},
            reduced_retraining=dict(wqi_model=reduced, **common),
            indicator_reconstruction=dict(
                wqi_model=stage2, indicator_models=indicators, **common
            ),
        )
        for mode in ("reduced_retraining", "indicator_reconstruction"):
            joblib.dump(bundles[mode], stage / (mode + ".joblib"), compress=0)
        write_json(
            stage / "dropout-model-reference.json",
            dict(
                path=str(base / "full_reference.joblib"),
                sha256=sha(base / "full_reference.joblib"),
            ),
        )
    training_s = time.perf_counter() - started
    rows, timings, stress = [], [], []
    for mode, bundle in bundles.items():
        for source, frame, indices in [
            ("internal_test", test, test_idx),
            (
                "external_10714",
                external,
                np.arange(len(subset), len(subset) + len(external)),
            ),
        ]:
            progress("predicting", mode=mode, source=source, rows=len(frame))
            pred = predict(bundle, frame, mode)
            np.savez_compressed(
                stage / f"{mode}-{source}.npz",
                row_id=indices,
                actual=frame.Score.to_numpy(),
                predicted=pred,
            )
            rows.append(
                dict(
                    seed=args.seed,
                    trees=args.trees,
                    missing=args.missing,
                    mode=mode,
                    source=source,
                    n=len(frame),
                    **metrics(frame.Score, pred),
                )
            )
        for n in config["timing"]["rows"]:
            progress("timing", mode=mode, batch_rows=n)
            probe = external.iloc[:n]
            for _ in range(config["timing"]["warmup"]):
                predict(bundle, probe, mode)
            measurements = []
            for _ in range(config["timing"]["repeats"]):
                then = time.perf_counter_ns()
                predict(bundle, probe, mode)
                measurements.append((time.perf_counter_ns() - then) / 1e6)
            timings.append(
                dict(
                    mode=mode,
                    rows=n,
                    repeats=len(measurements),
                    p50_ms=float(np.percentile(measurements, 50)),
                    p95_ms=float(np.percentile(measurements, 95)),
                    samples_ms=measurements,
                )
            )
        baseline = predict(bundle, external, mode)
        # Existing stress protocol perturbs each of 107 disjoint windows. As
        # these pipelines are row-independent, predicting the fully perturbed
        # frame once is algebraically equivalent within each affected window.
        for scenario in config["stress107"]["scenarios"]:
            for factor in config["stress107"]["severity_factors"]:
                progress("stress", mode=mode, scenario=scenario, factor=factor)
                changed = predict(bundle, perturb(external, scenario, factor), mode)
                if not np.isfinite(changed).all():
                    raise ValueError("Nonfinite stress prediction")
                for window, idx in enumerate(
                    np.array_split(
                        np.arange(len(external)), config["stress107"]["windows"]
                    )
                ):
                    delta = changed[idx] - baseline[idx]
                    stress.append(
                        dict(
                            mode=mode,
                            scenario=scenario,
                            factor=factor,
                            window=window,
                            n=len(idx),
                            start=int(idx[0]),
                            end_exclusive=int(idx[-1] + 1),
                            baseline_mean=float(baseline[idx].mean()),
                            stress_mean=float(changed[idx].mean()),
                            delta_mean=float(delta.mean()),
                            fraction_decreased=float((delta < 0).mean()),
                        )
                    )
    pd.DataFrame(rows).to_csv(stage / "metrics.csv", index=False)
    pd.DataFrame(stress).to_csv(stage / "stress107.csv", index=False)
    write_json(stage / "timing.json", timings)
    write_json(
        stage / "resources.json",
        dict(
            training_phase_wall_s=training_s,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            caveat="Whole-job high-water RSS includes training; not inference-only or device peak memory",
            model_bytes={p.name: p.stat().st_size for p in stage.glob("*.joblib")},
            utc=utc(),
            platform=platform.platform(),
        ),
    )
    progress("finalizing_outputs")
    write_json(
        stage / "completed.json",
        dict(
            phase="completed",
            utc=utc(),
            sha256={
                str(p.relative_to(stage)): sha(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        ),
    )
    if target.exists():
        raise FileExistsError(target)
    stage.rename(target)


def identity(args):
    import psutil  # Research runtime only; pure integrity helpers need no extra dependency.

    script_paths = [
        Path(__file__),
        args.config,
        ROOT / "src/wqsurrogatemodels/wqi.py",
        ROOT / "src/wqsurrogatemodels/settings.py",
    ]
    data_paths = [
        args.data_root / "data" / n for n in ["dataV1.csv", "dataV1_50000.csv"]
    ]
    manifest = args.data_root / "models/production_model_manifest.json"
    model_paths = [
        args.data_root / item["production_artifact"]
        for item in json.loads(manifest.read_text())["artifacts"]
    ]
    return dict(
        source_sha256={str(p): sha(p) for p in script_paths},
        data_sha256={str(p): sha(p) for p in data_paths},
        production_sha256={str(p): sha(p) for p in [manifest, *model_paths]},
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        memory_bytes=psutil.virtual_memory().total,
        base_commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    )


def aggregate(output):
    frames = []
    for path in sorted((output / "jobs").glob("*/metrics.csv")):
        if (path.parent / "completed.json").is_file():
            frames.append(pd.read_csv(path))
    if frames:
        reports = output / "reports"
        reports.mkdir(exist_ok=True)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(reports / "metrics_by_seed.csv", index=False)
        df.groupby(["trees", "missing", "mode", "source"])[
            ["mae", "rmse", "r2", "accuracy", "macro_f1"]
        ].agg(["mean", "std"]).to_csv(reports / "metrics_summary.csv")


def run(args, config):
    if args.output.exists() and not (args.output / "identity.json").exists():
        unexpected = [
            p
            for p in args.output.iterdir()
            if p.name not in ("controller.lock", "logs")
        ]
        if unexpected:
            raise FileExistsError(
                "Refuse to initialize over existing results: " + str(args.output)
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = identity(args)
        frozen = args.output / "identity.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != current:
                raise ValueError(
                    "Source/environment/data/production identity changed; use a new result directory"
                )
        else:
            write_json(frozen, current)
        (args.output / "logs").mkdir(exist_ok=True)
        jobs = list(all_jobs(config))
        completed_keys = {
            job_id(**j) for j in jobs if verified(args.output / "jobs" / job_id(**j))
        }
        for job in jobs:
            key = job_id(**job)
            if key in completed_keys:
                continue
            attempts_path = args.output / "logs" / (key + "-attempts.json")
            attempts = (
                json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            )
            if len(attempts) >= 3:
                raise ValueError(
                    "Three attempts failed; manual diagnosis required: " + key
                )
            record = dict(utc=utc(), phase="running", **job)
            attempts.append(record)
            write_json(attempts_path, attempts)
            status = dict(
                phase="workstation_rf",
                active_job=key,
                completed_jobs=len(completed_keys),
                total_jobs=len(jobs),
                utc=utc(),
            )
            write_json(args.output / "status.json", status)
            print(json.dumps(status), flush=True)
            cmd = [
                sys.executable,
                __file__,
                "job",
                "--config",
                str(args.config),
                "--data-root",
                str(args.data_root),
                "--output",
                str(args.output),
                "--seed",
                str(job["seed"]),
                "--trees",
                str(job["trees"]),
                "--missing",
                job["missing"],
            ]
            with (args.output / "logs" / (key + f"-attempt{len(attempts)}.log")).open(
                "w"
            ) as log:
                result = run_bounded(
                    cmd,
                    timeout_seconds=config["execution"]["rf_job_timeout_seconds"],
                    log=log,
                    evidence=args.output
                    / "logs"
                    / (key + f"-attempt{len(attempts)}-execution.json"),
                )
            record.update(
                phase=result["phase"], returncode=result["returncode"], ended_utc=utc()
            )
            write_json(attempts_path, attempts)
            if result["returncode"]:
                raise RuntimeError(
                    "Job failed; retained log and partial output: " + key
                )
            if not verified(args.output / "jobs" / key):
                raise ValueError("Worker exited without verified completion: " + key)
            completed_keys.add(key)
            aggregate(args.output)
        if identity(args) != current:
            raise ValueError("Input identities changed during execution")
        aggregate(args.output)
        write_json(
            args.output / "status.json",
            dict(
                phase="workstation_rf_completed",
                completed_jobs=len(jobs),
                total_jobs=len(jobs),
                utc=utc(),
                pending=[
                    "ONNX whole-pipeline evaluation",
                    "Nano benchmark",
                    "final reports and backup",
                ],
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "job"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--trees", type=int)
    p.add_argument("--missing")
    args = p.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    config = yaml.safe_load(args.config.read_text())
    for protected in [args.data_root / "data", args.data_root / "models"]:
        if args.output == protected or protected in args.output.parents:
            raise ValueError("Output overlaps protected inputs")
    with threadpool_limits(limits=1):
        if args.action == "run":
            run(args, config)
        else:
            if (
                args.seed not in config["seeds"]
                or args.trees not in config["trees"]
                or args.missing not in ["complete", *config["missing_sets"]]
            ):
                raise ValueError("Job outside approved matrix")
            run_job(args, config)


if __name__ == "__main__":
    main()
