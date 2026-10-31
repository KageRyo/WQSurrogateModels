"""Prepare portable replay packets, keeping archived artifacts read-only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from edge_replay.schema import Component, Condition, InputFrame, ReplayJob
from edge_replay.worker import cpu_estimator, exclusive_json, sha256
from wqsurrogatemodels.wqi import categorize_score

FEATURES = ["DO", "BOD", "NH3N", "EC", "SS"]
MODEL_ORDER = ["lightgbm", "lr", "mpr", "svm", "xgboost", "rf"]
ROOT = Path(__file__).resolve().parents[2]


def split_indices(frame, seed, test_size):
    strata = frame.Score.map(lambda value: categorize_score(float(value))[0])
    return next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(frame[FEATURES], strata))


def save_inputs(directory, frames):
    """frames: (InputFrame, full frame, original row IDs) tuples."""
    path = directory / "inputs.npz"
    if not path.exists():
        arrays = {}
        for spec, frame, indices in frames:
            arrays[spec.name + "/x"] = frame[FEATURES].to_numpy(dtype=np.float64)
            arrays[spec.name + "/y"] = frame.Score.to_numpy(dtype=np.float64)
            arrays[spec.name + "/row_id"] = np.asarray(indices, dtype=np.int64)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    return sha256(path)


def shared_inputs(output, protocol, seed):
    from run_missing_indicator_experiments import extract_external_set, stress_frames
    from run_stress107_event_windows import Stress107ConfigSchema, perturbed_frame, scenario_multipliers

    name = "missing_indicator_config.yaml" if protocol == "missing_indicator_core" else "missing_indicator_robustness_config.yaml"
    config = yaml.safe_load((ROOT / "configs" / name).read_text())
    subset = pd.read_csv(ROOT / config["dataset_50000"])
    full = pd.read_csv(ROOT / config["full_dataset"])
    external = extract_external_set(full, subset, config["expected_external_rows"], None)
    train_idx, test_idx = split_indices(subset, seed, config["test_size"])
    # Assert exact archived row IDs instead of relying only on a seed match.
    archived_splits = ROOT / "results" / protocol / "splits" / "split_indices.csv"
    if archived_splits.exists():
        original = pd.read_csv(archived_splits)
        expected = original.loc[(original.seed == seed) & (original.split == "internal_test"), "row_id"].to_numpy()
        if not np.array_equal(expected, test_idx):
            raise ValueError("Archived internal-test split differs from reconstructed split")
    frames = [
        (InputFrame(name="internal_test", kind="internal_test", n_rows=len(test_idx)), subset.iloc[test_idx], test_idx),
        (InputFrame(name="external_10714", kind="external_10714", n_rows=len(external)), external,
         np.arange(len(subset), len(full))),
    ]
    if protocol == "missing_indicator_core":
        for scenario, frame in stress_frames(external, config["stress_test"]["scenarios"]).items():
            frame["Score"] = external.Score.to_numpy()
            frames.append((InputFrame(name="stress_" + scenario, kind="stress", n_rows=len(frame), scenario=scenario), frame, np.arange(len(subset), len(full))))
    else:
        stress = Stress107ConfigSchema.model_validate(config["stress107_event_windows"])
        for scenario, scenario_spec in stress.scenarios.items():
            for rank, severity in enumerate(stress.severities):
                multipliers = scenario_multipliers(scenario_spec, severity)
                frame = perturbed_frame(external, multipliers)
                frame["Score"] = external.Score.to_numpy()
                spec = InputFrame(name=f"stress107_{scenario}_{severity}", kind="stress107",
                                  n_rows=len(frame), scenario=scenario, severity=severity,
                                  severity_rank=rank, perturbation_pct=stress.severities[severity]["perturbation_pct"],
                                  multipliers=multipliers)
                frames.append((spec, frame, np.arange(len(subset), len(full))))
    target = output / "shared_inputs" / f"{protocol}-seed-{seed}"
    target.mkdir(parents=True, exist_ok=True)
    digest = save_inputs(target, frames)
    return target / "inputs.npz", digest, [item[0] for item in frames]


def write_packet(output, job_id, protocol, seed, model_type, source, conditions, inputs, input_sha, frame_specs,
                 component_models, features, source_origin="archived_workstation_artifact", source_sha=None):
    packet = output / "packets" / job_id
    if (packet / "job.json").exists():
        return
    packet.mkdir(parents=True, exist_ok=True)
    if not (packet / "inputs.npz").exists():
        os.link(inputs, packet / "inputs.npz")
    components = []
    for key, model in component_models.items():
        target = packet / (key + ".joblib")
        cpu_estimator(model)
        if not target.exists():
            # Lossless transport compression, never quantization. The archived
            # source is untouched. A child process loads each component alone.
            temporary = packet / (key + ".partial")
            joblib.dump(model, temporary, compress=3)
            temporary.rename(target)
        components.append(Component(key=key, path=target.name, sha256=sha256(target),
                                    features=features[key], target="Score" if key == "wqi" else key))
    job = ReplayJob(job_id=job_id, protocol=protocol, seed=seed, model_type=model_type,
                    source_artifact=str(source.relative_to(ROOT)), source_sha256=source_sha or sha256(source),
                    source_origin=source_origin, inputs_sha256=input_sha, frames=frame_specs,
                    components=components, conditions=conditions)
    exclusive_json(packet / "job.json", job.model_dump())
    print(json.dumps({"event": "packet_ready", "job_id": job_id}), flush=True)


def missing_packets(output, protocols, selected_models, selected_seeds):
    for model_type in selected_models:
        for protocol in protocols:
            if protocol not in {"missing_indicator_core", "missing_indicator_robustness"}:
                continue
            manifest = json.loads((ROOT / "results" / protocol / "manifest.json").read_text())
            for seed in selected_seeds:
                inputs, digest, frames = shared_inputs(output, protocol, seed)
                directory = ROOT / "results" / protocol / "models" / f"seed_{seed}"
                paths = sorted(directory.rglob(model_type + ".joblib"))
                for path in paths:
                    relative = path.relative_to(directory)
                    source_mode = relative.parts[0]
                    missing_set = relative.parts[1] if len(relative.parts) == 3 else "missing_bod_nh3n"
                    job_id = f"{protocol}-s{seed}-{model_type}-" + "-".join(relative.parts[:-1])
                    if (output / "packets" / job_id / "job.json").exists():
                        continue
                    bundle = joblib.load(path)
                    component_models = {"wqi": bundle["wqi_model"]}
                    if source_mode in {"full_model", "full_reference"}:
                        conditions = [Condition(name="full_reference", mode="full_reference")]
                        missing_sets = manifest.get("missing_sets", {"missing_bod_nh3n": ["BOD", "NH3N"]})
                        for name, indicators in missing_sets.items():
                            mode = "full_inference_dropout" if protocol == "missing_indicator_core" else "inference_dropout"
                            conditions.append(Condition(name=mode if protocol == "missing_indicator_core" else f"{mode}_{name}", mode=mode, missing_set=name, missing_indicators=indicators))
                        features = {"wqi": FEATURES}
                    elif source_mode == "reduced_retraining":
                        missing = bundle.get("missing_indicators", ["BOD", "NH3N"])
                        available = bundle.get("available_features", ["DO", "EC", "SS"])
                        conditions = [Condition(name=source_mode if protocol == "missing_indicator_core" else f"{source_mode}_{missing_set}", mode=source_mode, missing_set=missing_set, missing_indicators=missing)]
                        features = {"wqi": available}
                    else:
                        missing = bundle.get("missing_indicators", ["BOD", "NH3N"])
                        available = bundle.get("available_features", ["DO", "EC", "SS"])
                        estimators = bundle.get("indicator_models", {})
                        if source_mode == "two_stage_reconstruction":
                            estimators = {"BOD": bundle["bod_model"], "NH3N": bundle["nh3n_model"]}
                        reconstruction = {indicator: indicator.lower() for indicator in missing}
                        features = {"wqi": FEATURES}
                        for indicator, model in estimators.items():
                            key = reconstruction[indicator]
                            component_models[key] = model
                            features[key] = available
                        conditions = [Condition(name=source_mode if protocol == "missing_indicator_core" else f"{source_mode}_{missing_set}", mode=source_mode, missing_set=missing_set, missing_indicators=missing, reconstruction=reconstruction)]
                    write_packet(output, job_id, protocol, seed, model_type, path, conditions, inputs, digest, frames, component_models, features)
                    del bundle, component_models
                    gc.collect()


def sample_packets(output, selected_models):
    family = {"lr": "LR", "mpr": "MPR", "svm": "SVM", "rf": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    manifest = json.loads((ROOT / "results/sample_size_experiments/manifest.json").read_text())
    for model_type in selected_models:
        for dataset in manifest["datasets"]:
            frame = pd.read_csv(ROOT / dataset)
            strata = frame.Score.map(lambda value: categorize_score(float(value))[0])
            splitter = StratifiedKFold(n_splits=manifest["n_splits"], shuffle=True, random_state=manifest["shuffle_seed"])
            for fold, (train_idx, test_idx) in enumerate(splitter.split(frame[FEATURES], strata), start=1):
                name = Path(dataset).stem
                job_id = f"sample-size-{name}-fold{fold}-{model_type}"
                if (output / "packets" / job_id / "job.json").exists():
                    continue
                path = ROOT / "models/sample_size_experiments" / family[model_type] / name / f"fold_{fold}" / f"{model_type}.pkl"
                shared = output / "shared_inputs" / f"sample-size-{name}-fold{fold}"
                shared.mkdir(parents=True, exist_ok=True)
                specs = [InputFrame(name="train", kind="train", n_rows=len(train_idx)), InputFrame(name="test", kind="test", n_rows=len(test_idx))]
                digest = save_inputs(shared, [(specs[0], frame.iloc[train_idx], train_idx), (specs[1], frame.iloc[test_idx], test_idx)])
                model = joblib.load(path)
                write_packet(output, job_id, "sample_size_experiments", fold, model_type, path,
                             [Condition(name="full_reference", mode="full_reference")],
                             shared / "inputs.npz", digest, specs, {"wqi": model}, {"wqi": FEATURES})
                del model
                gc.collect()


def training_packets(output, protocols, selected_models, selected_seeds):
    from reproduce_results import build_model
    from edge_replay.formula import DirectWQI5

    for protocol, filename in [("complete_input", "experiment_config.yaml"), ("reduced_indicator", "reduced_indicator_config.yaml")]:
        if protocol not in protocols:
            continue
        config = yaml.safe_load((ROOT / "configs" / filename).read_text())
        data = pd.read_csv(ROOT / config["dataset"])
        scenarios = config.get("scenarios", {"full": {"features": FEATURES}})
        for model_type in ["direct_wqi5", *selected_models]:
            for seed in selected_seeds:
                train_idx, test_idx = split_indices(data, seed, config["test_size"])
                shared = output / "shared_inputs" / f"{protocol}-seed{seed}"
                shared.mkdir(parents=True, exist_ok=True)
                specs = [InputFrame(name="test", kind="test", n_rows=len(test_idx))]
                digest = save_inputs(shared, [(specs[0], data.iloc[test_idx], test_idx)])
                for scenario, spec in scenarios.items():
                    if model_type == "direct_wqi5" and spec["features"] != FEATURES:
                        continue
                    job_id = f"{protocol}-s{seed}-{model_type}-{scenario}"
                    if (output / "packets" / job_id / "job.json").exists():
                        continue
                    features = spec["features"]
                    source = output / "new_x86_training" / job_id / "model.joblib"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if source.exists():
                        model = joblib.load(source)
                    else:
                        print(json.dumps({"event": "off_device_training", "job_id": job_id}), flush=True)
                        model = DirectWQI5() if model_type == "direct_wqi5" else build_model(model_type, compute_device="cpu")
                        before = time.perf_counter()
                        model.fit(data.iloc[train_idx][features], data.iloc[train_idx].Score.to_numpy())
                        with (source.parent / "training.json").open("x") as stream:
                            json.dump({"seconds": time.perf_counter()-before, "train_rows": len(train_idx), "test_rows": len(test_idx), "seed": seed, "features": features, "config": filename, "note": "New CPU x86 reference; archived training artifacts were not saved. Archived results are unchanged."}, stream, indent=2)
                        joblib.dump(model, source, compress=3)
                    write_packet(output, job_id, protocol, seed, model_type, source,
                                 [Condition(name=scenario, mode="full_reference" if features == FEATURES else "reduced_retraining", missing_set=scenario)],
                                 shared / "inputs.npz", digest, specs, {"wqi": model}, {"wqi": features},
                                 source_origin="new_off_device_cpu_training")
                    del model
                    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocols", nargs="+", default=["missing_indicator_robustness", "missing_indicator_core", "sample_size_experiments", "complete_input", "reduced_indicator"])
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / "results" not in output.parents:
        raise ValueError("Use a dedicated new directory under results for replay preparation")
    # Explicit marker prevents treating an arbitrary old results directory as
    # resumable staging. A job.json is written only after its packet is complete.
    marker = output / "edge-replay-staging.json"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise FileExistsError("Refusing to use a nonempty directory without a replay marker")
    output.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        exclusive_json(marker, {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    missing_packets(output, args.protocols, args.models, args.seeds)
    if "sample_size_experiments" in args.protocols:
        sample_packets(output, args.models)
    training_packets(output, args.protocols, args.models, args.seeds)
    expected = 0
    for protocol in args.protocols:
        if protocol == "missing_indicator_core":
            expected += 3 * len(args.models) * len(args.seeds)
        elif protocol == "missing_indicator_robustness":
            expected += 7 * len(args.models) * len(args.seeds)
        elif protocol == "sample_size_experiments":
            manifest = json.loads((ROOT / "results/sample_size_experiments/manifest.json").read_text())
            expected += len(manifest["datasets"]) * manifest["n_splits"] * len(args.models)
        elif protocol in {"complete_input", "reduced_indicator"}:
            filename = "experiment_config.yaml" if protocol == "complete_input" else "reduced_indicator_config.yaml"
            config = yaml.safe_load((ROOT / "configs" / filename).read_text())
            scenarios = config.get("scenarios", {"full": {"features": FEATURES}})
            expected += len(args.seeds) * (len(scenarios) * len(args.models) + sum(spec["features"] == FEATURES for spec in scenarios.values()))
        else:
            raise ValueError("Unknown protocol: " + protocol)
    count = len(list((output / "packets").glob("*/job.json")))
    if count != expected:
        raise ValueError(f"Incomplete preparation: expected {expected} packets, found {count}")
    completion = {"jobs": count, "protocols": args.protocols, "models": args.models, "seeds": args.seeds}
    marker = output / "preparation-complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != completion:
            raise ValueError("Existing preparation completion has a different experiment scope")
    else:
        exclusive_json(marker, completion)
