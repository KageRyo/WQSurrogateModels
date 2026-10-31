"""Aggregate verified replay results; incomplete runs never become final reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from edge_replay.runner import verify_complete
from edge_replay.schema import ReplayJob
from edge_replay.worker import exclusive_json


def summarize_stress_by_protocol(frame, function, *args):
    """Original stress helpers know one protocol; never pool replay protocols."""
    summaries = []
    for (protocol, dataset), group in frame.groupby(["protocol", "dataset"], sort=True):
        summary = function(group, *args).copy()
        summary["protocol"] = protocol
        summary["dataset"] = dataset
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def main():
    from run_missing_indicator_robustness_experiments import (
        aggregate_metric_rows, best_by_experiment_source, bootstrap_ci_from_metric_rows,
        paired_tests_from_metric_rows, score_to_category,
    )
    from run_stress107_event_windows import detection_summary, severity_monotonicity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-parity-failures", action="store_true", help="Include complete measurements with failed parity, explicitly reporting those findings")
    args = parser.parse_args()
    preparation = json.loads((args.staging / "preparation-complete.json").read_text())
    packets = sorted((args.staging / "packets").glob("*/job.json"))
    if len(packets) != preparation["jobs"]:
        raise ValueError("Prepared job count changed")
    if args.output.exists():
        raise FileExistsError("Choose a new aggregation directory; existing results are immutable")
    # Verify the complete run before creating anything that looks like a report.
    for manifest in packets:
        for platform in ["nano", "x86"]:
            verify_complete(manifest.parent, args.results / platform / manifest.parent.name, parity=platform == "nano", record_parity_failures=args.record_parity_failures)
    args.output.mkdir(parents=True)
    metric_frames = []
    parity_frames = []
    timing_rows = []
    resources = []
    stress_frames = []
    global_frames = []
    category_rows = []
    for manifest in packets:
        packet = manifest.parent
        job = ReplayJob.model_validate_json(manifest.read_text())
        dataset = Path(job.source_artifact).parts[-3] if job.protocol == "sample_size_experiments" else "dataV1"
        if job.protocol.startswith("missing_indicator"):
            dataset = "dataV1_50000_plus_external_10714"
        for platform in ["nano", "x86"]:
            result = args.results / platform / job.job_id
            metrics = pd.read_csv(result / "metrics.csv")
            metrics["platform"] = platform
            metrics["dataset"] = dataset
            metrics["job_id"] = job.job_id
            metrics["training_s"] = np.nan  # Inference replay performs no device training.
            times = pd.concat([pd.read_csv(result / (component.key + "-timing.csv")) for component in job.components], ignore_index=True)
            for row in metrics.to_dict("records"):
                matching = times[(times.condition == row["experiment"]) & (times.frame == row["source"])]
                measured = matching[matching.phase == "timing"]
                if measured.empty:
                    latency = matching[matching.phase == "accuracy"].seconds.sum()
                else:
                    latency = measured.groupby("repeat").seconds.sum().mean()
                metrics.loc[metrics.experiment.eq(row["experiment"]) & metrics.source.eq(row["source"]), "latency_s"] = latency
            metric_frames.append(metrics)
            measured = times[times.phase == "timing"]
            for (condition, frame, repeat), values in measured.groupby(["condition", "frame", "repeat"]):
                timing_rows.append({"platform":platform, "protocol":job.protocol, "dataset":dataset,
                                    "job_id":job.job_id, "seed":job.seed, "model_type":job.model_type,
                                    "experiment":condition, "source":frame, "repeat":repeat,
                                    "n_rows":int(values.n_rows.iloc[0]),
                                    "stage_sum_seconds":values.seconds.sum(),
                                    "stage_sum_per_row_ms":1000*values.seconds.sum()/values.n_rows.iloc[0]})
            for component in job.components:
                values = json.loads((result / (component.key + "-resources.json")).read_text())
                resources.append({"platform":platform, "protocol":job.protocol, "dataset":dataset,
                                  "job_id":job.job_id, "seed":job.seed, "model_type":job.model_type,
                                  "component":component.key, "load_seconds":values["load_seconds"],
                                  "peak_process_rss_mib":values["peak_rss_mib"],
                                  "inference_wall_seconds":values["inference_wall_seconds"],
                                  "inference_cpu_seconds":values["inference_cpu_seconds"]})
            if platform == "nano":
                comparison = []
                with np.load(result / "wqi.npz", allow_pickle=False) as actual, np.load(args.results / "x86" / job.job_id / "wqi.npz", allow_pickle=False) as reference:
                    for condition in job.conditions:
                        for spec in job.frames:
                            key = condition.name + "/" + spec.name
                            delta = np.abs(actual[key] - reference[key])
                            comparison.append({"job_id":job.job_id, "protocol":job.protocol, "model_type":job.model_type,
                                               "experiment":condition.name, "frame":spec.name, "n_rows":len(delta),
                                               "max_abs_difference":float(delta.max()), "mean_abs_difference":float(delta.mean()),
                                               "category_disagreements":int(np.count_nonzero(np.asarray(score_to_category(actual[key])) != np.asarray(score_to_category(reference[key])))),
                                               "tolerance":1e-6, "pass":bool(np.all(delta <= 1e-6))})
                parity_frames.append(pd.DataFrame(comparison))
                for filename, collection in [("stress107-windows.csv",stress_frames),("global-stress.csv",global_frames)]:
                    if (result / filename).exists():
                        frame = pd.read_csv(result / filename)
                        frame["protocol"] = job.protocol
                        frame["dataset"] = dataset
                        frame["job_id"] = job.job_id
                        collection.append(frame)
                with np.load(result / "wqi.npz", allow_pickle=False) as predictions, np.load(packet / "inputs.npz", allow_pickle=False) as inputs:
                    for condition in job.conditions:
                        for spec in job.frames:
                            if spec.kind not in {"internal_test", "external_10714", "test", "train"}:
                                continue
                            y = inputs[spec.name + "/y"]
                            p = predictions[condition.name + "/" + spec.name]
                            actual = np.asarray(score_to_category(y))
                            predicted = np.asarray(score_to_category(p))
                            for category in np.unique(actual):
                                mask = actual == category
                                error = y[mask]-p[mask]
                                category_rows.append({"protocol":job.protocol, "dataset":dataset,
                                                      "job_id":job.job_id, "missing_set":condition.missing_set,
                                                      "experiment_mode":condition.mode,
                                                      "source":spec.kind, "experiment":condition.name,
                                                      "model_type":job.model_type, "seed":job.seed,
                                                      "actual_category":category, "n":int(mask.sum()),
                                                      "mae":float(np.mean(np.abs(error))),
                                                      "rmse":float(np.sqrt(np.mean(error**2))),
                                                      "accuracy_within_band":float(np.mean(actual[mask]==predicted[mask]))})
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(args.output / "metrics_by_seed.csv", index=False)
    summaries, intervals, paired, best = [], [], [], []
    for (platform, protocol, dataset), group in metrics.groupby(["platform", "protocol", "dataset"]):
        rows = group.to_dict("records")
        for function, destination in [(aggregate_metric_rows,summaries),(bootstrap_ci_from_metric_rows,intervals),
                                      (paired_tests_from_metric_rows,paired),(best_by_experiment_source,best)]:
            result = function(rows, 200) if function == bootstrap_ci_from_metric_rows else function(rows)
            destination.extend({"platform":platform,"protocol":protocol,"dataset":dataset,**row} for row in result)
    for filename, rows in [("metrics_summary.csv",summaries),("bootstrap_ci.csv",intervals),
                           ("paired_error_tests.csv",paired),("best_by_experiment.csv",best),
                           ("cpu_inference_timing.csv",timing_rows),("component_resources.csv",resources),
                           ("error_by_wqi_band_by_seed.csv",category_rows)]:
        pd.DataFrame(rows).to_csv(args.output / filename, index=False)
    parity = pd.concat(parity_frames, ignore_index=True)
    parity.to_csv(args.output / "prediction_parity.csv", index=False)
    if stress_frames:
        frame = pd.concat(stress_frames, ignore_index=True)
        frame.to_csv(args.output / "stress107_windows.csv", index=False)
        summarize_stress_by_protocol(frame, detection_summary).to_csv(args.output / "stress107_detection_summary.csv", index=False)
        summarize_stress_by_protocol(frame, severity_monotonicity, ["low_30pct", "medium_100pct", "high_300pct"]).to_csv(args.output / "stress107_severity_monotonicity.csv", index=False)
    if global_frames:
        pd.concat(global_frames, ignore_index=True).to_csv(args.output / "global_stress.csv", index=False)
    exclusive_json(args.output / "manifest.json", {
        "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "verified_jobs":len(packets), "platforms":["nano","x86"],
        "all_parity_pass":bool(parity["pass"].all()),
        "parity_findings_recorded":args.record_parity_failures,
        "parity_difference_jobs":sorted(parity.loc[~parity["pass"], "job_id"].unique().tolist()),
        "max_prediction_difference":float(parity.max_abs_difference.max()),
        "timing_contract":"Sum of isolated component inference times, excluding imports, loading, transfers and process orchestration. Not API latency. Cold/orchestration time is retained per job.",
        "memory_contract":"Per-component process peak RSS; not total device/container memory. Driver RSS is recorded separately per job.",
        "training_contract":"Archived workstation artifacts reused where present; complete/reduced protocols without saved artifacts have new off-device CPU references. Original archived results are unchanged.",
    })
