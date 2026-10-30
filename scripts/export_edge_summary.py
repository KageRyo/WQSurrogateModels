"""Export compact, provenance-linked summaries from a completed edge replay.

This command only reads replay evidence. It never runs inference or writes to
original experiment results; choose a new destination for every export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_summary(experiment: Path, output: Path) -> dict:
    experiment, output = experiment.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new summary directory; existing results are immutable")
    if output.is_relative_to(experiment):
        raise ValueError("Export outside the source experiment to preserve its evidence")
    report = experiment / "reports/replay-complete"
    source_names = [
        "reports/replay-complete/manifest.json",
        "reports/replay-complete/metrics_summary.csv",
        "reports/replay-complete/prediction_parity.csv",
        "reports/replay-complete/cpu_inference_timing.csv",
        "reports/replay-complete/component_resources.csv",
        "replay-results/runner-complete.json",
        "replay-staging/preparation-complete.json",
        "provenance/original_x86_results_sha256.json",
    ]
    sources = {name: sha256(experiment / name) for name in source_names}
    aggregate = json.loads((report / "manifest.json").read_text())
    runner = json.loads((experiment / "replay-results/runner-complete.json").read_text())
    preparation = json.loads((experiment / "replay-staging/preparation-complete.json").read_text())
    packet_paths = sorted((experiment / "replay-staging/packets").glob("*/job.json"))
    jobs = [json.loads(path.read_text()) for path in packet_paths]
    job_ids = {job["job_id"] for job in jobs}
    if (not jobs or len(job_ids) != len(jobs)
            or len(jobs) != preparation["jobs"] or len(jobs) != aggregate["verified_jobs"]
            or len(jobs) != runner["jobs"] or runner["phase"] != "all_jobs_complete"):
        raise ValueError("Prepared packets, runner and aggregate must describe one complete run")
    packet_hashes = [(path.relative_to(experiment).as_posix(), sha256(path)) for path in packet_paths]
    packet_digest = hashlib.sha256(json.dumps(packet_hashes, separators=(",", ":")).encode()).hexdigest()

    parity = pd.read_csv(report / "prediction_parity.csv")
    if set(parity.job_id) != job_ids or not parity["pass"].isin([True, False]).all():
        raise ValueError("Parity evidence must include every prepared job with boolean outcomes")
    parity["weighted_difference"] = parity.mean_abs_difference * parity.n_rows
    compact = parity.groupby(["job_id", "protocol", "model_type"], as_index=False).agg(
        n_frame_comparisons=("frame", "size"), n_row_comparisons=("n_rows", "sum"),
        max_abs_difference=("max_abs_difference", "max"),
        weighted_difference=("weighted_difference", "sum"),
        category_disagreement_events=("category_disagreements", "sum"),
        tolerance_min=("tolerance", "min"), tolerance_max=("tolerance", "max"),
        parity_pass=("pass", "all"),
    )
    if len(compact) != len(jobs):
        raise ValueError("A job has conflicting protocol/model provenance")
    compact["mean_abs_difference_over_frame_rows"] = compact.pop("weighted_difference") / compact.n_row_comparisons
    compact["seed_or_fold"] = compact.job_id.map({job["job_id"]: job["seed"] for job in jobs})
    failed = sorted(compact.loc[~compact.parity_pass, "job_id"].tolist())
    if (failed != sorted(aggregate["parity_difference_jobs"])
            or failed != sorted(runner["parity_difference_jobs"])
            or bool(not failed) != aggregate["all_parity_pass"]):
        raise ValueError("Recorded parity findings disagree with the source arrays' aggregate evidence")

    timing = pd.read_csv(report / "cpu_inference_timing.csv")
    keys = ["platform", "protocol", "dataset", "model_type", "experiment", "source"]
    timing_summary = timing.groupby(keys, as_index=False).agg(
        n_jobs=("job_id", "nunique"), n_seed_or_fold_ids=("seed", "nunique"),
        n_timing_observations=("repeat", "size"), n_repeat_ids=("repeat", "nunique"),
        n_rows_min=("n_rows", "min"), n_rows_max=("n_rows", "max"),
        stage_sum_seconds_mean=("stage_sum_seconds", "mean"),
        stage_sum_seconds_median=("stage_sum_seconds", "median"),
        stage_sum_seconds_min=("stage_sum_seconds", "min"),
        stage_sum_seconds_max=("stage_sum_seconds", "max"),
        stage_sum_per_row_ms_mean=("stage_sum_per_row_ms", "mean"),
        stage_sum_per_row_ms_median=("stage_sum_per_row_ms", "median"),
        stage_sum_per_row_ms_min=("stage_sum_per_row_ms", "min"),
        stage_sum_per_row_ms_max=("stage_sum_per_row_ms", "max"),
    )
    repeats = timing.groupby(keys + ["job_id"]).repeat.nunique().groupby(keys).agg(
        n_repeats_per_job_min="min", n_repeats_per_job_max="max").reset_index()
    timing_summary = timing_summary.merge(repeats, on=keys, validate="one_to_one")
    resource = pd.read_csv(report / "component_resources.csv")
    resource_summary = resource.groupby(["platform", "model_type"], as_index=False).agg(
        n_jobs=("job_id", "nunique"), n_component_observations=("component", "size"),
        n_component_types=("component", "nunique"),
        component_process_peak_rss_mib_max=("peak_process_rss_mib", "max"),
        component_load_seconds_min=("load_seconds", "min"),
        component_load_seconds_median=("load_seconds", "median"),
        component_load_seconds_max=("load_seconds", "max"),
    )
    for frame in [timing, resource]:
        if set(frame.job_id) != job_ids or set(frame.platform) != {"nano", "x86"}:
            raise ValueError("Timing/resources do not cover the same complete two-platform run")
    original = json.loads((experiment / "provenance/original_x86_results_sha256.json").read_text())
    repository = experiment.parent.parent
    if not original or any(sha256(repository / item["path"]) != item["sha256"] for item in original):
        raise ValueError("Original x86 results no longer match their recorded SHA-256 baseline")
    protocol_coverage = [{"protocol": protocol,
                          "jobs": sum(job["protocol"] == protocol for job in jobs),
                          "conditions": sum(len(job["conditions"]) for job in jobs if job["protocol"] == protocol)}
                         for protocol in sorted({job["protocol"] for job in jobs})]
    manifest = {
        "schema_version": 1, "source_experiment": experiment.name,
        "verified_jobs": len(jobs), "conditions": sum(len(job["conditions"]) for job in jobs),
        "protocol_coverage": protocol_coverage,
        "parity_pass_jobs": len(jobs) - len(failed), "parity_fail_jobs": len(failed),
        "max_prediction_difference": float(compact.max_abs_difference.max()),
        "category_disagreement_events": int(compact.category_disagreement_events.sum()),
        "original_x86_baseline_files_verified": len(original),
        "sources_sha256": sources, "packet_manifest_set_sha256": packet_digest,
        "packet_manifest_set_hash_contract": "SHA-256 of UTF-8 compact JSON list of [experiment-relative path, file SHA-256], sorted by path",
        "timing_contract": aggregate["timing_contract"], "memory_contract": aggregate["memory_contract"],
        "timing_summary_contract": "Unweighted summaries across observed seed/fold and repeat measurements within platform/protocol/dataset/model/experiment/source; not API latency. Row-count range and repetition counts are retained. Accuracy-only frame timings are absent from the timing source.",
        "resource_summary_contract": "Ranges pool all replay protocols/components for each platform/model; load_seconds is component deserialization time, not end-to-end cold-start latency. Process RSS is not total device/container memory or model size.",
        "seed_contract": "sample_size_experiments uses fold IDs 1-5; other protocols use seeds 0-4. Folds are not independent repeated experiments.",
        "parity_contract": "Absolute tolerance retained from source; category disagreements and weighted means count frame-row events, including repeated protocols and synthetic scenarios, not unique people/samples.",
        "accuracy_contract": "metrics_summary.csv is an exact byte copy of the complete aggregate. training_s is missing because no device training occurred. Deterministic WQI5 is a definition baseline; fitting dataset Score is not proof of greater correctness.",
        "scope_contract": "CPU inference replay of current-state assessment and controlled synthetic stress; not forecasting or real-event validation. Research runtime results are not deployment API benchmarks.",
    }
    # Check provenance again before creating the public export. JSON rejects NaN.
    encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if sources != {name: sha256(experiment / name) for name in source_names}:
        raise ValueError("Source report changed during export")
    output.mkdir(parents=True)
    shutil.copyfile(report / "metrics_summary.csv", output / "metrics_summary.csv")
    for name, frame in [("parity_per_job.csv", compact), ("cpu_timing_summary.csv", timing_summary),
                        ("resource_summary.csv", resource_summary)]:
        frame.to_csv(output / name, index=False)
    (output / "manifest.json").write_text(encoded)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_summary(args.experiment, args.output)
    print(json.dumps({key: result[key] for key in ["verified_jobs", "conditions", "parity_pass_jobs", "parity_fail_jobs"]}))


if __name__ == "__main__":
    main()
