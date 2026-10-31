from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from export_edge_summary import export_summary


def source_run(tmp_path):
    experiment = tmp_path / "results/replay"
    report = experiment / "reports/replay-complete"
    report.mkdir(parents=True)
    def write_json(relative, value):
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write_json("replay-staging/preparation-complete.json", {"jobs": 2})
    write_json("replay-results/runner-complete.json", {
        "phase": "all_jobs_complete", "jobs": 2, "parity_difference_jobs": ["job2"]})
    write_json("reports/replay-complete/manifest.json", {
        "verified_jobs": 2, "parity_difference_jobs": ["job2"], "all_parity_pass": False,
        "timing_contract": "Component inference sums; not API latency", "memory_contract": "Process peak RSS"})
    for seed in [1, 2]:
        write_json(f"replay-staging/packets/job{seed}/job.json", {
            "job_id": f"job{seed}", "protocol": "sample_size_experiments", "seed": seed,
            "conditions": [{}] * seed})
    baseline = tmp_path / "original.csv"
    baseline.write_bytes(b"original scientific output\n")
    write_json("provenance/original_x86_results_sha256.json", [{
        "path": "original.csv", "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}])
    (report / "metrics_summary.csv").write_bytes(b"platform,training_s\nnano,\nx86,\n")
    parity, timing, resource = [], [], []
    for seed in [1, 2]:
        for frame in ["test", "stress"]:
            parity.append({"job_id": f"job{seed}", "protocol": "sample_size_experiments",
                           "model_type": "lr", "frame": frame, "n_rows": seed * 10,
                           "max_abs_difference": .1 if seed == 2 else 0.,
                           "mean_abs_difference": .02 if seed == 2 else 0.,
                           "category_disagreements": 1 if seed == 2 else 0,
                           "tolerance": 1e-6, "pass": seed == 1})
        for platform in ["nano", "x86"]:
            for repeat in range(seed + 1):
                timing.append({"platform": platform, "protocol": "sample_size_experiments", "dataset": "fixture",
                               "model_type": "lr", "experiment": "full", "source": "test",
                               "job_id": f"job{seed}", "seed": seed, "repeat": repeat,
                               "n_rows": seed * 10, "stage_sum_seconds": float(seed),
                               "stage_sum_per_row_ms": 100.})
            resource.append({"platform": platform, "model_type": "lr", "job_id": f"job{seed}",
                             "component": "wqi", "peak_process_rss_mib": seed * 100.,
                             "load_seconds": float(seed)})
    for name, rows in [("prediction_parity.csv", parity), ("cpu_inference_timing.csv", timing),
                       ("component_resources.csv", resource)]:
        pd.DataFrame(rows).to_csv(report / name, index=False)
    return experiment


def test_public_export_retains_exact_metrics_and_timing_sampling_semantics(tmp_path):
    experiment = source_run(tmp_path)
    output = tmp_path / "public"
    manifest = export_summary(experiment, output)
    assert manifest["verified_jobs"] == 2 and manifest["conditions"] == 3
    assert manifest["parity_pass_jobs"] == manifest["parity_fail_jobs"] == 1
    assert manifest["original_x86_baseline_files_verified"] == 1
    assert (output / "metrics_summary.csv").read_bytes() == (experiment / "reports/replay-complete/metrics_summary.csv").read_bytes()
    for name, digest in manifest["sources_sha256"].items():
        assert hashlib.sha256((experiment / name).read_bytes()).hexdigest() == digest
    timing = pd.read_csv(output / "cpu_timing_summary.csv").set_index("platform").loc["nano"]
    assert timing.n_seed_or_fold_ids == 2 and timing.n_timing_observations == 5
    assert timing.n_repeats_per_job_min == 2 and timing.n_repeats_per_job_max == 3
    assert timing.n_rows_min == 10 and timing.n_rows_max == 20
    assert timing.stage_sum_seconds_mean == pytest.approx(1.6)
    parity = pd.read_csv(output / "parity_per_job.csv").set_index("job_id").loc["job2"]
    assert parity.category_disagreement_events == 2 and parity.n_row_comparisons == 40
    assert parity.mean_abs_difference_over_frame_rows == pytest.approx(.02)
    resource = pd.read_csv(output / "resource_summary.csv").iloc[0]
    assert resource.component_process_peak_rss_mib_max == 200.
    assert resource.component_load_seconds_min == 1. and resource.component_load_seconds_max == 2.
    # JSON must be strict even though the copied CSV legitimately lacks training time.
    json.loads((output / "manifest.json").read_text(), parse_constant=lambda value: pytest.fail(value))
    with pytest.raises(FileExistsError):
        export_summary(experiment, output)


@pytest.mark.parametrize("changed", ["runner", "baseline"])
def test_export_refuses_incomplete_run_or_changed_original_results(tmp_path, changed):
    experiment = source_run(tmp_path)
    if changed == "runner":
        path = experiment / "replay-results/runner-complete.json"
        value = json.loads(path.read_text())
        value["jobs"] = 1
        path.write_text(json.dumps(value))
    else:
        (tmp_path / "original.csv").write_text("changed")
    output = tmp_path / "public"
    with pytest.raises(ValueError):
        export_summary(experiment, output)
    assert not output.exists()
