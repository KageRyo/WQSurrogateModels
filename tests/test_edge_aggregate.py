"""Replay stress summaries must retain independent experimental protocols."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from edge_replay.aggregate import summarize_stress_by_protocol
from run_stress107_event_windows import detection_summary, severity_monotonicity


def test_stress_summaries_do_not_pool_identical_reference_conditions_across_protocols():
    rows = []
    severities = ["low_30pct", "medium_100pct", "high_300pct"]
    for protocol, deltas in [("missing_indicator_core", [-1., -2., -3.]),
                             ("missing_indicator_robustness", [-3., -2., -1.])]:
        for rank, (severity, delta) in enumerate(zip(severities, deltas)):
            rows.append({"protocol": protocol, "dataset": "fixture", "source": "external_10714",
                         "missing_set": "complete", "experiment_mode": "full_reference",
                         "experiment": "full_reference", "model_type": "lr", "seed": 42,
                         "stress_scenario": "fixture", "severity": severity, "severity_rank": rank,
                         "perturbation_pct": [30, 100, 300][rank], "window_id": 0, "n": 100,
                         "mean_delta_stress_minus_baseline": delta,
                         "median_delta_stress_minus_baseline": delta,
                         "mean_score_drop_baseline_minus_stress": -delta,
                         "median_score_drop_baseline_minus_stress": -delta,
                         "pct_score_decreased": 100., "pct_drop_ge_1": 100., "pct_drop_ge_5": 0.,
                         "pct_category_worse": 0., "detected_mean_decrease": True,
                         "detected_mean_drop_ge_1": True, "detected_any_category_worse": False})
    frame = pd.DataFrame(rows)
    detection = summarize_stress_by_protocol(frame, detection_summary)
    low = detection[detection.severity.eq(severities[0])].set_index("protocol")
    assert low.loc["missing_indicator_core", "mean_delta_score"] == -1.
    assert low.loc["missing_indicator_robustness", "mean_delta_score"] == -3.
    assert (low.n_seed_window_cases == 1).all()
    monotonicity = summarize_stress_by_protocol(frame, severity_monotonicity, severities).set_index("protocol")
    assert monotonicity.loc["missing_indicator_core", "severity_monotonicity_rate_delta"] == 1.
    assert monotonicity.loc["missing_indicator_robustness", "severity_monotonicity_rate_delta"] == 0.
