# Jetson Nano 2GB inference results

The complete CPU replay finished on 2026-10-30 at 02:56 Asia/Taipei;
aggregation finished at 03:03. All **610 jobs / 730 conditions** produced
complete, verified output. Of these jobs, 597 met the absolute prediction
tolerance of `1e-6` against the paired x86 reference; 13 completed with
numerical differences. This is a parity result, not a model accuracy rate.

See the [public summary manifest](../statistics/outputs/jetson_nano_20261026/manifest.json),
[accuracy statistics](../statistics/outputs/jetson_nano_20261026/metrics_summary.csv),
[job-level parity](../statistics/outputs/jetson_nano_20261026/parity_per_job.csv),
[research inference timings](../statistics/outputs/jetson_nano_20261026/cpu_timing_summary.csv)
and [component resources](../statistics/outputs/jetson_nano_20261026/resource_summary.csv).
These new ARM64 results coexist with the original workstation outputs.
The 58 original result files in the baseline SHA-256 inventory remain unchanged.
Private packets, datasets, serialized models and row-level predictions are
excluded from Git.

## Device and runtime

| Item | Measured configuration |
| --- | --- |
| Device | Jetson Nano 2GB, four Cortex-A57 cores, aarch64 |
| Memory | 1,971 MiB reported RAM, approximately 5 GiB swap |
| Host | Ubuntu 18.04.6, L4T 32.7.1, kernel 4.9.253-tegra |
| Container runtime | Docker 20.10.7; Bookworm userspace, glibc 2.36 |
| Research Python | 3.11.17 |
| Research packages | NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.5.2, joblib 1.5.3, XGBoost 3.2.0, LightGBM 4.6.0 |
| Inference | CPU, one thread per numerical runtime; no CUDA/TensorRT |
| Power configuration | Original MAXN mode 0; separate 5W mode 1 measurements, then restored to mode 0 |

Power-mode names are configuration labels. No power sensor or external meter
was available, so these results do not establish watts, energy efficiency or
a measured ten-watt MAXN draw. Training stayed off-device. See the
[deployment procedure](jetson-nano.md) and [runtime contract](edge-runtime.md).

The paired research reference ran on the current WSL2 x86 host: Intel Core
i7-12700KF, 20 reported virtual CPUs and approximately 31.23 GiB reported RAM,
Python 3.11.16, with the same research package versions and CPU
thread limits. This host is distinct from the archived workstation hardware
recorded with the historical x86 experiments; the archive was not overwritten.

## Scope and experimental boundaries

| Protocol | Jobs | Conditions |
| --- | ---: | ---: |
| Complete input | 35 | 35 |
| Reduced indicators | 155 | 155 |
| Original double-missing study | 90 | 120 |
| Expanded missing-indicator robustness | 210 | 300 |
| Sample-size sensitivity | 120 | 120 |
| Total | 610 | 730 |

LR, MPR, SVM, RF, XGBoost and LightGBM each contribute 100 jobs; the complete-input
formula baseline contributes 10. The original five seeds, sample-size folds,
rows, warmups and repeats are retained. Stress107 includes 107 consecutive
windows, five synthetic event types and three perturbation levels; its
481,500 window records are retained privately. It is not temporal forecasting,
107-fold cross-validation or validation against observed pollution incidents.

Archived estimators were reused where available. Complete-input and reduced-
indicator protocols without saved estimators use explicitly labeled new
off-device CPU references. They do not reproduce historical GPU training bit
for bit. The replay compares the same estimator and input packet on x86 and ARM.

Accuracy metrics compare predictions with the existing dataset `Score`.
The current direct formula itself has nonzero error against that label
(complete-input test mean MAE 6.070945); label-generation/formula consistency
still needs separate investigation. Lower surrogate error against `Score`
does not prove that a learned model is more correct than the WQI5 definition.

## Missing-indicator numerical differences

All 13 differing jobs are reconstruction cases: 12 LR and one MPR. Both
platforms' saved arrays are float64. Upstream reconstructed-indicator differences
are approximately `1e-15` to `1e-14`; the second-stage estimators have coefficients
of approximately `1e13` to `1e14` and singular-value ratios of approximately
`2.52e15` to `1.54e16`. This supports amplification by severely ill-conditioned
models. It does not uniquely identify a particular BLAS kernel or CPU instruction
as the cause, and the presence of an imputer alone is not evidence of a different
imputation policy.

The maximum final score difference is **0.5930534429741812** for LR and **0.078125**
for the MPR case. There are 561 category-disagreement frame-row events, including
68 in internal/external frames. Protocols and synthetic conditions reuse source
rows, so these counts are not independent or unique samples. Some differences
therefore do cross WQI5 category boundaries. The MPR stress outputs also include
absolute predictions above 1,773, illustrating an extrapolation limitation.
No tolerance was relaxed, prediction rounded, model retrained or negative result
removed to obtain a passing label.

## Production API measurements

Direct/RF/LightGBM measurements below were recorded on 2026-10-26; corrected
XGBoost 3.2.0 measurements were recorded on 2026-10-31. All use the approved
production artifacts, 300 one-row requests per model, after warmup. Raw timing
excludes DataFrame construction and API overhead; HTTP uses Nano loopback.
Cold model load includes checksum validation and deserialization. It excludes
dependency imports. Cold and warm measurements must be reported separately.

| Model | Raw P50/P95 ms | MAXN HTTP P50/P95 ms | 5W HTTP P50/P95 ms | Cold load s | Raw process peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct WQI5 | 0.0258 / 0.0277 | 6.41 / 7.42 | 10.68 / 16.06 | N/A | 28.35 |
| LightGBM | 6.63 / 7.03 | 15.67 / 17.57 | 24.42 / 34.24 | 10.51 | 174.84 |
| RF | 137.25 / 148.24 | 146.90 / 163.81 | 212.33 / 270.32 | 167.78 | 1,228.46 |
| XGBoost 3.2.0 | 6.72 / 7.14 | 15.52 / 17.42 | 22.49 / 24.17 | 10.29 | 191.52 |

The RF artifact is 1,053,564,815 bytes. Its 30-minute sequential HTTP soak
completed 11,897 requests in 1,800.099 seconds: 6.609 requests/s, P50 147.78 ms,
P95 179.69 ms and maximum 416.92 ms, with parity checks passing. This supports
that particular sustained workload, not arbitrary concurrency or long-term
reliability. Readiness/model listings cache successful file checksums until file
identity changes; actual model loading always checks the complete artifact again.

The historical XGBoost 2.1.4 deployment loaded the artifact but produced incorrect
scores. Those outputs are retained as diagnostics and excluded from the table.
The research replay already used 3.2.0. Corrected production-image verification
and benchmarks are saved separately from the 610-job replay. Raw XGBoost scores
match the x86 reference exactly on the five fixed probes; HTTP differs by at most
0.000471207, within the 0.000501 allowance for the API's three-decimal output.
The artifact SHA-256 remains unchanged. The API runs as the unprivileged mount
owner, all four deployment paths pass smoke requests, and MAXN mode 0 was
restored after the 5W measurements. The four smoke scores are direct 89.542,
LightGBM 89.587, RF 89.927 and XGBoost 89.958.

The [selected raw deployment records](../statistics/outputs/jetson_nano_20261026/deployment_benchmarks.json)
include sample counts, package versions, artifact checksums, cold/warm timing,
process CPU/RSS and the sustained RF test. Their inputs are fixed synthetic
probes, not private dataset rows. Research uses its archived dependency pins;
the corrected production measurements record their separate locked runtime
(NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.5.2 and XGBoost 3.2.0).

Research batch timings are sums of isolated component inference times, excluding
load, imports, IPC, transfer and HTTP. Their per-row distributions cannot replace
the one-row API measurements above. Component-process RSS is not whole-device
or whole-container memory; RF required swap on the 2GB device. Completion of the
native workload alone is not evidence of low resource use.

## Reproduction and follow-up

Use [the replay workflow](jetson-experiment-replay.md) for packet preparation,
verification, recovery, aggregation and private backup. Export public summaries
with `scripts/export_edge_summary.py` and an Excel view of the complete statistical
tables with `scripts/export_edge_workbook.py`. The workbook indexes the full
481,500-row stress CSV instead of silently truncating it.

ONNX (#58) and smaller RF ensembles (#59) have not been evaluated. They remain
optional follow-up studies, with explicit accuracy/resource comparisons needed
before model replacement. No quantization or pruning was required to finish this
replay. No production artifact was replaced, and ARM support preserves the v2 API.
Merge and release remain subject to user review.
