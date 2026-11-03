# ONNX missing-input and model-family extension

Status: the 14-path x86/Nano measurement matrix and local backup completed on
2026-11-03. SVM numerical follow-up is also complete; final review remains pending. This is a separately dated
extension of the archived [RF/ONNX evaluation](edge-optimization-results.md),
not a production replacement or a rerun of the original 610 jobs.

## Measured scope and protocol

- Reuse fitted RF50, seed 0 artifacts from `edge_optimization_20261101`.
  Evaluate BOD, NH3N and double-missing conditions, each with inference dropout,
  reduced retraining and two-stage indicator reconstruction: nine graph paths.
- Compose the fitted imputers, indicator models and final WQI model into one
  ONNX graph. Preserve feature order; unavailable columns are selected/masked
  inside the graph. Poisoned-column probes test that unavailable values cannot
  enter the result. Graph metadata binds mode, feature order and source hash.
- Independently evaluate the manifest-selected complete-input LR, MPR, SVM,
  LightGBM and XGBoost pipelines. No fitting or artifact replacement occurs.
- Store native float64, native float32 and ONNX float32 predictions separately.
  Use unchanged absolute tolerance `1e-6` and report category changes. RF uses
  internal 10,000 and external 10,714 rows; other families use external rows.
- Nano jobs run sequentially, with one CPU, 1,100 MiB RAM, 3,500 MiB RAM+swap,
  network disabled and read-only inputs. Each has a 1,200-second deadline;
  conversion deadlines are 900 seconds for RF bundles and 300 for other models.
  Compare Nano with both x86 ONNX and native outputs; retain OOM/timeouts.
- Record 300 consecutive warm samples for batch 1/128, process-cold loading and
  process RSS. Checksum verification warms the file cache; the API shares the
  Nano. These are not OS-cold or isolated-device measurements.
- Pair network and Nano-loopback health probes during inference. An SSH
  transport failure leaves the loopback HTTP outcome unknown.

The historical RF50 full-input graph used ONNX 1.17.0. This extension uses
ONNX 1.23.2, ONNX Runtime 1.31.1 and skl2onnx 1.20.0; non-RF converters use
onnxmltools 1.16.0. The runtime change is explicit and historical measurements
are not overwritten or treated as a controlled converter-version comparison.

## Completed x86 findings

All nine RF missing-input graphs and five other complete-input families
converted and executed successfully. RF native reruns match the archived
predictions exactly. Conversion/execution success is distinct from numerical
parity; all paths have rows above `1e-6`.

External RF two-stage reconstruction:

| Missing | Maximum score difference | Rows above 1e-6 | Category changes |
|---|---:|---:|---:|
| BOD | 0.783605 | 9,635 / 10,714 | 1 |
| NH3N | 1.938010 | 9,491 / 10,714 | 1 |
| BOD + NH3N | 0.836398 | 9,607 / 10,714 | 0 |

RF dropout/reduced-retraining maximum differences are between `2.5e-5` and
`4.4e-5`, with zero category changes in the external set.

| Complete-input family | Maximum score difference | Rows above 1e-6 | Category changes |
|---|---:|---:|---:|
| LR | 0.00001056 | 7,372 | 0 |
| MPR | 0.00005995 | 9,890 | 0 |
| SVM | 0.00138654 | 10,644 | 0 |
| LightGBM | 0.00010320 | 10,398 | 0 |
| XGBoost | 0.00009918 | 9,964 | 0 |

## Reconstruction stage isolation

Post-hoc counterfactuals feed ONNX-reconstructed indicators into the native
second-stage WQI model. The large score changes are reproduced. With the same
ONNX reconstruction supplied to both final-stage implementations, maximum
remaining differences are below `4.6e-5` for all three settings.

For every external row with overall difference above `0.001`, saved tree paths
identify a changed native second-stage branch when reconstructed values are
substituted: 121 BOD rows, 237 NH3N rows and 429 double-missing rows. Saved
records include row ID, tree/node, feature, threshold and the two float32 inputs.
This localizes the large differences to sensitivity of the second-stage trees
to changed reconstruction values. It does not establish the cause of the
separate historical native x86/Nano 54-row finding.

## Nano results and decision

All **14/14** Nano paths completed: nine RF missing-input graphs and five
complete-input model families. No inference OOM or timeout occurred. Thirteen
paths match x86 ONNX predictions exactly across all 10,714 external rows each.
SVM differs across platforms: 6,996/10,714 rows exceed `1e-6`, maximum difference
`0.00020980835`, and one WQI5 category changes.

The changed category is a boundary case: native and x86 ONNX scores lie above
the 50-point threshold, while Nano ONNX lies below it. Exact per-row scores remain
in the private follow-up. Nano SVM differs from native on 10,657 rows above
`1e-6`, maximum `0.00134839176`; the x86 comparison above has different counts. A separate repeatability/batch/optimizer follow-up is
saved under `results/onnx_svm_validation_20261103`; see below.

| Path | Load s | Batch-1 P50 / P95 ms | Process peak MiB |
|---|---:|---:|---:|
| RF50 BOD reconstruction | 34.669 | 0.184 / 0.357 | 1103.6 |
| RF50 BOD dropout | 8.554 | 0.108 / 0.212 | 1123.9 |
| RF50 BOD reduced | 8.659 | 0.099 / 0.207 | 1117.3 |
| RF50 BOD+NH3N reconstruction | 161.457 | 0.213 / 0.439 | 1121.3 |
| RF50 BOD+NH3N dropout | 8.818 | 0.112 / 0.224 | 1126.3 |
| RF50 BOD+NH3N reduced | 9.271 | 0.089 / 0.181 | 1128.9 |
| RF50 NH3N reconstruction | 21.504 | 0.141 / 0.263 | 1126.6 |
| RF50 NH3N dropout | 8.984 | 0.122 / 0.287 | 1119.6 |
| RF50 NH3N reduced | 9.260 | 0.091 / 0.204 | 1126.8 |
| lightgbm | 0.240 | 0.124 / 0.233 | 103.0 |
| lr | 0.199 | 0.090 / 0.209 | 92.4 |
| mpr | 0.156 | 0.197 / 0.293 | 91.8 |
| svm | 0.159 | 2.250 / 2.392 | 118.2 |
| xgboost | 0.239 | 0.135 / 0.242 | 110.4 |

These are raw ONNX timings, not HTTP/app latency. Process RSS high-water and
charged cgroup memory are distinct measurements. RF graph load-memory costs
remain large despite sub-millisecond warm single-row timings. The API retained
its image, restart count and no-OOM state around every job.

Paired inference health probes total 108. Network HTTP succeeds in 107, with one
SVM `URLError`; loopback succeeds in 107, with one SSH `TimeoutExpired` leaving
that local HTTP result unknown. These observations do not establish uninterrupted
service, and do not explain the historical 14-case health finding.

Decision: keep the native production models. The extension establishes working
conversion and ARM64 execution for this measured matrix; it does not meet the
unchanged native `1e-6` equivalence requirement. In particular, composed
reconstruction and SVM have observed category changes. No quantization, model
promotion or release is included.

## SVM repeatability and batch follow-up

The same saved graph, inputs and ORT 1.31.1 were exercised on each platform with
four variants: two separate batch-128 runs, batch 1, and batch 128 with graph
optimization disabled. This is a post-hoc numerical diagnostic, not another
latency benchmark. It uses a separate result directory and 300-second deadline.

| Variant | x86/Nano rows above 1e-6 | Maximum cross-platform difference | Category changes |
|---|---:|---:|---:|
| Batch 128, first run | 6,996 | 0.00020980835 | 1 |
| Batch 128, repeat | 6,996 | 0.00020980835 | 1 |
| Batch 1 | 10,652 | 0.00302886963 | 1 |
| Batch 128, graph optimization disabled | 6,996 | 0.00020980835 | 1 |

Within each platform the two batch-128 runs are bit-identical, and disabling
graph optimization also leaves those outputs unchanged. Nano batch 1 matches
its batch-128 output exactly; x86 batch 1 differs from its batch-128 output on
10,670 rows above `1e-6`, maximum `0.00302886963`. The one Nano category change
persists across all four variants. These tests identify reproducible platform
and batch-size sensitivity and do not isolate a particular runtime kernel or
hardware instruction as the cause. Changing graph optimization alone does not
resolve this case; a blanket claim of ONNX batch/platform equivalence is invalid.

The follow-up includes three CSVs and a three-sheet Excel workbook. A separate
`20261103-onnx-svm-validation` E-drive backup contains 23 files / 384,965 bytes,
verified source/destination SHA-256. Manifest SHA-256:
`e09423325101577a50f33020b8c7392aeb3e61d24fdc64550e09f943468b16dc`.
The original 294-file extension snapshot was not rewritten.

## Reports and preservation

The private `results/onnx_missing_20261103` root is frozen. It retains executed
source snapshots, runtime/host provenance, source hashes, row predictions,
progress, bounded worker logs, container state, health probes and stage-isolation
results. A first 30-second SCP transfer timeout before any worker is retained;
a separate staging attempt used 600-second transfers.

`reports/final/` contains six CSV tables, Markdown and a seven-sheet Excel
workbook: coverage 28 rows, x86 parity 69, x86 metrics 23, Nano timing 28,
health 14 and stage isolation 3. The 28 coverage entries are paired x86/Nano
records for 14 model paths, not 28 independent conditions. The workbook was
reopened and checked for error cells; table values were derived from saved
predictions. Public Git contains summaries and code, not graphs/private rows.

Separate local E-drive backup: `20261103-onnx-extension`, completed at 15:45 CST,
294 files and 1,304,368,452 bytes. All source and destination file hashes were
independently rechecked during final acceptance. Manifest SHA-256:
`e226e63dd03605226d719c480a7e8eebe386ef3420795249e6a81baac29a5312`.
Remote Nextcloud readback and restoration remain unverified.

This screen uses one RF seed/size and a reused external dataset. It does not
cover larger missing-input RF graphs or missing-input bundles for every other
model family. The original 610 jobs, 20261101 run and production model/manifest
hashes are unchanged. Identical x86/Nano ONNX output in 13 paths does not establish
native numerical equivalence.

## Final export acceptance safeguards

The maintained exporter requires all 14 x86 controllers/results and all 14 Nano
completion records/containers to report successful execution. It checks the
recorded hashes of consumed evidence, external prediction shapes/finiteness,
and exact unique keys for all six tables: Coverage 28, X86Parity 69,
X86Metrics 23, Nano 28, Health 14 and StageIsolation 3. Missing, unsuccessful,
duplicate or incomplete evidence is rejected before creating an output directory.
Completed numerical-parity and health findings remain reportable observations;
the export gate does not turn them into parity or continuity passes.

A new destination inside the source result root must be under `reports/`;
source-evidence directories and symlinked destinations are refused. Acceptance
re-export used a separate temporary directory: all six CSVs were byte-identical
to the archived tables. The frozen experiment and its two backups were retained.
The family evaluator also checks model SHA-256 explicitly before deserialization,
including when Python is started with `-O`.

## Reproduction

Install the hash-locked research dependencies plus
`requirements-edge-onnxmltools.txt` into an isolated environment. Supervise each
conversion with `scripts/edge_optimization_control.py`; preserve its log and
terminal JSON. For example:

```sh
python scripts/edge_optimization_control.py \
  --timeout-seconds 900 \
  --evidence /ABS/results/NEW/bod-control.json \
  --log /ABS/results/NEW/bod.log -- \
  python scripts/edge_onnx_missing_evaluate.py \
  --job /ABS/results/edge_optimization_20261101/jobs/s0-rf50-missing_bod \
  --complete-job /ABS/results/edge_optimization_20261101/jobs/s0-rf50-complete \
  --mode indicator_reconstruction --data-root /ABS/WQSurrogateModels \
  --output /ABS/results/NEW/bod
```

The extension worker is `edge_onnx_missing_benchmark.py`; the historical worker
continues to reject unsupported missing-input ONNX requests. Use distinct
outputs, retain unsuccessful attempts, and never overwrite an existing run.

Graph composition follows the [ONNX composition API](https://onnx.ai/onnx/api/compose.html).
External family registration follows the [skl2onnx converter integration guide](https://onnx.ai/sklearn-onnx/tutorial_1-5_external.html).
