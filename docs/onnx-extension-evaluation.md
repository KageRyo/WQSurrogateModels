# ONNX missing-input and model-family extension

Status: actual evaluation in progress, 2026-11-03. This is a separately dated
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

## Nano, reports and preservation

Nano measurements are in progress; terminal outcomes and measurements will be
added after retrieval and hash verification. The running API image and restart
count are checked around every job. The first two single-missing reconstructions
completed with identical x86/Nano ONNX predictions across all 10,714 rows.

New private root: `results/onnx_missing_20261103`. It retains executed-source
snapshots, runtime/host provenance, source hashes, row predictions, progress,
bounded worker logs, container state, health probes and stage-isolation results.
Final CSV/Excel exports and a separate E-drive SHA-256 backup follow terminal
measurement states. A first 30-second SCP transfer timeout is retained; it did
not start a Nano worker. A separate staging attempt uses 600-second transfers.

This screen uses one RF seed/size and a reused external dataset. It does not
cover larger missing-input RF graphs or missing-input bundles for every other
model family. Neither zero category changes nor identical x86/Nano ONNX output
establishes native `1e-6` equivalence. No quantization, model promotion or release
is included.

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
