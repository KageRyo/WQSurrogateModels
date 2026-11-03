# RF compression and ONNX evaluation — conclusions

Status: evaluation complete, with the deployment decisions below approved for
this stage on 2026-11-03. The original measurements, negative findings and local
backup are preserved. There is no production model replacement or app release.

A separately dated [ONNX missing-input and model-family extension](onnx-extension-evaluation.md)
has completed actual measurements; its new results are distinct from the archived stage below.

## Closure decisions

| Evaluation | Decision | Evidence and scope |
|---|---|---|
| ONNX (#58) | Keep the native production format. | Conversion and ARM64 measurements are complete for the tested full-input RF pipelines. Float32 parity, loading memory and RF300 OOM prevent adopting these artifacts under the current requirements. |
| Smaller RF ensembles (#59) | Keep the current production RF. | RF50/100/200 were compared with the matched RF300 baseline. RF200 is a candidate for a future complete-input confirmation study; missing-input exceptions prevent promoting it across all conditions. |

These issues close as **completed evaluations**. The limitations and possible
future investigations are recorded in this document; they are not active tasks
or new Issues. Any later experiment needs its own scope, configuration and new
result directory. The [experiment plan](edge-optimization-plan.md) and committed
tools describe reproduction and engineering controls. Controls added after the
archived measurements have unit/CI validation, not a rerun of the full matrix.

## Evaluation coverage

| Scope | Actual outcome |
|---|---|
| RF compression on the workstation | All 80 jobs completed: four tree counts, five splits, complete and missing-input conditions; 400 dataset/condition metric records. |
| Native RF on Nano | 39 of 40 jobs completed; RF300 double-missing reconstruction was administratively stopped without completed predictions. |
| Full-input RF ONNX conversion on x86 | RF50 and RF300 converted; native/ONNX numerical differences were quantified. |
| Full-input RF ONNX on Nano | RF50 completed; RF300 hit the recorded container memory limit. |
| Missing-input ONNX bundles and other model families | Not evaluated in this stage. No parity, resource or portability conclusion is claimed for them. |

The deployment decision applies to the tested RF paths. The initial full-input
comparison already exposed precision and loading-resource constraints; this
stage did not proceed to missing-bundle conversions or a single universal ONNX
runtime for all model families. Those untested extensions are not counted as
completed experiments.

Workstation: 80 jobs, 200 conditions, 400 unique dataset/condition metric rows.
RF50/100/200/300 use five matched stratified splits, first50,000-row source,
80/20 split, and external10,714 rows. Three missing sets cover BOD, NH3N and both,
with dropout, reduced retraining and indicator reconstruction. Previously
examined datasets are comparative reuse, not a new untouched holdout.

## Complete-input comparison

Accuracy numbers are means over five splits. Nano raw timing/RSS is seed0 under
one CPU,1100MiB RAM/3500MiB RAM+swap, alongside the unchanged API. Not HTTP/UI
latency or an isolated-device, empty-cache measurement.

|Trees|External MAE|Category agreement|Nano P50/P95 ms|Peak process MiB|
|---|---|---|---|---|
|50|0.418169|97.786%|15.09/15.49|470.2|
|100|0.410352|97.805%|23.40/23.90|676.0|
|200|0.407318|97.822%|40.38/41.58|922.7|
|300|0.405997|97.853%|58.07/67.87|1046.8|

RF200 mean MAE increases0.3256%, artifact bytes decrease about33.34%, warm
P50 decreases about30.46%, and process high-water decreases only about11.9%.
It is a complete-input candidate for limited confirmation, not a universal winner.
Two external missing-input seed conditions exceed exploratory thresholds:
seed1 double-missing reconstruction MAE +1.1142%; seed2 BOD reconstruction
category agreement -1.0454 percentage points. No statistical significance or
automatic promotion follows from this screen.

## Terminal outcomes and unresolved findings

Nano:40 completed,1 ONNX300 OOM,1 RF300 double-missing reconstruction stopped
after more than nine hours. Exit137 alone is not OOM: the timeout case has
`OOMKilled=false`. Nonblocking stack samples stayed in the same tree and chunk
offset2176; this does not uniquely identify a cause. Diagnostic GDB attach
attempts briefly paused that worker. The administrative stop was post-hoc,
not a preregistered deadline. It has no completed parity or valid latency sample.

RF300 BOD reconstruction:54/10,714 rows exceed1e-6, maximum difference0.6905,
one category change. Cause remains unresolved; compare reconstructed values and
tree branch thresholds before attributing it to ordinary floating-point error.

Of40 completed cases,14 have at least one failed health poll:135/11,244 polls,
64 TimeoutError and71 URLError. RF50 ONNX has6/18 failures. The API retained
its image, restart count0 and no OOM, but service continuity did not pass in
these cases. Network and resource contention have not been isolated.

## ONNX decision

RF50 full pipeline including imputation converts from175,540,436 to99,106,964
bytes. Native float64 vs ONNX float32 external max difference4.80835e-5;
9,585/10,714 exceed1e-6. Native-float32 and converter/runtime effects are
reported separately. No observed category changes in the tested complete-input
sets does not establish numerical equivalence. RF300 external max is2.49951e-4.

RF50 ONNX warm batch1 P50/P95 is0.088/0.131ms, process-cold load35.67s and
peak1059.6MiB; smaller files are not evidence of lower loading memory. RF300
hit OOM in the stated Nano container budget. Its workstation conversion also
hit an8GiB budget before succeeding in a14GiB budget, peak about11.8GiB;
conversion and inference memory are different costs.

Keep the native production format. Missing-bundle ONNX conversion is untested;
the worker now refuses ONNX missing-value requests instead of running full inputs.
No tolerance changes, quantization, production replacement or claim that every
Nano2GB configuration fails is warranted.

## Preserved deliverables

Private Markdown,7-sheet Excel,CSV and raw evidence remain in the separately
dated run. Old610 jobs,13 old parity findings and58 archived x86 files are
unchanged. WQI5 formula/dataset Score disagreement remains a limitation.

Local E-drive copy completed2026-11-02 18:00 CST:1,744 files,
126,450,969,295 bytes; each copied file SHA-256 verified. Manifest SHA-256:
`684f7d1491d1640dd5245e5198a1586567484d8c37261baecc222c3c64ae483e`.
Folder:`20261101-edge-optimization`. Remote Nextcloud readback and environment
restoration have not been verified. No private models,data,row predictions or
full workbooks are published in Git.

## Research limitations and future work

The following questions remain documented after this stage closes. None is a
claim of an established cause or a commitment to start another experiment.

1. **Cross-platform reconstruction differences.** RF300 BOD reconstruction has
   54 rows above the unchanged 1e-6 threshold and one category change. A future
   investigation could compare reconstructed indicators and the corresponding
   tree branch thresholds for those exact rows, with runtime and artifact hashes
   fixed. The current evidence does not establish ordinary floating-point error
   as the cause.
2. **Service continuity.** Fourteen completed cases contain failed health polls.
   A future study could align network-local and remote health probes with CPU,
   memory, swap and I/O samples to distinguish network faults from resource
   contention. Zero API restarts and no API OOM do not resolve those failures.
3. **RF300 double-missing noncompletion and ONNX300 OOM.** The isolated run's
   memory budget and shared API constrain the conclusion. The administrative
   timeout is post-hoc, with diagnostic interference recorded. A future bounded,
   instrumented reproduction could examine the stalled chunk and model stages;
   the current record neither proves a specific loop defect nor establishes
   failure for every Nano configuration.
4. **Comparison limits.** Five reused data splits support paired descriptive
   accuracy comparisons. Nano resource/timing measurements use seed 0, a fixed
   execution order and an uncontrolled OS cache. The 300 consecutive warm timing
   samples are not 300 independent cold starts. In-sample indicator reconstruction
   follows the original protocol and does not establish a new leakage-free method.
5. **Runtime and backup scope.** The archived ONNX conversion used version 1.17.0;
   the maintained research lock now uses 1.23.2 for security fixes. Archived parity
   has not been remeasured under that newer version. The verified local E-drive
   copy does not establish remote Nextcloud integrity or successful restoration.
6. **Untested ONNX extensions and how to evaluate them.** A future missing-bundle
   study would first convert each fitted reconstruction stage and its imputer,
   preserve feature order, then compose and validate the full path for BOD,
   NH3N and double-missing conditions. Start with the existing RF50 case, record
   native float64, native float32 and ONNX predictions separately, and inspect
   branch-boundary/category changes under the unchanged 1e-6 threshold. Only
   after x86 validation should a bounded Nano test measure loading, peak memory,
   warm latency and health continuity. Other model families require their own
   converter/operator-support and preprocessing checks. These are proposed
   evaluation steps, not results or an automatically scheduled continuation.

The WQI5 formula/dataset `Score` disagreement and the original replay's 13 parity
findings remain part of the [first-stage benchmark](jetson-nano-benchmark.md).
Neither closing this evaluation nor merging its tools resolves those limitations.
