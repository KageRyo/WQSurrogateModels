# RF compression and ONNX evaluation — review evidence

Status: measurements and local backup complete; independent review remains open.
No production promotion or app release. Engineering deadlines/progress added
after these measurements have not been validated by replaying this full matrix.

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

Keep the native production format. Missing-bundle ONNX conversion is untested.
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
full workbooks are published in Git. See issues #58 and #59 for review status.
