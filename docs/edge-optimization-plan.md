# Edge optimization: independent second-stage evaluation

The archived plan below remains unchanged in scope. The [ONNX extension](onnx-extension-evaluation.md) completed its actual measurements in a separate result root.

Authorized on 2026-11-01, after completion of WaterMirror Edge Deployment v1.
Issues #58 and #59 record this completed evaluation. The closure decisions,
research limitations and optional future work are maintained in the
[results document](edge-optimization-results.md); no further experiment is
scheduled by this plan. Source branch: `feature/edge-optimization`.
Production remains WQSurrogateModels 2.3.0 / WaterMirror 2.3.6.

## Fixed comparisons

1. Train RF candidates independently at 50, 100, 200 and 300 trees on the workstation.
   Keep `random_state=0`, default tree parameters, and the established five
   stratified splits (seeds 0–4). Vary only tree count. No truncation of the
   production model. Training `n_jobs=2` limits host contention; inference uses
   one thread. Training time is separate from inference time.
2. Match the missing-indicator protocol: first 50,000 rows, 80/20 stratified
   split, final 10,714 rows as the external evaluation set. Validate the 50,000
   row source against the full dataset prefix before training. Save row IDs.
3. Complete-input references: 20 jobs. Three missing sets × four tree counts ×
   five splits: 60 further jobs. Each missing job evaluates inference dropout,
   reduced retraining, and indicator reconstruction using training data only.
   Reconstruction follows the established in-sample reconstruction contract;
   this preserves comparability and is not a new leakage-free method claim.
   Total: 80 checkpointed jobs, 200 condition evaluations on two datasets.
4. Keep all predictions, MAE/RMSE/R², category agreement, macro F1, model bytes,
   single-thread native latency, and process resource records. Repeat the 107
   consecutive-window controlled stress summaries with the existing five
   scenarios and three severity levels. These are synthetic perturbations,
   not real event validation or 107-fold cross-validation.
5. ONNX starts with seed-0 50- and 300-tree complete-input RF pipelines, including
   their fitted imputer. Evaluate original float64 native input, native float32
   input, and ONNX float32 separately. Do not describe ONNX as lossless, faster,
   or smaller before measuring. Absolute parity tolerance stays 1e-6; record
   every exceedance, category change, converter failure and runtime blocker.
   Extend to missing bundles only after the first whole-pipeline path works.
6. Run a separate Nano CPU benchmark with immutable copied artifacts, one thread,
   fixed representative inputs, batch 1 and 128, cold-process load/RSS and warm
   P50/P95. Verify native and ONNX outputs against the workstation, including
   near-category-boundary and missing-input probes. Do not restart or replace
   `watermirror-edge`. No claim of empty OS caches or measured watts.

## Interpretation and boundaries

The exploratory screen is at most 1% relative MAE increase and 1 percentage
point category-agreement loss versus the newly trained matched 300-tree model,
with at least 20% improvement in the stated resource metric. Report per-seed
paired changes and variability; a screen is not statistical significance or
automatic production approval. Do not select different winners independently
on the external set and then call that set untouched. Both datasets have been
examined previously; this is comparative reuse, not a new pristine holdout.
Any later model promotion needs separate review.

The original 610 jobs, 13 parity findings, 58 archived x86 files, production
manifest/models and three old backup folders remain unchanged. The WQI5 formula
versus dataset `Score` discrepancy is unresolved. Missing-input surrogate scores
do not replace a deterministic WQI5 definition requiring all five indicators.

## Execution and deliverables

Use the hash-locked `requirements-edge-optimization.txt` in an isolated venv.
`scripts/edge_optimization.py` runs one subprocess per job, validates completed
output checksums on resume, and retains failed partial directories. A user
systemd service provides logs and bounded restart; it is not a chat terminal.
Never count an active or failed job as complete. A job marker is published only
after output hashes exist. Source/config/input identities bind the run.

Private output: `results/edge_optimization_20261101/`, distinct from all old
results. Write CSV/Markdown/Excel, environment and source identities, data/model
SHA-256, and failure records there. Public Git contains code/configuration and
reviewed compact aggregate tables only, never models, data or row predictions.
Final E-drive backup must use a new `20261101-edge-optimization` dated folder
(or a fresh later date), SHA-256 verify source/copy, and preserve all old folders.
Remote Nextcloud readback is a separate verification, not implied by local copy.

Stages are separately reported: environment/preflight → RF workstation matrix →
ONNX conversion/parity → Nano measurements → report/workbook/backup → draft PR.
User review is required before merge. No release is needed for experiments alone.

## Engineering revision after the archived run

The archived `20261101` source snapshots and results are immutable. The current
branch adds prospective RF job deadlines (3600s) and a required explicit Nano
deadline (example budget 14400s); these were **not** the original protocol.
The last archived Nano run was administratively stopped after over nine hours,
not stopped by these new controls. Never retroactively relabel it as passing a
predeclared timeout policy or rerun into the old output directory.

RF children run in a dedicated process group: TERM, then KILL after a grace
period, with logs and execution JSON retained. The whole group is killed even
when the leader honours TERM, so TERM-ignoring descendants cannot survive.
Everything after launch, including the first evidence write, is supervised, and
every cleanup wait is bounded; a worker that cannot be reaped (for example in an
uninterruptible kernel wait) is recorded as `termination: unconfirmed` instead
of blocking the controller. A timeout cannot publish a successful completion
marker. Partial output directories remain available.
Workers now report input verification, model loading/training, prediction
chunks, stress evaluation and timing repeats. A running-stage message is not a
completed-row count; unchanged chunk progress is a reason to investigate.

`scripts/edge_optimization_nano.py` supervises a pre-staged research packet.
It refuses production container names and existing container/output names,
labels each new container with a random ownership token, limits each remote
control operation to30s, and retains stopped containers plus state/log evidence.
The image's `timeout` independently bounds the worker even if SSH/controller
dies; the supervisor separately enforces the wall deadline. Inputs stay mounted
read-only and no production container is stopped. A typical invocation uses:

```bash
python scripts/edge_optimization_nano.py \
  --host USER@NANO --control /tmp/owned-control --name wm-opt-NEW-RUN \
  --image watermirror-edge-onnx:PINNED-EVALUATION-TAG \
  --packet /home/USER/edge-optimization-NEW/packet \
  --worker /home/USER/edge-optimization-NEW/worker \
  --remote-output /home/USER/edge-optimization-NEW/output \
  --evidence results/NEW-RUN/execution.json --timeout-seconds 14400 \
  --kind native --model-filename full_reference.joblib
```

Run through systemd `LoadCredentialEncrypted` with credential name
`nano-password`; no password argument or file is published. Pre-stage the exact
worker as `benchmark.py` with `edge_optimization_paths.py` alongside, the model
and `inputs.npz`, verifying hashes before
running; this supervisor does not choose/transfer private data or promote models.
The research image must contain coreutils `timeout` and have no ENTRYPOINT.
The wrapper binds filenames to the standard packet layout before any remote
operation: native full/dropout use `full_reference.joblib`, reduced uses
`reduced_retraining.joblib`, reconstruction uses `indicator_reconstruction.joblib`,
and the supported complete-input ONNX case uses `pipeline.onnx`. Kind, mode and
missing-indicator operands are serialized from validated domain constants.
Only complete-input pipelines have an ONNX conversion, so the worker and this
supervisor refuse `--kind onnx` with a missing-value mode or `--missing`, and
any mode/`--missing` combination whose labels would not describe the
computation, before creating remote or local outputs. `--missing` accepts only
the five indicators, without repeats. After loading a native bundle the worker
also requires its trained `missing_indicators` to equal `--missing` for
`reduced_retraining`/`indicator_reconstruction` (which compute from the
bundle's own features), and a complete-input bundle for `full_reference` and
`inference_dropout`; all 42 archived Nano tasks satisfy these checks.

Ordinary CI runs pure integrity/deadline tests using `.[dev]`, without psutil
or ONNX at test collection. `Edge optimization research checks` installs the
hash-locked research environment and directly imports conversion dependencies;
its actual imputer/RF forward-path test must run with zero skipped tests.
Missing research dependencies fail that job rather than silently skipping it.
The archived `20261101` conversions used onnx 1.17.0, as recorded in that run's
provenance lock and `source.json`. The committed research lock now pins onnx
1.23.2 (plus its `ml-dtypes` dependency) for the 2026 external-data, path and
shape-inference security fixes; archived ONNX parity was not re-measured with it,
so any new ONNX comparison must record its own versions.

For a standalone ONNX conversion, use the same bounded subprocess entry point:

```bash
python scripts/edge_optimization_control.py --timeout-seconds 3600 \
  --evidence results/NEW-RUN/onnx-execution.json --log results/NEW-RUN/onnx.log \
  -- python scripts/edge_onnx_evaluate.py --job results/NEW-RUN/jobs/JOB \
  --data-root /path/to/private-inputs --output results/NEW-RUN/onnx
```

Existing evidence/log/output paths are refused.

The CSV/Excel package is exported from completed evidence only:

```bash
python scripts/edge_optimization_report.py --root results/RUN \
  --destination results/RUN/reports/NEW-EXPORT
```

The experiment matrix comes only from a config that byte-matches a source hash
in `identity.json`: the repository `configs/edge_optimization.yaml`, else the
run's `provenance/source/configs` copy (used by the archived `20261101` run,
because the committed config later gained prospective deadlines); no config
path is accepted from the command line. Each consumed `metrics.csv`
and Nano `result/result.json` is re-verified against its completion SHA-256,
and each job's rows must be exactly its (seed, trees, missing, mode, source)
keys; a full export requires the whole matrix (400 rows for this config).
Destinations follow the artifact rules below and, inside the run, must be under
`reports/`. ONNX `result.json` files have no separate completion digest and are
read as recorded. Research CI needs only tiny
synthetic fixtures; it does not train on private research data or contact Nano.
Write destinations are restricted to designated `results/` trees, the `/output/`
container mount, or temporary scratch. Traversal components, symbolic links and
input/model/Git directories are refused. Metadata uses exclusive random temporary
files and atomic publication; predictable `.tmp` links cannot redirect writes.
