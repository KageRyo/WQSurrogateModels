# Replaying workstation experiments on Jetson Nano

The replay workflow keeps archived x86 results and model artifacts read-only.
All new outputs belong in a separate dated directory, such as
`results/jetson_nano_20261026/`. Do not point a replay command at an archived
experiment directory. Raw datasets, model artifacts, packets and row-level
predictions remain local/private; only explicitly selected summaries belong
in Git.

## Scientific scope

Replay the complete-input, reduced-indicator, original double-missing,
expanded missing-indicator robustness, sample-size sensitivity and Stress107
protocols. Preserve the original five seeds, model families, feature order,
stratified splits, internal/external row sets, missing indicators, stress
scenarios and perturbation levels. The sample-size protocol retains its four
dataset sizes and all five folds, with both train and test predictions.

The expanded robustness workflow includes complete-input reference,
inference-time dropout, reduced-feature retraining artifacts and indicator
reconstruction artifacts. It covers unavailable BOD, unavailable NH3N, and
both unavailable. Direct WQI5 is evaluated only with complete input.

Archived models are reused where saved. The earlier complete-input and
reduced-indicator scripts did not save their estimators, so those protocols
need new **off-device CPU training references**, recorded separately. These
are new reference experiments, not claims to have reproduced the original
GPU training bit for bit. Nano performs inference on the same exported
estimators as the new x86 parity reference. No archived result is overwritten.

Stress107 means 107 consecutive event windows, five controlled synthetic
event types and 30%, 100%, and 300% perturbations. It is not 107-fold
cross-validation or observed real-world event validation.

## Runtime and bounded memory

The research artifacts record NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.5.2,
SciPy 1.15.3, joblib 1.5.3, XGBoost 3.2.0 and LightGBM 4.6.0.
`Dockerfile.edge-research` uses a separate locked environment for this contract.
It does not change the production API image's model runtime.

Reconstruction bundles can contain several large RF estimators. Preparation
exports their existing estimators as separate, losslessly compressed
components. This changes storage and execution scheduling, not learned
parameters or precision. Each component runs in its own child process, so its
memory is released before the next component loads. The final WQI estimator
uses the reconstructed indicators for the same input rows. Unit verification
compares this staged path with the original bundle prediction functions.

Only one packet is transferred and executed at a time. Every source artifact,
exported component and input packet has a SHA-256 record. A missing or changed
component is rejected before deserialization. Successful remote staging is
removed only after results have been copied back and verified; source models
and workstation results are never removed.

Inference timing is the **sum of component inference times** with the original
warmup/repeat settings. It excludes model loading, dependency imports,
inter-process orchestration and transfer. Those costs are recorded separately.
Peak component-process RSS is not whole-device or whole-container memory;
driver RSS is reported separately. The separate API benchmark measures actual
HTTP latency, including serialization and service work.

## Commands

Install the locked research runtime into an isolated environment on x86.
Keep `PYTHONPATH` pointed at this repository's `src` and `scripts`, and set
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` before starting either command.

```bash
python scripts/replay_edge_experiments.py prepare \
  --output results/jetson_nano_20261026/replay-staging

python scripts/replay_edge_experiments.py run \
  --staging results/jetson_nano_20261026/replay-staging \
  --output results/jetson_nano_20261026/replay-results \
  --remote user@nano-ip \
  --remote-root /home/user/water-quality/edge-research-20261026 \
  --ssh-control /path/to/existing-authenticated-ssh-control-socket
```

For a password-authenticated long experiment, establish the SSH control
connection with `ControlPersist=yes` and keepalive enabled, rather than a
fixed one-hour control-connection lifetime:

```bash
ssh -MNf -S /tmp/watermirror-nano-control \
  -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
  user@nano-ip
```

Pass that socket to `--ssh-control`. The runner asks for sudo credentials once and retains them in memory, not
files or process arguments. It waits for the named sustained-test container
to finish successfully before building or running research workloads.
Configure `--wait-container` and `--api-container` for the task-owned containers
on that host. The API is temporarily stopped to release its model cache during
research replay and restarted when the runner finishes or fails.

For an unattended password-authenticated run, add `--ssh-password-auth`. The
runner prompts separately for the SSH password and keeps it in process memory.
If authentication fails before a remote command starts, it renews the control
connection through SSH askpass and retries that unauthenticated operation once.
The temporary askpass helper contains code only, not the credential. Host-key
checking remains enabled. A connection lost during a potentially running
inference is not replayed automatically; the existing worker must be inspected
and its completed output recovered.

Preparation can resume verified complete packets. A completed packet is made
visible only when its job manifest has been written. The runner skips only
outputs whose job checksum, prediction keys, row counts, finite values, ARM64
platform and numerical comparison against the saved x86 arrays pass validation.
After reconnecting, the runner inspects an interrupted remote job. It waits if
that worker is still running; if the worker finished, it downloads and verifies
its result without repeating inference. Previous partial downloads are retained.
An incomplete remote job without successful output requires investigation;
the runner does not silently restart it or delete its partial results.

For a research sweep, add `--record-parity-failures` to `run`, `aggregate`, and
`backup` to measure all models while retaining numerical mismatches as failed
parity findings. The `1e-6` tolerance, predictions and category disagreements
are unchanged. This option does not accept missing rows, non-finite outputs,
wrong artifacts or incomplete workers. Completion of the experiment is
reported separately from cross-platform prediction agreement; it does not
make a failing model eligible for deployment.

The 2026-10-26 replay uses this explicit research option after identifying an
ill-conditioned archived LR double-missing reconstruction model. Its second
stage has coefficients of approximately `1.46e13` and a design condition
number of approximately `1.24e16`; reconstruction differences near `1e-15`
produce score differences as large as `0.079` on the audited probes. Original
models and negative findings are retained, rather than rounding predictions,
changing the tolerance, or replacing the estimator.

After every expected packet succeeds on both platforms:

```bash
python scripts/replay_edge_experiments.py aggregate \
  --staging results/jetson_nano_20261026/replay-staging \
  --results results/jetson_nano_20261026/replay-results \
  --output results/jetson_nano_20261026/reports/replay-complete
```

Aggregation refuses incomplete runs or an existing destination. It produces
per-seed metrics, summary metrics, bootstrap confidence intervals, paired
tests, WQI-band errors, timing/resource tables, prediction parity, global
stress results and Stress107 detection/monotonicity summaries. Hardware,
runtime versions, power modes and raw deployment evidence accompany the
final dated backup. Feature branches remain subject to user review before
merge or release.

The backup command verifies all archived-result checksums, requires a complete
parity-verified run, and compares copied files with their sources. It refuses
an existing destination, writes into a temporary sibling, and renames that
directory only after verification. Include private model packets and datasets
in this backup, never in the Git publication. Reports outside the experiment
are copied into `report/`; archived x86 comparison files are copied under
`original-x86/`. The snapshot includes the untracked replay systemd units.
Private replay packets contain lossless model components; original archived
model bundles remain referenced by source hash rather than duplicated. Hardware
and runtime provenance under the experiment are copied with all other results.
Only pass a reviewed evidence directory: obvious credential filenames are skipped,
but arbitrary log contents are not automatically redacted.


```bash
python scripts/replay_edge_experiments.py backup \
  --experiment results/jetson_nano_20261026 \
  --report results/jetson_nano_20261026/reports/replay-complete \
  --destination /path/to/backup/20261026-nano
```
# Automatic recovery on the x86 controller

The separate `watermirror-report.timer` checks every five minutes for the
completed replay marker. It runs `finalize` only after all packets finish.
Aggregation verifies every job again, retains failed attempts in separate
directories, and publishes `reports/replay-complete` only after its job counts
and parity findings agree with the completed runner. This prepares statistical
tables; narrative reports, final deployment checks and backup remain separate
completion steps. Inspect its logs with
`journalctl --user -u watermirror-report.service`.

`deploy/systemd/watermirror-replay.{service,timer}` contains the configuration
for this dated run. The user timer checks every two minutes. An active service
is left running; an exited controller is started again. A completed run is
excluded by `runner-complete.json`. The user manager must be running (this host
has lingering enabled); shutdown of WSL or loss of power pauses computation
until the host starts again. This does not detect or kill a live but hung model.

The runner holds an exclusive lock per output directory. Systemd owns the local
worker process group, so a failed service cannot leave a local inference worker
behind when it restarts. `--resume-incomplete` preserves unfinished x86 output
under `interruptions/` before retrying that packet. On Nano it waits for an
existing container, collects completed output, or archives terminal incomplete
staging under the remote `interruptions/` directory before retrying. Verified
results are reused; failed numerical parity stays recorded as a finding.

The service reads `nano-password` through `LoadCredentialEncrypted`, from
`~/.config/watermirror-edge/nano-password.cred`, encrypted for this host/user
with `systemd-creds encrypt --user --name=nano-password`. No password is placed
in source, Git, service arguments, or logs. This run uses the same credential
for Nano SSH and sudo. Credential rotation requires replacing that encrypted
file. Keep it and the host credential key outside shared experiment backups.

Inspect with `systemctl --user status watermirror-replay.service` and
`journalctl --user -u watermirror-replay.service`. Disable automatic retries
with `systemctl --user disable --now watermirror-replay.timer`; stopping the
service alone leaves the timer enabled. Install the supplied units into
`~/.config/systemd/user/`, run `systemctl --user daemon-reload`, then
`systemctl --user enable --now watermirror-replay.timer`.

Before enabling the supplied public service template, copy
`deploy/systemd/replay.env.example` to `~/.config/watermirror-edge/replay.env`
and set its SSH target and absolute remote staging directory for your device.
The template deliberately contains no real device address or account.
