# Jetson Nano CPU inference

This deployment runs the existing v2 API on ARM64 without installing training
tools, CUDA, TensorRT or Python packages on the Jetson host. Training and model
promotion remain off-device. See [the artifact runtime contract](edge-runtime.md).

## Validated host

The initial device audit on 2026-10-26 found a Jetson Nano 2GB, Ubuntu 18.04.6,
L4T R32.7.1, kernel `4.9.253-tegra`, Docker 20.10.7, four Cortex-A57 cores,
1,972 MiB RAM and approximately 5 GiB swap. Host Python is 3.6.9; the container
supplies Python 3.11.17. This is CPU inference and does not depend on the host's
CUDA libraries. It is evidence for this configuration, not every JetPack image.

`Dockerfile.jetson` pins the Bookworm Python base by digest. The general
Dockerfile remains available for other deployments. Build natively on the Nano
or use `docker buildx build --platform linux/arm64` on a suitable build host:

```bash
docker build -f Dockerfile.jetson -t watermirror-edge:local .
# Only when these mounted model families are required:
docker build -f Dockerfile.jetson --build-arg MODEL_EXTRAS=models \
  -t watermirror-edge:local .
```

The default image supports direct WQI5 and Random Forest. The `models` extra
adds pinned XGBoost and LightGBM runtimes. The image copies only runtime source
and the public manifest; research artifacts and datasets are excluded.

## Mount and start

Copy approved artifacts into the exact paths declared by
`models/production_model_manifest.json`. Verify their SHA-256 checksums after
transfer. Do not train on the Nano or change the manifest to accept a different
artifact. Create a data directory even if no dataset is needed:

```bash
mkdir -p data
docker run -d --name watermirror-edge --restart unless-stopped \
  --log-opt max-size=10m --log-opt max-file=3 \
  -p 8001:8001 \
  -e DEFAULT_MODEL=direct_wqi5 -e AUTO_PORT=false \
  -e REQUIRE_DATASET_FOR_READINESS=false \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  --mount type=bind,src="$PWD/models",dst=/app/models,readonly \
  --mount type=bind,src="$PWD/data",dst=/app/data,readonly \
  watermirror-edge:local

curl --fail http://127.0.0.1:8001/api/v2/health
curl --fail --max-time 180 http://127.0.0.1:8001/api/v2/ready
curl --fail http://127.0.0.1:8001/api/v2/assessment \
  -H 'Content-Type: application/json' \
  -d '{"DO":96.2,"BOD":1.5,"NH3N":0.22,"EC":171,"SS":2.6,"model_type":"direct_wqi5"}'
```

The expected formula score is `89.542`. Repeat the assessment with
`"model_type":"rf"` to verify the mounted RF artifact. Its first request needs
an extended client timeout: the production RF artifact is 1,053,564,815 bytes
and a measured cold load took approximately 168 seconds on the audited device
under memory and I/O pressure. Warm inference timing excludes this cost.
Swap must be available for this artifact on the 2GB device. Never interpret
successful formula readiness as proof that every surrogate has been loaded.

Readiness and model listings reuse successful checksums while the resolved
file identity, size, mtime and ctime remain unchanged. A changed or missing
artifact invalidates the status cache. Actual model deserialization always
performs a fresh complete checksum validation.

For WaterMirror, set `EXPO_PUBLIC_API_BASE_URL=http://<nano-ip>:8001` in its local
`.env` and restart Metro. Alternatively save that URL in **Backend settings**
in an installed APK. Both devices must share a LAN or reachable Tailscale
network. Do not append `/api/v2`. Set `EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5`.
For a cold edge deployment, increase `EXPO_PUBLIC_REQUEST_TIMEOUT_MS` to cover
artifact validation and model loading. Keep a dataset mounted if the app needs
percentile and category-distribution views; formula readiness alone does not
require one. Restrict CORS to the actual web origin when exposing a web client.

## Repeatable measurements

Use `scripts/benchmark_edge.py` with the same source, manifest and artifact on
x86 and ARM64. Each invocation measures one model in a fresh process. Raw
prediction excludes DataFrame construction; HTTP timing includes the real API,
serialization and networking. The first prediction, model load, warmup and
steady-state timings are reported separately. Model load includes validation
and deserialization, but excludes dependency imports. A process's peak RSS and
CPU usage refer to that process, not the API server when it is an HTTP client.
Capture server resources separately with Docker statistics and `tegrastats`.

```bash
# On x86, from the repository root, with the locked model runtimes installed:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run --locked --extra models python scripts/benchmark_edge.py \
  --model rf --power-mode x86-reference --output /tmp/x86-rf.json

# Copy the reference JSON to Nano; mount it with a writable evidence directory.
# Run inside the deployed image with models mounted read-only and the script
# mounted at /app/benchmark_edge.py. Use a separate container/process for each
# raw model so RF memory is released before measuring the next model.
python benchmark_edge.py --model rf --power-mode MAXN-mode-0 \
  --samples 300 --reference /evidence/x86-rf.json \
  --output /evidence/raw-rf.json

# An HTTP client on the Nano measures loopback rather than Tailscale latency.
python benchmark_edge.py --model rf --power-mode MAXN-mode-0 \
  --api-url http://127.0.0.1:8001 --samples 300 \
  --reference /evidence/x86-rf.json --output /evidence/http-rf.json

# At least 30 minutes of sequential, sustained requests after warmup:
python benchmark_edge.py --model rf --power-mode MAXN-mode-0 \
  --api-url http://127.0.0.1:8001 --duration-seconds 1800 \
  --reference /evidence/x86-rf.json --output /evidence/soak-rf.json
```

Repeat for `direct_wqi5`, `lightgbm` and `xgboost` where deployed. Outputs are
created exclusively to preserve previous experiments. A failed request or
parity check makes the command fail. Raw parity tolerance defaults to `1e-6`;
HTTP permits `0.000501` because API scores are rounded to three decimals.
The five fixed probes are synthetic deployment checks. Their MAE versus the
formula is not the held-out accuracy reported by the production manifest.

Record `sudo nvpmodel -q`, the original mode, CPU governor/frequencies, image
ID and package versions for every experiment. This device's mode 0 is named
`MAXN`, not `10W`; do not relabel it as a measured ten-watt draw. Its other
configured mode is `5W` (mode 1). If comparing modes, use `sudo nvpmodel -m 1`,
repeat the measurements, then restore the original mode. Do not change power
modes during a sustained run. Report actual power only if an appropriate
sensor or external meter is available; power-mode labels are not measurements.
