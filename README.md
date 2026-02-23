# Agri Auditor

**Post-mission safety auditing for autonomous tractor operations.**
Deterministic CV/ML pipeline that transforms raw mission telemetry and surround-view camera feeds into ranked, explainable incident reports — with optional Gemini AI enrichment for field-level context.

---

## Executive Summary

Autonomous agricultural vehicles generate thousands of telemetry frames per mission across multiple IMU channels, GPS, 6-camera surround views, and stereo depth maps. Operators today lack a systematic, repeatable way to identify the highest-risk moments, understand root causes, and triage follow-up actions.

Agri Auditor solves this by implementing a physics-grounded, five-signal severity scoring model that fuses terrain roughness, obstacle proximity, steering dynamics, sensor health, and localization confidence into a single composite score per frame. The system detects the statistically most significant incidents using constrained peak extraction, optionally enriches each with a Gemini vision caption, and renders an interactive mission dashboard — map, synced telemetry charts, signal decomposition, and surround-view galleries — in a single portable HTML file or scalable split-asset bundle.

The pipeline is fully deterministic offline, schema-validated, CI-gated, and containerized. It processes 1,085 frames with depth features in under 1 second cold, and under 40 ms warm-cached.

---

## Key Capabilities

| Capability | Implementation |
|---|---|
| **5-signal severity model** | Weighted composite: roughness (0.35), proximity (0.15), yaw rate (0.20), IMU fault (0.15), localization fault (0.15) |
| **Robust normalization** | MAD-based z-score mapping (5th–95th quantile clip, ±3σ → [0,1]) with automatic minmax fallback |
| **Constrained peak detection** | `scipy.signal.find_peaks` with configurable prominence, width, and minimum inter-event distance |
| **Multiprocess depth extraction** | `ProcessPoolExecutor` with Parquet-backed persistent cache and in-memory hot cache |
| **Gemini AI enrichment** | SDK + REST dual path, exponential backoff with jitter, circuit breaker, SHA-256 deterministic cache |
| **Interactive HTML dashboard** | Leaflet dark-tile map, 4-row synced Plotly telemetry panel, event cards with signal bars and surround-view gallery |
| **Security-hardened rendering** | CSP nonce-restricted scripts, JSON payload isolation via `<script type="application/json">`, Jinja2 autoescape, `sanitize_model_text` |
| **Production governance** | JSON Schema–validated run metadata, CI lint/type/perf gates, DVC lineage stages, optional MLflow logging |

---

## System Architecture

```text
Mission Logs                          6-Camera Surround + Depth
(manifest.csv, calibrations.json)     (1,085 frames × 7 streams)
        │                                       │
        └──────────────┬────────────────────────┘
                       ▼
              ┌─────────────────┐
              │    LogLoader    │  Ingest manifest, calibrations,
              │  ingestion.py   │  camera intrinsics, velocity derivation
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  FeatureEngine  │  Roughness (high-pass RMS on dual IMU)
              │   features.py   │  Depth clearance + canopy proxy
              │                 │  Quaternion → Euler orientation
              │                 │  Rolling IMU cross-correlation
              │                 │  GPS cleaning + pose confidence
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  EventDetector  │  Normalize → weight → composite score
              │ intelligence.py │  Peak extraction → top-K candidates
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 ┌──────────────────┐     ┌────────────────┐
 │ GeminiAnalyst    │     │  ReportBuilder  │
 │ (optional)       │     │  reporting.py   │
 │ SDK/REST + cache │     │ single │ split  │
 │ circuit breaker  │     │ Plotly + Leaflet│
 └────────┬─────────┘     │ CSP hardened    │
          │               └────────┬────────┘
          └────────────┬───────────┘
                       ▼
            artifacts/features.csv
            artifacts/events.json
            artifacts/audit_report.html
            artifacts/*_assets/ (split mode)
```

---

## Scoring Model

Each frame receives a composite severity score:

$$S = 0.35 \cdot \hat{R} + 0.15 \cdot \hat{P} + 0.20 \cdot \hat{Y} + 0.15 \cdot \hat{I} + 0.15 \cdot \hat{L}$$

| Signal | Symbol | Derivation |
|---|---|---|
| **Roughness** | $\hat{R}$ | High-pass RMS of dual-IMU accel-z (camera + syslogic), averaged, then normalized |
| **Proximity** | $\hat{P}$ | $(10\text{m} - \text{clearance}) / (10\text{m} - \min)$, clipped [0, 1]. Clearance = 5th-percentile of center-crop depth |
| **Yaw rate** | $\hat{Y}$ | Quaternion-derived yaw rate (deg/s) with ±180° wraparound, abs-value normalized |
| **IMU fault** | $\hat{I}$ | $1 - \text{normalize}(\text{rolling cross-correlation of camera vs syslogic accel-z})$ |
| **Localization fault** | $\hat{L}$ | $1 - \text{normalize}(\text{pose\_confidence})$ |

Default normalization is **robust**: values are clipped to the 5th–95th percentile, a MAD-based robust sigma ($1.4826 \times \text{MAD}$) is computed, z-scores are clipped to [-3, 3] and mapped to [0, 1]. Falls back to minmax if the population is degenerate.

Events are classified by dominant signal component: `roughness`, `proximity`, `steering`, `sensor_fault`, `localization_fault`, or `mixed` (when the top two components are within 0.1 of each other).

---

## Repository Structure

```text
src/agri_auditor/
  ingestion.py         Dataset loading, camera intrinsics, velocity derivation
  features.py          Feature extraction, depth cache, multiprocess depth path
  intelligence.py      Severity scoring, peak detection, Gemini AI runtime
  reporting.py         Interactive HTML dashboard renderer (1,660 lines)
  cli.py               Unified production CLI with 5 subcommands
  config.py            RuntimeConfig dataclass + env-var parsing/validation
  logging_config.py    Structured logging (structlog JSON/console, stdlib fallback)
  mlops.py             Optional MLflow lineage logging (non-blocking)

scripts/
  prepare_test_data.py       Synthetic deterministic test data generator
  check_perf_budget.py       Cold/warm cache performance budget gate
  build_features.py          Standalone feature build entrypoint
  build_events.py            Standalone event build entrypoint
  build_report.py            Standalone report build entrypoint
  benchmark_gemini.py        Multi-model Gemini latency/throughput benchmarking

tests/                       108 deterministic tests + 3 live Gemini tests
schemas/                     run_metadata.schema.json (JSON Schema 2020-12)
.github/workflows/ci.yml    Lint → type check → schema → tests → perf gate → Gemini live
Dockerfile                   Non-root Python 3.13-slim production image
dvc.yaml                     Reproducible pipeline stages (features → events → report)
```

---

## Installation

Requires **Python 3.13+**.

```powershell
# Core runtime (features + events + CLI)
python -m pip install -e .

# Core + interactive report rendering
python -m pip install -e .[report]

# Full development environment (pytest, ruff, mypy, jsonschema)
python -m pip install -e .[dev]

# MLOps extras (MLflow, DVC)
python -m pip install -e .[mlops]
```

Core dependencies: `pandas`, `numpy`, `pillow`, `scipy`, `pyarrow`, `google-genai`, `python-dotenv`, `structlog`.
Report extras: `plotly`, `jinja2`.

---

## Quickstart

**1. Prepare dataset** (synthetic deterministic data, if you don't have real mission data):

```powershell
python scripts/prepare_test_data.py --output-dir ../provided_data
```

**2. Run the full pipeline** (Gemini-enabled primary path):

```powershell
$env:GEMINI_API_KEY = "your-key-here"

python -m agri_auditor process `
  --data-dir ../provided_data `
  --output-features artifacts/features.csv `
  --output-events artifacts/events.json `
  --output-report artifacts/audit_report.html `
  --report-mode split `
  --report-telemetry-downsample 2 `
  --report-feature-columns "timestamp_sec,_elapsed,gps_lat,gps_lon,velocity_mps,min_clearance_m,severity_score"
```

**3. Run offline** (deterministic fallback, no network required):

```powershell
python -m agri_auditor process `
  --data-dir ../provided_data `
  --output-features artifacts/features.csv `
  --output-events artifacts/events.json `
  --output-report artifacts/audit_report.html `
  --disable-gemini `
  --report-mode split
```

**4. Open** `artifacts/audit_report.html` in any browser.

---

## CLI Reference

```powershell
python -m agri_auditor --help
```

| Subcommand | Purpose | Key Flags |
|---|---|---|
| `features` | Build feature table only | `--data-dir`, `--output` |
| `events` | Build ranked event JSON | `--data-dir`, `--output`, `--top-k`, `--disable-gemini` |
| `report` | Render dashboard from precomputed events | `--events-json`, `--report-mode single\|split` |
| `process` | End-to-end: features → events → report | All of the above |
| `benchmark-gemini` | Latency/throughput benchmark across models | `--models`, `--repeats`, `--events-json` |

**Compose individual stages or run the full pipeline.** Precomputed artifacts are accepted at each stage boundary, enabling cache reuse and partial reruns.

Examples:

```powershell
# Offline events only
python -m agri_auditor events --data-dir ../provided_data --output artifacts/events.json --disable-gemini

# Report from prebuilt artifacts
python -m agri_auditor report --data-dir ../provided_data --events-json artifacts/events.json --output artifacts/report.html --report-mode split

# Benchmark Gemini models (p50/p95/p99 latency, error rate, token throughput)
python -m agri_auditor benchmark-gemini --data-dir ../provided_data --events-json artifacts/events.json --models gemini-3-flash-preview gemini-2.5-flash --repeats 3
```

---

## Data Contract

Default dataset path: `../provided_data` relative to repo root.

```text
provided_data/
  manifest.csv          1,085 rows × 29 columns (IMU, GPS, pose, depth flags)
  calibrations.json     Per-camera intrinsic parameters (fx, fy, cx, cy, resolution)
  frames/
    front_center_stereo_left/   1,085 JPEG frames
    front_left/                 1,085 JPEG frames
    front_right/                1,085 JPEG frames
    rear_left/                  1,085 JPEG frames
    rear_center_stereo_left/    1,085 JPEG frames
    rear_right/                 1,085 JPEG frames
    depth/                      16-bit PNG depth maps
```

Required manifest columns: `frame_idx`, `timestamp_sec`, `pose_front_center_stereo_left_{x,y,z,qx,qy,qz,qw}`, `has_depth`, `imu_camera_accel_z`, `imu_syslogic_accel_z`. Additional columns (`gps_lat`, `gps_lon`, `gps_alt`, `pose_confidence`) are used when present.

---

## Runtime Configuration

Configuration precedence: **CLI flags → environment variables → defaults**.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables live Gemini vision calls |
| `AGRI_AUDITOR_GEMINI_MODEL` | `gemini-3-flash-preview` | Model identifier |
| `AGRI_AUDITOR_GEMINI_TIMEOUT_SEC` | `20.0` | Per-call SLA timeout (seconds) |
| `AGRI_AUDITOR_GEMINI_WORKERS` | `4` | Parallel caption workers |
| `AGRI_AUDITOR_GEMINI_RETRIES` | `2` | Retry attempts per call |
| `AGRI_AUDITOR_GEMINI_BACKOFF_MS` | `250` | Exponential backoff base (ms) |
| `AGRI_AUDITOR_GEMINI_JITTER_RATIO` | `0.2` | Backoff jitter ratio |
| `AGRI_AUDITOR_GEMINI_CACHE_DIR` | — | Deterministic on-disk caption cache |
| `AGRI_AUDITOR_DEPTH_WORKERS` | `4` | Multiprocess depth extraction workers |
| `AGRI_AUDITOR_DEPTH_CACHE_DIR` | — | Parquet-backed depth feature cache |
| `AGRI_AUDITOR_SCORE_NORMALIZATION` | `robust` | `robust` (MAD-based) or `minmax` |
| `AGRI_AUDITOR_SCORE_Q_LOW` | `0.05` | Robust quantile low bound |
| `AGRI_AUDITOR_SCORE_Q_HIGH` | `0.95` | Robust quantile high bound |
| `AGRI_AUDITOR_PEAK_PROMINENCE` | `0.05` | Peak prominence threshold |
| `AGRI_AUDITOR_PEAK_WIDTH` | `1` | Minimum peak width |
| `AGRI_AUDITOR_PEAK_MIN_DISTANCE` | `150` | Minimum frames between events |
| `AGRI_AUDITOR_REPORT_MODE` | `single` | `single` (portable) or `split` (scalable) |
| `AGRI_AUDITOR_REPORT_TELEMETRY_DOWNSAMPLE` | `1` | Downsample factor for report payload |
| `AGRI_AUDITOR_REPORT_FEATURE_COLUMNS` | — | CSV list of columns to embed in report |
| `AGRI_AUDITOR_LOG_LEVEL` | `INFO` | Named or numeric log level |
| `AGRI_AUDITOR_LOG_FORMAT` | `auto` | `auto`, `json`, or `console` |
| `AGRI_AUDITOR_MLFLOW_ENABLED` | — | Enable MLflow run lineage |
| `AGRI_AUDITOR_MLFLOW_TRACKING_URI` | `artifacts/mlruns` | MLflow tracking URI |

---

## Performance

Measured on the reference 1,085-frame dataset with depth extraction, single worker:

| Metric | Value |
|---|---|
| Cold feature build | **0.90 s** |
| Warm feature build (Parquet cache hit) | **0.03 s** |
| Warm/cold ratio | **0.04** (26× speedup) |
| Performance budget ceiling | 20.0 s cold, 0.95 warm/cold ratio |
| Row-count parity | Cold = Warm = 1,085 rows |

The `ProcessPoolExecutor` depth path scales linearly with `--depth-workers` for missions with thousands of frames. Persistent Parquet cache uses atomic tmp-rename writes to prevent corruption. In-memory hot cache eliminates disk I/O for repeated access within a single run.

---

## Security Model

All model-generated text is treated as **untrusted input**:

1. **Sanitization** — `sanitize_model_text()` strips control characters, normalizes whitespace, and enforces a 600-character cap before any template binding.
2. **Payload isolation** — Data is embedded in `<script type="application/json">` blocks and bootstrapped via `JSON.parse(textContent)`. No inline JavaScript data variables.
3. **CSP enforcement** — Report template applies a Content Security Policy with nonce-restricted `<script>` execution and locked-down `object`, `frame`, and `base-uri` directives.
4. **Autoescape** — Jinja2 rendering has autoescaping enabled globally.

Gemini is **advisory only** — captions are displayed for operator context but never influence scoring, ranking, or pipeline control flow.

---

## Gemini Runtime Resilience

| Control | Implementation |
|---|---|
| Dual provider path | Google GenAI SDK primary, raw REST (`urllib.request`) fallback |
| Exponential backoff | Base 250 ms × 2^attempt, with ±20% uniform jitter |
| Per-call timeout | SDK calls wrapped in `ThreadPoolExecutor` with configurable SLA |
| Circuit breaker | Opens after 3 consecutive failures, 30 s cooldown |
| Deterministic cache | SHA-256 key over (model + system prompt + MIME type + temperature + thinking level + max words + image bytes), stored as JSON |
| Parallel workers | Bounded `ThreadPoolExecutor` (default 4) for multi-event caption batch |
| Non-retryable detection | HTTP 400, 401, 403, 404 bypass the retry loop |

---

## Testing

**108 deterministic tests** pass offline with zero network dependency.
**3 live Gemini integration tests** validate the primary AI-enriched path when `GEMINI_API_KEY` is configured.

```powershell
# Full deterministic suite
python -m pytest -q
# → 108 passed, 3 deselected

# Live Gemini integration (requires API key)
python -m pytest -q -m gemini_live -o addopts="-p no:cacheprovider"

# CI-equivalent local gates
python -m ruff check .
python -m mypy src
python scripts/check_perf_budget.py --summary-json artifacts/perf_budget_summary.json
```

Test coverage spans: ingestion loading, feature math, normalization edge cases, peak detection, scoring weights, Gemini retry/circuit/cache/timeout paths, report rendering (single + split), CSP/XSS hardening, CLI argument parsing, config parsing/validation, run metadata schema validation, and performance budget enforcement.

---

## Governance and MLOps

**Run metadata** is schema-validated against `schemas/run_metadata.schema.json` (JSON Schema 2020-12):

```json
{
  "dataset_hash":       "sha256 of manifest + calibrations",
  "code_version":       "git rev-parse HEAD",
  "config_fingerprint": "sha256 of sorted runtime config",
  "latency_summary": {
    "count": 5,
    "avg_ms": 1842.3,
    "p50_ms": 1650.0,
    "p95_ms": 2900.0,
    "p99_ms": 3100.0
  }
}
```

**CI pipeline** (`.github/workflows/ci.yml`):

1. `ruff check .` — lint gate
2. `mypy src` — static type check
3. Schema validation — `tests/test_run_metadata_schema.py`
4. Full deterministic test suite — `pytest -q`
5. Performance budget gate — `scripts/check_perf_budget.py`
6. Live Gemini suite — conditional on `GEMINI_API_KEY` secret

**Reproducibility**: `dvc.yaml` defines three stages (`features` → `events` → `report`) with explicit dependency tracking and artifact outputs. Optional MLflow lineage via `src/agri_auditor/mlops.py` is fire-and-forget — it never blocks the pipeline on logging failures.

---

## Docker

```powershell
# Build
docker build -t agri-auditor:latest .

# Run (mount dataset at /data)
docker run --rm -v "${PWD}\..\provided_data:/data" agri-auditor:latest
```

Image runs as non-root (`agri` user) on `python:3.13-slim`. Default entrypoint executes the full `process` pipeline against `/data`.

---

## Roadmap

| Phase | Status | Scope |
|---|---|---|
| 1. Security hardening | Complete | CSP nonce, JSON payload isolation, `sanitize_model_text` |
| 2. Report scalability | Complete | Split/single modes, image deduplication, downsample controls |
| 3. Gemini resilience | Complete | Jittered backoff, SDK timeout wrapper, circuit breaker, cache |
| 4. Depth performance | Complete | `ProcessPoolExecutor`, Parquet persistent cache, atomic writes |
| 5. Event scoring quality | Complete | Robust normalization defaults, constrained peak selection |
| 6. Packaging hygiene | Complete | Lazy imports, optional report extras, graceful fallback hints |
| 7. Log-level correctness | Complete | Numeric level as `int` in `RuntimeConfig`, `str` | `int` accepted |
| 8.1 MLOps governance | Complete | CI gates, run metadata schema, perf budget, DVC stages, MLflow hook |
| **8.2 Service orchestration** | **Next** | FastAPI submission layer, async worker queue, artifact store |

Full technical narrative: [docs/intelligence_roadmap.md](docs/intelligence_roadmap.md)
Execution contract: [docs/cto_upgrade_completion_handoff.md](docs/cto_upgrade_completion_handoff.md)

---

## License

Internal / proprietary. See project governance documentation for access and contribution policies.
