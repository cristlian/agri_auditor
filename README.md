# Agri Auditor

Production-oriented CV/ML incident auditing pipeline for autonomous tractor mission logs.

Agri Auditor ingests mission telemetry + camera/depth frames, computes physically grounded safety features, detects top incidents, enriches incidents with Gemini captions, and renders a security-hardened interactive mission report.

## Why This Exists

Autonomous operations teams need a deterministic post-mission audit pipeline that is:

1. Explainable: feature-driven scoring with traceable signals.
2. Operationally resilient: deterministic offline mode and robust AI fallbacks.
3. Scalable: split-asset reporting, downsample controls, parallel depth processing.
4. Production-governed: CI quality gates, run metadata schema, perf budgets.

## Current Validation Status

Validated locally on **February 22, 2026**:

1. `python -m pytest -q` -> `108 passed, 3 deselected`
2. `python -m ruff check .` -> pass
3. `python -m mypy src` -> pass
4. `python scripts/check_perf_budget.py` -> pass

## System Architecture

```text
mission logs + frames
        |
        v
  LogLoader (ingestion.py)
        |
        v
 FeatureEngine (features.py)
  - roughness
  - depth-derived clearance + canopy proxy
  - orientation (yaw/pitch/roll)
  - IMU/pose health
  - GPS cleanup
        |
        v
 EventDetector (intelligence.py)
  - robust/minmax normalization
  - weighted severity scoring
  - constrained peak selection
        |
        +-----------------------------+
        |                             |
        v                             v
 IntelligenceOrchestrator       ReportBuilder (reporting.py)
  - Gemini optional             - single-file mode (portable)
  - retries/backoff/jitter      - split mode (scalable assets)
  - timeout + circuit breaker   - CSP + safe JSON embedding
  - deterministic cache         - interactive map/charts/event feed
        |
        v
artifacts/features.csv
artifacts/events.json
artifacts/audit_report.html (+ optional *_assets/)
```

## Repo Layout

```text
src/agri_auditor/
  ingestion.py         # dataset loading, camera models, velocity derivation
  features.py          # feature extraction + depth cache + multiprocess path
  intelligence.py      # event scoring/detection + Gemini runtime
  reporting.py         # mission dashboard renderer (single/split modes)
  cli.py               # unified production CLI
  config.py            # runtime config + env parsing/validation
  logging_config.py    # structured logging setup
  mlops.py             # optional MLflow lineage logging

scripts/
  prepare_test_data.py
  check_perf_budget.py
  build_features.py
  build_events.py
  build_report.py
  benchmark_gemini.py

docs/
  intelligence_roadmap.md
  cto_upgrade_completion_handoff.md

schemas/run_metadata.schema.json
.github/workflows/ci.yml
Dockerfile
```

## Installation

Python requirement: **3.13+**

Install modes:

1. Core runtime:
```powershell
python -m pip install -e .
```

2. Core + reporting UI dependencies:
```powershell
python -m pip install -e .[report]
```

3. Development (tests, lint, typing, schema tooling):
```powershell
python -m pip install -e .[dev]
```

4. MLOps extras (MLflow, DVC):
```powershell
python -m pip install -e .[mlops]
```

## Data Contract

Default dataset location is `../provided_data` (relative to repo root), expected shape:

```text
provided_data/
  manifest.csv
  calibrations.json
  frames/
    front_center_stereo_left/
    front_left/
    front_right/
    rear_left/
    rear_center_stereo_left/
    rear_right/
    depth/
```

Required manifest fields include at least:

1. `frame_idx`
2. `timestamp_sec`
3. `pose_front_center_stereo_left_x|y|z`
4. `has_depth`
5. `imu_camera_accel_z`
6. `imu_syslogic_accel_z`

## Quickstart

1. Prepare deterministic synthetic test data (if needed):
```powershell
python scripts/prepare_test_data.py --output-dir ../provided_data
```

2. Configure Gemini API key (required for primary end-to-end run):
```powershell
copy .env.example .env
# then set GEMINI_API_KEY in .env or your shell environment
```

3. Run full pipeline (Gemini required):
```powershell
python -m agri_auditor process `
  --data-dir ../provided_data `
  --output-features artifacts/features.csv `
  --output-events artifacts/events.json `
  --output-report artifacts/audit_report.html `
  --report-mode split `
  --report-telemetry-downsample 2 `
  --report-feature-columns "timestamp_sec,_elapsed,gps_lat,gps_lon,velocity_mps,min_clearance_m,severity_score"
```

4. Fallback demonstration mode only (Gemini disabled):
```powershell
python -m agri_auditor process `
  --data-dir ../provided_data `
  --output-features artifacts/features.csv `
  --output-events artifacts/events.json `
  --output-report artifacts/audit_report.html `
  --disable-gemini `
  --report-mode split `
  --report-telemetry-downsample 2 `
  --report-feature-columns "timestamp_sec,_elapsed,gps_lat,gps_lon,velocity_mps,min_clearance_m,severity_score"
```

5. Open `artifacts/audit_report.html`.

## CLI Reference

Primary entrypoint:

```powershell
python -m agri_auditor --help
```

Subcommands:

1. `features` -> build feature table only.
2. `events` -> build event JSON only.
3. `report` -> build report from live detection or precomputed events.
4. `process` -> end-to-end run (features + events + report).
5. `benchmark-gemini` -> benchmark Gemini models on selected frames.

High-value examples:

1. Offline deterministic events:
```powershell
python -m agri_auditor events --data-dir ../provided_data --output artifacts/events.json --disable-gemini
```

2. Report from prebuilt events:
```powershell
python -m agri_auditor report --data-dir ../provided_data --events-json artifacts/events.json --output artifacts/audit_report.html --report-mode split
```

3. Gemini benchmark:
```powershell
python -m agri_auditor benchmark-gemini --data-dir ../provided_data --events-json artifacts/events.json --models gemini-3-flash-preview gemini-2.5-flash --repeats 3
```

## Runtime Configuration

Precedence order:

1. CLI flags
2. Environment variables
3. Defaults in `src/agri_auditor/config.py`

Core environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Enables live Gemini calls |
| `AGRI_AUDITOR_LOG_LEVEL` | `INFO` | Named or numeric log level |
| `AGRI_AUDITOR_LOG_FORMAT` | `auto` | `auto`, `json`, `console` |
| `AGRI_AUDITOR_GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model |
| `AGRI_AUDITOR_GEMINI_TIMEOUT_SEC` | `20.0` | Per-call SLA timeout |
| `AGRI_AUDITOR_GEMINI_WORKERS` | `4` | Gemini parallel workers |
| `AGRI_AUDITOR_GEMINI_RETRIES` | `2` | Retry attempts |
| `AGRI_AUDITOR_GEMINI_BACKOFF_MS` | `250` | Retry base backoff |
| `AGRI_AUDITOR_GEMINI_JITTER_RATIO` | `0.2` | Backoff jitter ratio |
| `AGRI_AUDITOR_GEMINI_CACHE_DIR` | unset | Deterministic Gemini cache |
| `AGRI_AUDITOR_DEPTH_WORKERS` | `4` | Depth extraction workers |
| `AGRI_AUDITOR_DEPTH_CACHE_DIR` | unset | Parquet depth cache directory |
| `AGRI_AUDITOR_SCORE_NORMALIZATION` | `robust` | `robust` or `minmax` |
| `AGRI_AUDITOR_SCORE_Q_LOW` | `0.05` | Robust low quantile |
| `AGRI_AUDITOR_SCORE_Q_HIGH` | `0.95` | Robust high quantile |
| `AGRI_AUDITOR_PEAK_PROMINENCE` | `0.05` | Peak prominence threshold |
| `AGRI_AUDITOR_PEAK_WIDTH` | `1` | Peak width threshold |
| `AGRI_AUDITOR_PEAK_MIN_DISTANCE` | `150` | Min frames between peaks |
| `AGRI_AUDITOR_REPORT_MODE` | `single` | `single` or `split` |
| `AGRI_AUDITOR_REPORT_TELEMETRY_DOWNSAMPLE` | `1` | Report payload downsample |
| `AGRI_AUDITOR_REPORT_FEATURE_COLUMNS` | unset | CSV list of included feature columns |
| `AGRI_AUDITOR_MLFLOW_ENABLED` | unset | Enable optional MLflow lineage |
| `AGRI_AUDITOR_MLFLOW_TRACKING_URI` | local `artifacts/mlruns` fallback | MLflow tracking path |

Reference file: `.env.example`

## Security and Safety Model

Report rendering treats model output as untrusted input:

1. Gemini caption text is sanitized (`sanitize_model_text`) before template binding.
2. Payload data is embedded via `<script type="application/json">` blocks and parsed with `JSON.parse`.
3. Report template enforces CSP with nonce-restricted scripts and locked-down object/frame/base policies.
4. Auto-escaping is enabled in Jinja2 rendering.

Safety posture:

1. Gemini is advisory only; it never controls decisions.
2. Primary success path requires Gemini-enabled execution.
3. `--disable-gemini` exists as an explicit fallback demonstration mode.

## Reliability and Performance

### Gemini runtime resilience

1. Bounded parallel caption workers.
2. Exponential backoff with jitter.
3. Per-call timeout wrapper.
4. Circuit breaker with cooldown.
5. Deterministic cache keyed by model + prompt + image bytes.

### Feature pipeline scalability

1. Multiprocess depth extraction path (`ProcessPoolExecutor`) with controlled fallback when spawn is restricted.
2. Persistent Parquet depth cache with atomic replace semantics.
3. In-memory hot cache for repeated access within run.

### Report scalability

1. `single` mode: portable self-contained HTML.
2. `split` mode: externalized JSON and image assets under `*_assets/`.
3. Payload size control via telemetry downsample + feature-column selection.

### Performance budget gate

`scripts/check_perf_budget.py` enforces:

1. Cold-run latency ceiling.
2. Warm-cache speedup ratio threshold.
3. Row-count parity between cold and warm runs.

## MLOps and Governance

Run metadata contract (schema-validated):

1. `dataset_hash`
2. `code_version`
3. `config_fingerprint`
4. `latency_summary` (`count`, `avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`)

Governance assets:

1. Schema: `schemas/run_metadata.schema.json`
2. Schema tests: `tests/test_run_metadata_schema.py`
3. Perf gate: `scripts/check_perf_budget.py` + `tests/test_perf_budget.py`
4. CI gates: `.github/workflows/ci.yml`
5. Optional lineage hook: `src/agri_auditor/mlops.py`
6. DVC stage scaffold: `dvc.yaml`

## Testing Strategy

Deterministic lane (default):

```powershell
python -m pytest -q
```

Live Gemini lane (required for validating primary path):

```powershell
python -m pytest -q -m gemini_live -o addopts="-p no:cacheprovider"
```

CI-equivalent local gates:

```powershell
python -m ruff check .
python -m mypy src
python scripts/check_perf_budget.py --summary-json artifacts/perf_budget_summary.json
```

## Docker

Build:

```powershell
docker build -t agri-auditor:latest .
```

Run end-to-end on mounted dataset:

```powershell
docker run --rm `
  -v "${PWD}\..\provided_data:/data" `
  agri-auditor:latest
```

The image runs as a non-root user and defaults to:

```text
python -m agri_auditor process --data-dir /data --output-features /data/features.csv --output-events /data/events.json --output-report /data/audit_report.html
```

## Roadmap and CTO Narrative

Primary roadmap: `docs/intelligence_roadmap.md`  
Execution contract: `docs/cto_upgrade_completion_handoff.md`

Current state aligns with the completed upgrade portfolio:

1. Security hardening for report rendering.
2. Split/single report modes for portability and scale.
3. Gemini resilience controls (retry/backoff/jitter/cache/circuit).
4. Multiprocess depth path + persistent cache.
5. Robust event scoring defaults.
6. Packaging/import hygiene for optional report stack.
7. Numeric log-level correctness.
8. CI and metadata governance hardening (8.1 complete, 8.2 staged next).

## Known Constraints and Next Phase

1. Primary run requires outbound network access and valid `GEMINI_API_KEY`.
2. Extremely long missions should prefer `--report-mode split`.
3. Next planned platform extension (8.2): thin service orchestration layer for multi-run scheduling and artifact serving.
