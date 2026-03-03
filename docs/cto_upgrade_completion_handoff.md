# Engineering Design & Implementation Record

Last updated: February 21, 2026  
Scope: `agri_auditor`

## 1. Objectives

Build a production-grade post-mission auditing pipeline with the following priority ordering:
1. Security and correctness first.
2. Measurable performance and scalability second.
3. Operational maturity and team readiness third.
4. Full deterministic test suite green.

## 2. Design Decisions

Selected implementation approach per subsystem:

| Area | Approach chosen | Rationale |
|---|---|---|
| Report security | JSON payload script blocks + CSP nonce + `sanitize_model_text` | Eliminates XSS surface from model-generated content without parsing overhead |
| Report scalability | Dual-mode single/split rendering + payload downsample + selective feature columns | Single mode for portable demos; split mode for production with thousands of frames |
| Gemini resilience | Jittered exponential backoff + per-call SLA timeout + circuit breaker + SHA-256 deterministic cache | Covers rate-limiting, partial outages, and full outages with graceful degradation |
| Depth performance | `ProcessPoolExecutor` + Parquet persistent cache + atomic writes + in-memory hot cache | CPU-bound depth extraction scales linearly; Parquet enables fast columnar reads across runs |
| Event scoring | Robust MAD-based normalization + configurable peak prominence/width/distance | Resistant to outliers and degenerate distributions across different mission profiles |
| Packaging | Optional dependency groups + lazy imports + graceful CLI fallback hints | Core pipeline runs without heavy report or MLOps dependencies |
| Observability | Numeric/named log levels as `int` in RuntimeConfig, structlog JSON/console modes | Supports both human debugging and structured log aggregation |
| MLOps governance | Run metadata schema + CI gates + DVC stages + optional MLflow | Reproducibility and auditability without imposing hard dependencies on MLflow availability |

Public interface contracts:
1. `--report-mode single|split` + payload reduction controls.
2. Gemini runtime knobs: workers, retries, backoff, jitter, cache dir, timeout.
3. Scoring knobs: robust scaling + prominence/width/min-distance.
4. Report dependencies as optional extras + lazy import with install hint.
5. Run metadata schema: `dataset_hash`, `code_version`, `config_fingerprint`, `latency_summary`.

## 3. Implementation Snapshot

### 3.1 Implemented

1. **Optional report dependency split and lazy imports:**
   - `pyproject.toml` optional extras: `[report]`, `[mlops]`, `[dev]`
   - Lazy import with `RuntimeError` + install hint in `src/agri_auditor/cli.py`

2. **Robust scoring and peak controls:**
   - MAD-based z-score normalization with 5th–95th quantile clipping
   - `scipy.signal.find_peaks` with configurable prominence, width, and minimum inter-event distance
   - Mixed-type event classification when top two signal components are within 0.1

3. **Gemini reliability:**
   - Dual provider path (SDK primary, REST fallback)
   - Exponential backoff with jitter: `base × 2^attempt × jitter_factor`
   - SDK timeout via `ThreadPoolExecutor` wrapper
   - Circuit breaker: opens after 3 consecutive failures, 30 s cooldown
   - SHA-256 deterministic cache keyed on (model + prompt + MIME + temperature + image bytes)
   - Non-retryable HTTP status detection (400, 401, 403, 404)

4. **Report rendering:**
   - Split mode: external JSON + image assets under `*_assets/`
   - Single mode: fully self-contained HTML
   - Payload downsample and feature column selection
   - CSP nonce-restricted scripts, Jinja2 autoescape

5. **Security hardening:**
   - `sanitize_model_text()`: strips control chars, normalizes whitespace, enforces max length
   - Data embedded in `<script type="application/json">` blocks, bootstrapped via `JSON.parse(textContent)`
   - No inline JavaScript data variables

6. **Run metadata:**
   - JSON Schema 2020-12 validated: `dataset_hash`, `code_version`, `config_fingerprint`, `latency_summary`
   - Schema gate in CI

7. **Deterministic test suite:**
   - 108 passed, 3 deselected (`gemini_live`)

### 3.2 Gaps Addressed During Implementation

These issues were identified during development and resolved in the same pass:

1. **Inline JS data injection** → replaced with JSON script blocks + `JSON.parse` bootstrap.
2. **Duplicate image embedding** → deduplicated in event payload; split mode externalizes images.
3. **No backoff jitter** → added configurable jitter ratio (`AGRI_AUDITOR_GEMINI_JITTER_RATIO`).
4. **No SDK-side timeout** → added `ThreadPoolExecutor` wrapper with configurable SLA timeout.
5. **Thread-based depth extraction** → replaced with `ProcessPoolExecutor` (with fallback for restricted environments).
6. **JSON depth cache** → replaced with Parquet-backed persistent cache with atomic writes.
7. **String-typed log levels** → numeric levels now stored as `int` in `RuntimeConfig`.
8. **Missing CI gates** → added lint, type check, schema validation, and perf budget gates.

## 4. Work Package Breakdown

### WP-1: Security Hardening

**Goal:** Eliminate XSS surface from model-generated content in rendered reports.

Changes:
1. Replace inline data var injection with `<script type="application/json" id="...">` blocks.
2. Parse via `textContent` + `JSON.parse` in JS bootstrap.
3. `sanitize_model_text(value: str) -> str`: strip control chars except `\n`/`\t`, normalize whitespace, enforce max length.
4. Apply sanitizer in `_event_ctx` for `gemini_caption` before template binding.

Validation:
- JSON script blocks present; inline JS data vars absent.
- Script-break payload `</script><script>alert(1)</script>` rendered inert.
- Sanitizer strips disallowed control chars.

### WP-2: Report Payload Scalability

**Goal:** Prevent payload size from growing linearly with mission length.

Changes:
1. Remove duplicate image embedding paths.
2. Split mode: write image assets to `*_assets/images/`, reference with relative paths.
3. Single mode: avoid duplicating same image data across fields.
4. Run-time downsample and feature column selection.

Validation:
- Split mode produces materially smaller HTML with externalized assets.
- Single mode remains fully self-contained.

### WP-3: Gemini SLA and Resilience

**Goal:** Enforce bounded latency and graceful degradation under API instability.

Changes:
1. Jittered exponential backoff: `sleep = base × 2^attempt × uniform(1 - jitter, 1 + jitter)`.
2. SDK timeout: `ThreadPoolExecutor` wrapper with configurable SLA.
3. Preserve circuit breaker + cache determinism.

Validation:
- Retry path for `429` and `5xx` with jittered sleeps (mock sleep and random).
- SDK timeout triggers retries then failure.
- Cache hit bypasses provider calls.
- Circuit breaker opens after consecutive failure threshold.

### WP-4: Depth Extraction Performance

**Goal:** Scale depth feature extraction for large missions via parallelism and caching.

Changes:
1. Replace `ThreadPoolExecutor` with `ProcessPoolExecutor` (top-level worker function for pickling).
2. Replace per-frame JSON cache with Parquet-backed persistent cache.
3. Atomic write strategy for cache replacement.
4. In-memory hot cache for current run.

Cache schema columns: `cache_key`, `frame_idx`, `min_clearance_m`, `canopy_density_proxy`, `mtime_ns`, `size_bytes`, `params_hash`.

Validation:
- Process pool path executes when `depth_workers > 1`.
- Warm Parquet cache avoids image decoding.
- Cache invalidates when depth file metadata changes.

### WP-5: Event Scoring Quality

**Status:** Implemented; regression guarded.

Robust normalization and peak constraints remain as defaults with full test coverage.

### WP-6: Packaging and Import Hygiene

**Status:** Implemented; regression guarded.

`features` and `events` subcommands run without report extras. Report commands fail gracefully with install hint when extras missing.

### WP-7: Log-Level Type Correctness

**Goal:** Accept numeric log levels without string coercion.

Changes:
1. `RuntimeConfig.log_level` type: `str | int`.
2. `normalize_log_level()` returns named string or numeric int.
3. `configure_logging()` accepts `str | int`.

### WP-8: MLOps Governance (Phase 1)

**Goal:** Reproducibility and auditability for every pipeline run.

Changes:
1. CI gates: `ruff check`, `mypy src`, schema validation, deterministic tests, perf budget.
2. Run metadata schema (`schemas/run_metadata.schema.json`).
3. Performance budget gate (`scripts/check_perf_budget.py`).
4. DVC lineage stages (`dvc.yaml`): features → events → report.
5. Optional MLflow lineage (`src/agri_auditor/mlops.py`): fire-and-forget, non-blocking.

### Phase 2 (Planned): Service Orchestration

Not implemented in this pass. Planned scope:
1. Thin FastAPI service for multi-run submission and status tracking.
2. Background worker queue for asynchronous feature/event/report jobs.
3. Artifact store abstraction for run outputs and lineage metadata.

## 5. Test Coverage Map

| Area | Tests | Passes |
|---|---|---|
| Report rendering + security | `tests/test_reporting.py` | 41 |
| Scoring + detection + Gemini resilience | `tests/test_intelligence.py` | 21 |
| Feature extraction + depth cache | `tests/test_features.py` | 17 |
| Config + log level validation | `tests/test_config_logging.py` | 9 |
| CLI argument parsing + fallback | `tests/test_cli.py` | 5 |
| Run metadata schema | `tests/test_run_metadata_schema.py` | 3 |
| Docker artifact structure | `tests/test_docker_artifacts.py` | varies |
| Performance budget enforcement | `tests/test_perf_budget.py` | varies |
| **Full deterministic suite** | `pytest -q` | **108 passed** |
| Live Gemini integration | `pytest -m gemini_live` | 3 (conditional) |

## 6. Verification Commands

```powershell
# Focused test lanes
pytest -q tests/test_reporting.py
pytest -q tests/test_intelligence.py
pytest -q tests/test_features.py
pytest -q tests/test_config_logging.py
pytest -q tests/test_cli.py
pytest -q tests/test_run_metadata_schema.py

# Full deterministic suite
pytest -q

# CI-equivalent local gates
ruff check .
mypy src
python scripts/check_perf_budget.py
```

## 7. Definition of Done

All items true before merge:
1. Security hardening complete — no untrusted model text enters JS execution context.
2. Report payload scales — split mode externalizes assets, single mode self-contained.
3. Gemini resilience — jitter, timeout, circuit, cache all tested.
4. Depth extraction parallelized — `ProcessPoolExecutor` active, Parquet cache verified.
5. Event scoring quality regression-guarded.
6. Packaging hygiene verified — optional extras work independently.
7. Numeric log levels stored as `int`.
8. MLOps phase 1 complete — CI gates, schema, DVC stages, MLflow hook.
9. Deterministic test suite fully green (108 passed).
10. Documentation synced.

