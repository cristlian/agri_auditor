# CTO Upgrade Completion Handoff

Last updated: February 21, 2026  
Scope: `agri_auditor` repo

## 1) Mission

Finish the CTO Story-First Upgrade Portfolio so that the result is not "mostly done" but "all tasks exactly as planned".

Non-negotiable end state:
1. Security and correctness first.
2. Measurable performance and scalability second.
3. Operational maturity and team readiness third.
4. Full deterministic test suite green.

## 2) Canonical Upgrade Plan (Locked)

Chosen combination:
1. `1.1 + 1.2`
2. `2.1 + 2.2`
3. `3.1 + 3.3`
4. `4.1 + 4.3`
5. `5.1`
6. `6.1 + 6.2`
7. `7.1`
8. `8.1` with `8.2` explicitly staged next

Public interface requirements:
1. `--report-mode single|split` + payload reduction controls.
2. Gemini runtime knobs: workers, retries, backoff, cache dir.
3. Scoring knobs: robust scaling + prominence/width/min-distance.
4. Report dependencies as optional extra + lazy import.
5. Run metadata schema: dataset hash, code version, config fingerprint, latency summary.

## 3) Current Implementation Snapshot

## 3.1 Implemented Well

1. Optional report dependency split and lazy imports:
   - `pyproject.toml:20`
   - `src/agri_auditor/__init__.py:35`
   - `src/agri_auditor/cli.py:45`
2. Robust scoring and peak controls:
   - `src/agri_auditor/intelligence.py:152`
   - `src/agri_auditor/intelligence.py:212`
   - `src/agri_auditor/intelligence.py:382`
3. Gemini reliability baseline (workers/retries/backoff/circuit/cache):
   - `src/agri_auditor/intelligence.py:584`
   - `src/agri_auditor/intelligence.py:604`
   - `src/agri_auditor/intelligence.py:906`
   - `src/agri_auditor/intelligence.py:990`
   - `src/agri_auditor/intelligence.py:1149`
4. Report split mode + downsample + feature-column selection:
   - `src/agri_auditor/reporting.py:304`
   - `src/agri_auditor/reporting.py:360`
   - `src/agri_auditor/reporting.py:380`
   - `src/agri_auditor/reporting.py:385`
5. CSP and escaping baseline:
   - `src/agri_auditor/reporting.py:344`
   - `src/agri_auditor/reporting.py:703`
   - `src/agri_auditor/reporting.py:1093`
6. Run metadata fields:
   - `src/agri_auditor/cli.py:197`
   - `src/agri_auditor/cli.py:392`
   - `src/agri_auditor/cli.py:610`
7. Current deterministic tests:
   - `pytest -q` -> `94 passed, 3 deselected`

## 3.2 Gaps Blocking "Exactly as Planned"

1. Point 1 (`1.1 + 1.2`) is partial:
   - Data still injected as inline JS vars, not JSON script blocks:
     - `src/agri_auditor/reporting.py:1118`
     - `src/agri_auditor/reporting.py:1119`
     - `src/agri_auditor/reporting.py:1120`
     - `src/agri_auditor/reporting.py:1121`
   - No explicit Gemini plain-text sanitization function before template binding.
2. Point 2 (`2.1 + 2.2`) is partial:
   - Duplicate embedded images still inflate payload:
     - Event payload includes base64 `primary_image`: `src/agri_auditor/reporting.py:681`
     - Template repeats same image: `src/agri_auditor/reporting.py:1056`, `src/agri_auditor/reporting.py:1073`
3. Point 3 (`3.1 + 3.3`) is partial:
   - No jitter in exponential backoff (`src/agri_auditor/intelligence.py:604`).
   - No explicit SDK-side timeout wrapper (REST has timeout at `src/agri_auditor/intelligence.py:767`; SDK call at `src/agri_auditor/intelligence.py:699` does not).
4. Point 4 (`4.1 + 4.3`) is partial:
   - Depth parallelism is thread-based, not multiprocess:
     - `src/agri_auditor/features.py:297`
   - Persistent cache is JSON files, not Parquet:
     - `src/agri_auditor/features.py:328`
     - `src/agri_auditor/features.py:352`
5. Point 7 (`7.1`) is partial:
   - Numeric log levels accepted but config still stores `str`, not `int`:
     - `src/agri_auditor/config.py:35`
     - `src/agri_auditor/config.py:56`
6. Point 8 (`8.1`) is major partial:
   - CI currently runs tests + gemini-live gate, but no lint/type/perf budget gates and no DVC/MLflow lineage plumbing:
     - `.github/workflows/ci.yml:28`
     - `.github/workflows/ci.yml:65`

## 4) Exact Completion Work Packages

## WP-1 Security Hardening Completion (Point 1 exact)

Goal: fully satisfy `1.1 + 1.2`.

Required changes:
1. Replace inline data var injection with `<script type="application/json" id="...">` blocks.
2. Parse via `textContent` + `JSON.parse` in JS bootstrap.
3. Add explicit sanitizer for Gemini captions before rendering.
4. Keep CSP nonce policy and autoescape active.

Files:
1. `src/agri_auditor/reporting.py`
2. `tests/test_reporting.py`

Implementation notes:
1. Add helper `sanitize_model_text(value: str) -> str`:
   - Strip control chars except `\n`, `\t`.
   - Normalize whitespace.
   - Enforce max length.
2. Apply sanitizer in `_event_ctx` for `gemini_caption` before template.
3. Emit payload blocks:
   - `payload-events`, `payload-gps-path`, `payload-features`, `payload-telemetry`.
4. In split mode, keep JSON blocks minimal (metadata only), load large JSON from assets.

Tests to add/update:
1. Assert JSON script blocks exist and inline JS data vars are removed.
2. Assert script-break payload `</script><script>alert(1)</script>` is inert in output.
3. Assert sanitizer strips disallowed control chars.

Acceptance:
1. No untrusted model text enters JS execution context.
2. No executable payload from Gemini text under report rendering tests.

## WP-2 Report Payload Scale Completion (Point 2 exact)

Goal: fully satisfy `2.1 + 2.2`.

Required changes:
1. Remove duplicate image embedding paths.
2. Keep `single` demo mode portable.
3. Make `split` mode production-scalable with external JSON and image assets.

Files:
1. `src/agri_auditor/reporting.py`
2. `tests/test_reporting.py`
3. `tests/test_cli.py`

Implementation notes:
1. In split mode, write image assets to `*_assets/images/` and reference relative paths.
2. In single mode, avoid duplicating same image data in multiple fields.
3. Ensure map overlay and event cards reuse event payload references, not duplicated blobs.

Tests to add/update:
1. Compare split vs single report size on same synthetic run.
2. Assert split assets include JSON + image files.
3. Assert UI loads first frame and events in split mode.

Acceptance:
1. Split mode produces materially smaller HTML and externalized assets.
2. Single mode remains fully self-contained.

## WP-3 Gemini SLA/Jitter Completion (Point 3 exact)

Goal: fully satisfy `3.1 + 3.3`.

Required changes:
1. Add jittered exponential backoff.
2. Add explicit per-call SLA timeout wrapper for SDK path.
3. Preserve circuit breaker + cache determinism.

Files:
1. `src/agri_auditor/intelligence.py`
2. `src/agri_auditor/config.py`
3. `src/agri_auditor/cli.py` (only if exposing jitter knob)
4. `tests/test_intelligence.py`

Implementation notes:
1. Add config/env for jitter ratio (for example `AGRI_AUDITOR_GEMINI_JITTER_RATIO`, default `0.2`).
2. Backoff formula:
   - `sleep = base * 2^attempt * jitter_factor`
   - `jitter_factor` sampled in bounded range, clamped non-negative.
3. SDK timeout:
   - Wrap SDK call with timeout guard and convert to retryable timeout exception.

Tests to add/update:
1. Retry path for `429` and `5xx` with jittered sleeps (mock sleep and random).
2. SDK timeout triggers retries then failure.
3. Cache hit still bypasses provider calls.
4. Circuit breaker still opens after threshold.

Acceptance:
1. Backoff includes jitter.
2. SDK and REST both enforce per-call SLA behavior.

## WP-4 Depth Multiprocess + Parquet Cache Completion (Point 4 exact)

Goal: fully satisfy `4.1 + 4.3`.

Required changes:
1. Replace thread pool with process pool for depth extraction.
2. Replace JSON depth cache with Parquet-backed cache.

Files:
1. `src/agri_auditor/features.py`
2. `src/agri_auditor/config.py` (if needed for cache format controls)
3. `pyproject.toml` (add Parquet engine dependency if needed)
4. `tests/test_features.py`

Implementation notes:
1. Introduce top-level worker function for process-safe pickling.
2. Cache schema columns:
   - `cache_key`, `frame_idx`, `min_clearance_m`, `canopy_density_proxy`, `mtime_ns`, `size_bytes`, `params_hash`.
3. Use atomic write strategy for cache file replacement.
4. Keep in-memory hot cache for current run.

Tests to add/update:
1. Verify process pool path executes when `depth_workers > 1`.
2. Verify warm Parquet cache avoids image decoding.
3. Verify cache invalidates when depth file metadata changes.

Acceptance:
1. Multiprocess execution path active.
2. Persistent cache is Parquet, not per-frame JSON files.

## WP-5 Event Quality Stability (Point 5)

Status: already implemented.  
Action: keep as-is, only guard against regressions.

Files:
1. `tests/test_intelligence.py`

Acceptance:
1. Robust normalization and peak constraints remain default and tested.

## WP-6 Packaging/Import Hygiene (Point 6)

Status: already implemented.  
Action: regression checks only.

Files:
1. `tests/test_cli.py`
2. Optionally add dedicated import smoke tests.

Acceptance:
1. `features` and `events` run without report extras.
2. Report commands fail gracefully with install hint when extras missing.

## WP-7 Numeric Log Level Type Completion (Point 7 exact)

Goal: fully satisfy `7.1` exactly.

Required changes:
1. Store numeric log level as `int` in `RuntimeConfig`, not string.

Files:
1. `src/agri_auditor/config.py`
2. `src/agri_auditor/logging_config.py`
3. `tests/test_config_logging.py`

Implementation notes:
1. Change `RuntimeConfig.log_level` type to `str | int`.
2. `normalize_log_level` returns named string or numeric int.
3. `configure_logging` accepts `str | int`.

Acceptance:
1. Numeric env values propagate as ints to logger configuration.
2. Named values still supported.

## WP-8 MLOps Hardening Completion (Point 8 exact for this phase)

Goal: fully satisfy `8.1` and explicitly stage `8.2`.

Required changes for 8.1:
1. Add lineage plumbing (DVC and MLflow local path).
2. Add lint/type gates in CI.
3. Add performance budget gate in CI.
4. Add explicit run metadata schema validation gate.

Files:
1. `.github/workflows/ci.yml`
2. `pyproject.toml`
3. `src/agri_auditor/cli.py` (if extending metadata payload)
4. `schemas/run_metadata.schema.json` (new)
5. `tests/test_run_metadata_schema.py` (new)
6. `scripts/check_perf_budget.py` (new)
7. `dvc.yaml` (new), plus minimal DVC config files
8. `src/agri_auditor/mlops.py` (new, optional-no-op if mlflow unavailable)

Implementation notes:
1. CI jobs:
   - lint (`ruff check`)
   - type (`mypy src`)
   - deterministic tests
   - perf budget script
2. Perf budget:
   - stable synthetic workload
   - enforce upper bound and warm-cache speedup threshold
3. Metadata schema:
   - enforce required keys and value types (`dataset_hash`, `code_version`, `config_fingerprint`, `latency_summary`).
4. MLflow:
   - optional local logging, no hard fail when disabled.
5. DVC:
   - define reproducible stages for features/events/report artifacts.

Explicit next-phase note for 8.2:
1. Add thin FastAPI + worker queue + artifact store for multi-run orchestration.
2. Do not implement full 8.2 in this completion pass; only stage plan and interfaces.

Acceptance:
1. CI contains lint, type, perf budget, and metadata schema gates.
2. Lineage tooling files exist and are executable in local/dev environments.

## 5) Required Test Expansion Before Final Claim

Add or update tests for:
1. Security:
   - JSON script block embedding and sanitization.
   - XSS/script-break payload inertness.
2. Reporting scale:
   - split asset parity and size reduction.
   - image asset externalization in split mode.
3. Gemini resilience:
   - `429`, `5xx`, timeout, retry/jitter, breaker, cache.
4. Depth performance:
   - multiprocess execution and warm-cache speedup path.
5. Config:
   - numeric log level stored as int.
6. CI/MLOps:
   - metadata schema validation.
   - perf budget pass/fail behavior.

## 6) Command Checklist for Completion

Run in this order:

```powershell
# 1) Focused test lanes while implementing
pytest -q tests/test_reporting.py
pytest -q tests/test_intelligence.py
pytest -q tests/test_features.py
pytest -q tests/test_config_logging.py
pytest -q tests/test_cli.py

# 2) New MLOps tests
pytest -q tests/test_run_metadata_schema.py

# 3) Full deterministic suite
pytest -q

# 4) Local CI-equivalent checks
ruff check .
mypy src
python scripts/check_perf_budget.py
```

Target:
1. No test deselections in deterministic lane except intentional `gemini_live`.
2. All enabled checks pass.

## 7) Documentation Sync Requirements

After code completion:
1. Update `high level discussion/intelligence_roadmap_1` in-place using only validated content.
2. Mirror same substance to `docs/intelligence_roadmap.md`.
3. Use explicit dates and internally consistent status lines.
4. Add final "Implemented vs Next Phase" table:
   - Points 1-8 exact status
   - 8.2 listed as staged next-phase extension

## 8) Definition of Done (Strict)

All items below must be true before claiming success:
1. Point 1 exact (`1.1 + 1.2`) complete.
2. Point 2 exact (`2.1 + 2.2`) complete.
3. Point 3 exact (`3.1 + 3.3`) complete.
4. Point 4 exact (`4.1 + 4.3`) complete.
5. Point 5 complete.
6. Point 6 complete.
7. Point 7 exact (`7.1`) complete with numeric level as int.
8. Point 8 exact (`8.1`) complete with `8.2` explicitly staged.
9. Deterministic suite fully green.
10. Roadmap documents synced and CTO-ready.

## 9) Paste-Ready New Session Instruction

Use this file as the execution contract.  
In the new session, enforce this directive:

1. Implement remaining gaps from sections 4 and 5 exactly.
2. Do not stop after partial fixes.
3. Keep iterating until all tests and gates pass.
4. Return final report with:
   - changed files
   - why each gap is now closed
   - proof commands and outcomes
   - explicit confirmation against each of the 8 points

