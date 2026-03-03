# Technical Roadmap

Last updated: February 22, 2026

---

## Overview

This document tracks the phased implementation status of Agri Auditor's core subsystems: security hardening, report scalability, API resilience, compute performance, scoring quality, packaging, observability, and MLOps governance.

Priorities enforced throughout:
1. Security and correctness first.
2. Measurable performance and scalability second.
3. Operational maturity and testability third.

---

## Implementation Status

| Area | Scope | Status | Key Evidence |
|---|---|---|---|
| Security hardening | JSON payload isolation, CSP nonce enforcement, `sanitize_model_text` caption sanitation | **Complete** | `src/agri_auditor/reporting.py`; XSS/script-break regression tests in `tests/test_reporting.py` |
| Report scalability | Split-mode external JSON + image assets; single-mode self-contained; payload downsample controls | **Complete** | `src/agri_auditor/reporting.py`; split vs single parity/size tests |
| Gemini resilience | Jittered exponential backoff, per-call SDK timeout wrapper, circuit breaker, SHA-256 deterministic cache | **Complete** | `src/agri_auditor/intelligence.py`; retry/timeout/circuit/cache tests in `tests/test_intelligence.py` |
| Depth performance | `ProcessPoolExecutor` extraction, Parquet persistent cache with atomic writes, in-memory hot cache | **Complete** | `src/agri_auditor/features.py`; process-pool and cache-hit tests in `tests/test_features.py` |
| Event scoring quality | Robust MAD-based normalization, constrained peak selection with prominence/width/distance controls | **Complete** | `src/agri_auditor/intelligence.py`; regression coverage in `tests/test_intelligence.py` |
| Packaging hygiene | Optional report dependency split, lazy imports, graceful fallback hints | **Complete** | `pyproject.toml` optional extras; CLI fallback tests in `tests/test_cli.py` |
| Log-level correctness | Numeric log level stored as `int` in `RuntimeConfig`; `str \| int` accepted throughout | **Complete** | `src/agri_auditor/config.py`, `logging_config.py`; tests in `tests/test_config_logging.py` |
| MLOps governance (phase 1) | CI gates (lint/type/schema/tests/perf), run metadata schema, DVC lineage, optional MLflow hook | **Complete** | `.github/workflows/ci.yml`, `schemas/run_metadata.schema.json`, `dvc.yaml`, `src/agri_auditor/mlops.py` |
| **Service orchestration (phase 2)** | FastAPI submission layer, async worker queue, artifact store | **Next** | Planned; scoped but not implemented |

---

## MLOps and Governance Artifacts

| Artifact | Path |
|---|---|
| Run metadata schema (JSON Schema 2020-12) | `schemas/run_metadata.schema.json` |
| Schema validation tests | `tests/test_run_metadata_schema.py` |
| Performance budget gate | `scripts/check_perf_budget.py` |
| Performance gate tests | `tests/test_perf_budget.py` |
| DVC reproducibility stages | `dvc.yaml`, `.dvc/config` |
| Optional MLflow lineage (non-blocking) | `src/agri_auditor/mlops.py` |
| CI pipeline | `.github/workflows/ci.yml` — ruff, mypy, schema, tests, perf gate, conditional Gemini live |

---

## Test Validation Summary

Focused lanes:

| Lane | Result |
|---|---|
| `tests/test_reporting.py` | 41 passed |
| `tests/test_intelligence.py` | 21 passed |
| `tests/test_features.py` | 17 passed |
| `tests/test_config_logging.py` | 9 passed |
| `tests/test_cli.py` | 5 passed |
| `tests/test_run_metadata_schema.py` | 3 passed |
| **Full suite** | **108 passed**, 3 deselected (`gemini_live`) |

CI-equivalent local gates: `ruff check .`, `mypy src`, `scripts/check_perf_budget.py` — all pass.

---

## Operational Notes

1. **Depth extraction fallback** — Process-pool depth extraction is active when the environment permits process creation; controlled in-process fallback preserves deterministic behavior in restricted sandboxes (e.g., CI runners).
2. **Run metadata contract** — `dataset_hash`, `code_version`, `config_fingerprint`, and `latency_summary` are schema-validated in both tests and CI gates.
3. **MLflow** — Optional and non-blocking by design; never fails the pipeline on logging errors.

---

## Next Phase: Service Orchestration

Planned scope for the service layer extension:
1. Thin FastAPI service for multi-run submission and status tracking.
2. Background worker queue for asynchronous feature/event/report jobs.
3. Artifact store abstraction for run outputs and lineage metadata.

