# Agri-Auditor Intelligence Roadmap

## CTO Upgrade Completion Snapshot

Last updated: February 22, 2026
Canonical source: `high level discussion/intelligence_roadmap_1`
Repo mirror: `docs/intelligence_roadmap.md`
Execution contract: `docs/cto_upgrade_completion_handoff.md` (dated February 21, 2026)

---

## 1) Completion Scope

This update closes the CTO Story-First Upgrade Portfolio completion pass for points 1 through 8.1, with point 8.2 explicitly staged as the next phase.

Non-negotiable ordering that was enforced:
1. Security and correctness.
2. Performance and scalability.
3. Operational maturity and team readiness.
4. Deterministic testability and CI gating.

---

## 2) Implemented vs Next Phase

| Point | Contract | Status (2026-02-22) | Validated implementation evidence |
|---|---|---|---|
| 1 | `1.1 + 1.2` security hardening | Implemented | JSON payload script blocks + `JSON.parse` bootstrap + `sanitize_model_text` caption sanitation in `src/agri_auditor/reporting.py`; security tests in `tests/test_reporting.py` |
| 2 | `2.1 + 2.2` report payload scale | Implemented | Split-mode external JSON assets + image materialization under `*_assets/images/`; single-mode self-contained behavior in `src/agri_auditor/reporting.py`; parity/size tests in `tests/test_reporting.py` |
| 3 | `3.1 + 3.3` Gemini SLA and jitter | Implemented | Jittered exponential backoff + SDK timeout wrapper + preserved circuit/cache behavior in `src/agri_auditor/intelligence.py`; resilience tests in `tests/test_intelligence.py` |
| 4 | `4.1 + 4.3` depth multiprocess + persistent cache | Implemented | Process-pool execution path (with controlled fallback when process spawn is blocked) + Parquet persistent cache with atomic replacement and in-memory hot cache in `src/agri_auditor/features.py`; new coverage in `tests/test_features.py` |
| 5 | `5.1` event quality stability | Implemented | Robust normalization and peak constraints preserved as defaults; regression coverage retained in `tests/test_intelligence.py` |
| 6 | `6.1 + 6.2` packaging/import hygiene | Implemented | Optional report dependency split/lazy import preserved; CLI behavior regression coverage in `tests/test_cli.py` |
| 7 | `7.1` numeric log-level type exactness | Implemented | Numeric log level persists as `int` in runtime config and logging configuration accepts `str | int`; coverage in `tests/test_config_logging.py` |
| 8 | `8.1` MLOps hardening for this phase | Implemented | CI gates expanded (lint/type/schema/tests/perf), metadata schema added, perf gate script added, DVC lineage stages added, optional MLflow lineage hook added |
| 8.2 | Service/API orchestration extension | Staged next phase (not implemented in this pass) | Scope preserved as next increment: thin FastAPI + worker queue + artifact store |

---

## 3) New MLOps and Governance Artifacts (8.1)

Added in this completion pass:
1. Run metadata schema: `schemas/run_metadata.schema.json`
2. Schema tests: `tests/test_run_metadata_schema.py`
3. Perf budget gate: `scripts/check_perf_budget.py`
4. Perf gate tests: `tests/test_perf_budget.py`
5. DVC lineage stages: `dvc.yaml`, `.dvc/config`, `.dvc/.gitignore`
6. Optional MLflow lineage logging (no hard fail when disabled/unavailable): `src/agri_auditor/mlops.py`
7. CI gate expansion: `.github/workflows/ci.yml` now runs `ruff`, `mypy`, metadata schema lane, deterministic tests, and perf budget gate.

---

## 4) Deterministic Validation Evidence (February 22, 2026)

Focused lanes:
1. `pytest -q tests/test_reporting.py` -> `41 passed`
2. `pytest -q tests/test_intelligence.py` -> `21 passed`
3. `pytest -q tests/test_features.py` -> `17 passed`
4. `pytest -q tests/test_config_logging.py` -> `9 passed`
5. `pytest -q tests/test_cli.py` -> `5 passed`
6. `pytest -q tests/test_run_metadata_schema.py` -> `3 passed`

Full deterministic lane:
1. `pytest -q` -> `108 passed, 3 deselected` (`gemini_live` intentionally deselected)

Local CI-equivalent gates:
1. `ruff check .` -> pass
2. `mypy src` -> pass
3. `python scripts/check_perf_budget.py` -> pass

---

## 5) Operational Notes

1. Process-pool depth extraction is active when environment permits process creation; controlled fallback to in-process execution is kept to preserve deterministic behavior in restricted sandboxes.
2. Run metadata contract (`dataset_hash`, `code_version`, `config_fingerprint`, `latency_summary`) is now schema-validated in test and CI gates.
3. MLflow lineage is optional and non-blocking by design to avoid hard runtime failures when MLflow is not installed or disabled.

---

## 6) Next-Phase Staging (8.2 Only)

This completion pass does not implement 8.2 runtime orchestration.

Planned 8.2 extension:
1. Thin FastAPI service for multi-run submission.
2. Background worker queue for asynchronous feature/event/report jobs.
3. Artifact store abstraction for run outputs and lineage metadata.

